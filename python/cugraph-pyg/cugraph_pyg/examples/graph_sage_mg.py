# Copyright (c) 2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a multi-GPU benchmark that assumes the data has already been
# processed using the BulkSampler.  This workflow WILL ONLY WORK when
# reading already-processed sampling results from disk.

import time
import argparse
import gc

import numpy as np
import torch

from torch_geometric.nn import CuGraphSAGEConv

import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as td
import torch.multiprocessing as tmp
from torch.nn.parallel import DistributedDataParallel as ddp

from typing import List


class CuGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(CuGraphSAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            conv = CuGraphSAGEConv(hidden_channels, hidden_channels)
            self.convs.append(conv)

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge, size):
        edge_csc = CuGraphSAGEConv.to_csc(edge, (size[0], size[0]))
        for conv in self.convs:
            x = conv(x, edge_csc)[: size[1]]
            x = F.relu(x)
            x = F.dropout(x, p=0.5)

        return self.lin(x)


def enable_cudf_spilling():
    import cudf

    cudf.set_option("spill", True)


def init_pytorch_worker(rank, devices, manager_ip, manager_port) -> None:
    import cupy
    import rmm

    device_id = devices[rank]

    rmm.reinitialize(
        devices=[device_id],
        pool_allocator=False,
    )

    torch.cuda.change_current_allocator(rmm.rmm_torch_allocator)
    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

    cupy.cuda.Device(device_id).use()
    torch.cuda.set_device(device_id)

    # Pytorch training worker initialization
    dist_init_method = f"tcp://{manager_ip}:{manager_port}"

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=dist_init_method,
        world_size=len(devices),
        rank=rank,
    )

    # enable_cudf_spilling()


def start_cugraph_dask_client(rank, dask_scheduler_file):
    print(
        "Connecting to dask... "
        "(warning: this may take a while depending on your configuration)"
    )
    from distributed import Client
    from cugraph.dask.comms import comms as Comms

    client = Client(scheduler_file=dask_scheduler_file)
    Comms.initialize(p2p=True)

    print(f"Successfully connected to dask on rank {rank}")
    return client


def stop_cugraph_dask_client():
    from cugraph.dask.comms import comms as Comms

    Comms.destroy()

    from dask.distributed import get_client

    get_client().close()


def train(
    rank,
    torch_devices: List[int],
    manager_ip: str,
    manager_port: int,
    dask_scheduler_file: str,
    num_epochs: int,
    features_on_gpu=True,
) -> None:
    """
    Parameters
    ----------
    device: int
        The CUDA device where the model, graph data, and node labels will be stored.
    features_on_gpu: bool
        Whether to store a replica of features on each worker's GPU.  If False,
        all features will be stored on the CPU.
    """

    world_size = len(torch_devices)
    device_id = torch_devices[rank]
    features_device = device_id if features_on_gpu else "cpu"
    init_pytorch_worker(rank, torch_devices, manager_ip, manager_port)
    td.barrier()

    start_cugraph_dask_client(rank, dask_scheduler_file)

    from distributed import Event as Dask_Event

    event = Dask_Event("cugraph_store_creation_event")

    td.barrier()

    import cugraph
    from cugraph_pyg.data import CuGraphStore
    from cugraph_pyg.loader import CuGraphNeighborLoader

    root = "/datasets/abarghi/ogbn_papers100M/converted/"
    import os

    with open(os.path.join(root, "ogbn_papers100M_meta.json"), "r") as f:
        import json

        meta = json.load(f)

    feature_data = np.fromfile(
        os.path.join(root, "ogbn_papers100M_node_feat_paper_part_0_of_1"),
        dtype="float32",
    )
    feature_data = feature_data.reshape(
        meta["nodes"][0]["num_nodes"], meta["nodes"][0]["emb_dim"]
    )

    G = np.fromfile(
        os.path.join(root, "ogbn_papers100M_edge_index_paper_cites_paper_part_0_of_1"),
        dtype="int32",
    )
    G = {
        ("paper", "cites", "paper"): G.reshape(2, meta["edges"][0]["num_edges"]).astype(
            "int64"
        )
    }

    N = {"paper": meta["nodes"][0]["num_nodes"]}

    fs = cugraph.gnn.FeatureStore(backend="torch")

    print(feature_data)
    print(features_device)
    fs.add_data(
        torch.as_tensor(feature_data, device=features_device),
        "paper",
        "x",
    )

    num_papers = N["paper"]

    with open(os.path.join(root, "ogbn_papers100M_data_and_label.pkl"), "rb") as f:
        import pickle

        label = pickle.load(f)

    y = torch.full((num_papers,), -1, dtype=torch.long)
    y[label["train_idx"]] = torch.as_tensor(label["train_label"].T[0], device="cpu").to(
        torch.long
    )
    y[label["test_idx"]] = torch.as_tensor(label["test_label"].T[0], device="cpu").to(
        torch.long
    )
    # y[label['val_idx']] = torch.as_tensor(label['val_label'].T[0], device='cpu').to(
    # torch.long
    # )

    fs.add_data(torch.as_tensor(y, device=device_id), "paper", "y")

    all_train_nodes = label["train_idx"]
    num_train_nodes_per_rank = len(all_train_nodes) // world_size
    train_nodes = all_train_nodes[
        num_train_nodes_per_rank * rank : num_train_nodes_per_rank * (rank + 1)
    ]

    train_mask = torch.full((num_papers,), 0, device=device_id)
    train_mask[train_nodes] = 1
    fs.add_data(train_mask, "paper", "train")

    print(f"Rank {rank} finished loading graph and feature data")

    if rank == 0:
        store_create_start_time = time.perf_counter_ns()
        cugraph_store = CuGraphStore(fs, G, N, multi_gpu=True)
        event.set()
        store_create_end_time = time.perf_counter_ns()
        print(
            "rank 0 created store in "
            f"{(store_create_end_time - store_create_start_time) / 1e9:3.4f} s"
        )
    else:
        if event.wait(timeout=1000):
            store_create_start_time = time.perf_counter_ns()
            cugraph_store = CuGraphStore(fs, G, N, multi_gpu=True)
            store_create_end_time = time.perf_counter_ns()
            print(
                f"Rank {rank} done with cugraph store creation, took "
                f"{(store_create_end_time - store_create_start_time) / 1e9:3.4f} s"
            )
        else:
            raise RuntimeError("timeout")

    print(f"rank {rank}: train {train_nodes.shape}")
    td.barrier()
    model = (
        CuGraphSAGE(
            in_channels=128, hidden_channels=256, out_channels=172, num_layers=3
        )
        .to(torch.float32)
        .to(device_id)
    )
    model = ddp(model, device_ids=[device_id], output_device=device_id)
    td.barrier()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        start_time_train = time.perf_counter_ns()
        model.train()

        cugraph_bulk_loader = CuGraphNeighborLoader(
            cugraph_store,
            train_nodes,
            batch_size=1024,
            num_neighbors=[30, 30, 30],
            seeds_per_call=1024,
            batches_per_partition=32,
        )

        total_loss = 0
        num_batches = 0

        for epoch in range(num_epochs):
            print(f"rank {rank} starting epoch {epoch}")
            with td.algorithms.join.Join([model]):
                for iter_i, hetero_data in enumerate(cugraph_bulk_loader):
                    num_batches += 1
                    if iter_i % 20 == 0:
                        print(f"iteration {iter_i}")

                    # train
                    train_mask = hetero_data.train_dict["paper"]
                    y_true = hetero_data.y_dict["paper"]

                    y_pred = model(
                        hetero_data.x_dict["paper"].to(device_id).to(torch.float32),
                        hetero_data.edge_index_dict[("paper", "cites", "paper")].to(
                            device_id
                        ),
                        (len(y_true), len(y_true)),
                    )

                    y_true = F.one_hot(
                        y_true[train_mask].to(torch.int64), num_classes=349
                    ).to(torch.float32)

                    y_pred = y_pred[train_mask]

                    loss = F.cross_entropy(y_pred, y_true)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    del y_true
                    del y_pred
                    del loss
                    del hetero_data
                    gc.collect()

                end_time_train = time.perf_counter_ns()
                print(
                    f"epoch {epoch} "
                    f"time: {(end_time_train - start_time_train) / 1e9:3.4f} s"
                )
                print(f"loss after epoch {epoch}: {total_loss / num_batches}")

    td.barrier()
    if rank == 0:
        print("DONE", flush=True)
        event.clear()

    td.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch_devices",
        type=str,
        default="0,1",
        help="GPU to allocate to pytorch for model, graph data, and node label storage",
        required=False,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
        required=False,
    )

    parser.add_argument(
        "--features_on_gpu",
        type=bool,
        default=False,
        help="Whether to store the features on each worker's GPU",
        required=False,
    )

    parser.add_argument(
        "--torch_manager_ip",
        type=str,
        default="127.0.0.1",
        help="The torch distributed manager ip address",
        required=False,
    )

    parser.add_argument(
        "--torch_manager_port",
        type=str,
        default="12346",
        help="The torch distributed manager port",
        required=False,
    )

    parser.add_argument(
        "--dask_scheduler_file",
        type=str,
        help="The path to the dask scheduler file",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    torch_devices = [int(d) for d in args.torch_devices.split(",")]

    train_args = (
        torch_devices,
        args.torch_manager_ip,
        args.torch_manager_port,
        args.dask_scheduler_file,
        args.num_epochs,
        args.features_on_gpu,
    )

    tmp.spawn(train, args=train_args, nprocs=len(torch_devices))


if __name__ == "__main__":
    main()
