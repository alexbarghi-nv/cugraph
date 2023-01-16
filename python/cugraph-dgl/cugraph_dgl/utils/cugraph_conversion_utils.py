# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

# Utils to convert b/w dgl heterograph to cugraph GraphStore
from __future__ import annotations
from typing import Dict, Tuple, Union

import cudf
import dask_cudf
import dask.array as da
from dask.distributed import get_client
import numpy as np
import cupy as cp
from cugraph.utilities.utils import import_optional
from cugraph.gnn.dgl_extensions.dgl_uniform_sampler import src_n, dst_n

dgl = import_optional("dgl")
F = import_optional("dgl.backend")
torch = import_optional("torch")


# Feature Tensor to DataFrame Utils
def convert_to_column_major(t: torch.Tensor):
    return t.t().contiguous().t()


def create_ar_from_tensor(t: torch.Tensor):
    t = convert_to_column_major(t)
    if t.device.type == "cuda":
        ar = cp.as_array(t)
    else:
        ar = t.numpy()
    return ar


def _create_edge_frame(src_t: torch.Tensor, dst_t: torch.Tensor, single_gpu: bool):
    """
    Create a edge dataframe from src_t and dst_t
    """
    src_ar = create_ar_from_tensor(src_t)
    dst_ar = create_ar_from_tensor(dst_t)
    edge_ar = np.stack([src_ar, dst_ar], axis=1)
    edge_df = _create_df_from_edge_ar(edge_ar, single_gpu=single_gpu)
    edge_df = edge_df.rename(
        columns={edge_df.columns[0]: src_n, edge_df.columns[1]: dst_n}
    )
    return edge_df


def _create_df_from_edge_ar(ar, single_gpu=True):
    if not single_gpu:
        n_workers = len(get_client().scheduler_info()["workers"])
        n_partitions = n_workers * 2

    n_rows, _ = ar.shape
    if single_gpu:
        ar = cp.asarray(ar)
        df = cudf.DataFrame(data=ar)
    else:
        chunksize = (n_rows + n_partitions - 1) // n_partitions
        ar = da.from_array(ar, chunks=(chunksize, -1)).map_blocks(cp.asarray)
        df = ar.to_dask_dataframe()

    df = df.reset_index(drop=True)
    return df


def get_edges_dict_from_dgl_HeteroGraph(
    graph: dgl.DGLHeteroGraph, single_gpu: bool
) -> Dict[Tuple[str, str, str], Union[cudf.DataFrame, dask_cudf.DataFrame]]:
    etype_d = {}
    for can_etype in graph.canonical_etypes:
        src_t, dst_t = graph.edges(form="uv", etype=can_etype)
        etype_d[can_etype] = _create_edge_frame(src_t, dst_t, single_gpu)
    return etype_d


# def add_ndata_from_data_dict(gs: CuGraphStorage, ):
