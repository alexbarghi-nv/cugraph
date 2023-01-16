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
import numpy as np
from cugraph.gnn.dgl_extensions.utils.sampling import eid_n
from cugraph.utilities.utils import import_optional, MissingModule

dgl = import_optional("dgl")
F = import_optional("dgl.backend")


def _assert_valid_canonical_etype(canonical_etype):
    if not _is_valid_canonical_etype:
        error_message = (
            f"Invalid canonical_etype {canonical_etype} "
            + "canonical etype should be is a string triplet (str, str, str)"
            + "for source node type, edge type and destination node type"
        )
        raise dgl.DGLError(error_message)


def _is_valid_canonical_etype(canonical_etype):
    if not isinstance(canonical_etype, tuple):
        return False

    if len(canonical_etype) != 3:
        return False

    for t in canonical_etype:
        if not isinstance(t, str):
            return False
    return True


def add_edge_ids_to_edges_dict(edge_data_dict, edge_id_offset_d):
    eids_data_dict = {}
    for etype, df in edge_data_dict.items():
        df[eid_n] = 1
        df[eid_n] = df[eid_n].cumsum()
        df[eid_n] = df[eid_n] + edge_id_offset_d[etype] - 1
        eids_data_dict[etype] = df
    return eids_data_dict


if isinstance(F, MissingModule):
    backend_dtype_to_np_dtype_dict = MissingModule("dgl")
else:
    backend_dtype_to_np_dtype_dict = {
        F.bool: np.bool,
        F.uint8: np.uint8,
        F.int8: np.int8,
        F.int16: np.int16,
        F.int32: np.int32,
        F.int64: np.int64,
        F.float16: np.float16,
        F.float32: np.float32,
        F.float64: np.float64,
    }
