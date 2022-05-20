# Copyright (c) 2019-2022, NVIDIA CORPORATION.
#
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
#

from pylibcugraph.experimental import (MGGraph,
                                       ResourceHandle,
                                       GraphProperties,
                                       bfs as pylibcugraph_bfs,
                                       )

from collections.abc import Iterable
from dask.distributed import wait, default_client
from cugraph.dask.common.input_utils import (get_distributed_data,
                                             get_vertex_partition_offsets)
import cugraph.dask.comms.comms as Comms
import cudf
import dask_cudf

def _call_plc_mg_bfs(
    sID,
    data,
    sources,
    depth_limit,
    src_col_name,
    dst_col_name,
    graph_properties,
    num_edges,
    direction_optimizing=False,
    do_expensive_check=False,
    return_predecessors=True):
    comms_handle = Comms.get_handle(sID)
    resource_handle = ResourceHandle(comms_handle.getHandle())

    srcs = cudf.Series(data[0][src_col_name], dtype='int32')
    dsts = cudf.Series(data[0][dst_col_name], dtype='int32')
    weights = cudf.Series(data[0]['value'], dtype='float32') \
        if 'value' in data[0].columns \
        else cudf.Series((srcs + 1) / (srcs + 1), dtype='float32')

    mg = MGGraph(
        resource_handle = resource_handle,
        graph_properties = graph_properties,
        src_array = srcs,
        dst_array = dsts,
        weight_array = None,
        store_transposed = False,
        num_edges = num_edges,
        do_expensive_check = do_expensive_check
    )

    print({
        'resource_handle': resource_handle,
        'mg': mg,
        'sources': f'type {type(sources)}',
        'weights': f'{weights}',
        'direction_optimizing': direction_optimizing,
        'depth_limit': depth_limit,
        'return_predecessors': return_predecessors,
        'do_expensive_check': do_expensive_check,
        'data': data,
        'sources': sources
    })

    res = \
        pylibcugraph_bfs(
            resource_handle,
            mg,
            cudf.Series(sources, dtype='int32'),
            direction_optimizing,
            depth_limit if depth_limit is not None else 0,
            return_predecessors,
            True
        )
    
    return res

def convert_to_cudf(cp_arrays):
    """
    create a cudf DataFrame from cupy arrays
    """
    cupy_distances, cupy_predecessors, cupy_vertices = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["distance"] = cupy_distances
    df["predecessor"] = cupy_predecessors
    return df

def bfs(input_graph,
        start,
        depth_limit=None,
        return_distances=True):
    """
    Find the distances and predecessors for a breadth first traversal of a
    graph.
    The input graph must contain edge list as  dask-cudf dataframe with
    one partition per GPU.

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph graph instance, should contain the connectivity information
        as dask cudf edge list dataframe(edge weights are not used for this
        algorithm).

    start : Integer
        Specify starting vertex for breadth-first search; this function
        iterates over edges in the component reachable from this node.

    depth_limit : Integer or None, optional (default=None)
        Limit the depth of the search

    return_distances : bool, optional (default=True)
        Indicates if distances should be returned

    Returns
    -------
    df : dask_cudf.DataFrame
        df['vertex'] gives the vertex id

        df['distance'] gives the path distance from the
        starting vertex (Only if return_distances is True)

        df['predecessor'] gives the vertex it was
        reached from in the traversal

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> import dask_cudf
    >>> # ... Init a DASK Cluster
    >>> #    see https://docs.rapids.ai/api/cugraph/stable/dask-cugraph.html
    >>> # Download dataset from https://github.com/rapidsai/cugraph/datasets/..
    >>> chunksize = dcg.get_chunksize(datasets_path / "karate.csv")
    >>> ddf = dask_cudf.read_csv(datasets_path / "karate.csv",
    ...                          chunksize=chunksize, delimiter=" ",
    ...                          names=["src", "dst", "value"],
    ...                          dtype=["int32", "int32", "float32"])
    >>> dg = cugraph.Graph(directed=True)
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
    ...                            edge_attr='value')
    >>> df = dcg.bfs(dg, 0)

    """

    client = default_client()

    input_graph.compute_renumber_edge_list(transposed=False, legacy_renum_only=True)
    ddf = input_graph.edgelist.edgelist_df
    vertex_partition_offsets = get_vertex_partition_offsets(input_graph)
    num_verts = vertex_partition_offsets.iloc[-1]

    graph_properties = GraphProperties(
        is_multigraph=False)

    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    src_col_name = input_graph.renumber_map.renumbered_src_col_name
    dst_col_name = input_graph.renumber_map.renumbered_dst_col_name

    start_list = cudf.Series(start)
    if input_graph.renumbered:
        start_list = input_graph.lookup_internal_vertex_id(
            start_list).compute()

    #count_src_results = client.map(count_src, start)
    #cg = client.gather(count_src_results)
    #if sum(cg) < original_start_len:
    #    raise ValueError('At least one start vertex provided was invalid')

    cupy_result = [client.submit(
              _call_plc_mg_bfs,
              Comms.get_session_id(),
              wf[1],
              start,
              depth_limit,
              src_col_name,
              dst_col_name,
              graph_properties,
              num_edges,
              False,
              True,
              return_distances,
              workers=[wf[0]])
              for idx, wf in enumerate(data.worker_to_parts.items())]
    wait(cupy_result)
    print('wait 1')
    print('wait 2')
    cudf_result = [client.submit(convert_to_cudf,
                                 cp_arrays)
                   for cp_arrays in cupy_result]
    print('wait 3')
    wait(cudf_result)
    print('wait 4')

    ddf = dask_cudf.from_delayed(cudf_result)
    print('wait 5')

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, 'vertex')
        ddf = input_graph.unrenumber(ddf, 'predecessor')
        ddf = ddf.fillna(-1)
    return ddf
