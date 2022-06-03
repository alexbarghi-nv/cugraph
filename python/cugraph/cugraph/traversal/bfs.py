# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import cudf
import dask_cudf

from pylibcugraph import (ResourceHandle,
                          GraphProperties,
                          SGGraph,
                          )
from pylibcugraph import bfs as pylibcugraph_bfs

from cugraph.structure.graph_classes import Graph, DiGraph
from cugraph.utilities import (ensure_cugraph_obj,
                               is_matrix_type,
                               is_cp_matrix_type,
                               is_nx_graph_type,
                               cupy_package as cp,
                               )


def _ensure_args(G, start, i_start, directed):
    """
    Ensures the args passed in are usable for the API api_name and returns the
    args with proper defaults if not specified, or raises TypeError or
    ValueError if incorrectly specified.
    """
    # checks common to all input types
    if (start is not None) and (i_start is not None):
        raise TypeError("cannot specify both 'start' and 'i_start'")
    if (start is None) and (i_start is None):
        raise TypeError("must specify 'start' or 'i_start', but not both")

    start = start if start is not None else i_start

    G_type = type(G)
    # Check for Graph-type inputs
    if (G_type in [Graph, DiGraph]) or is_nx_graph_type(G_type):
        if directed is not None:
            raise TypeError("'directed' cannot be specified for a "
                            "Graph-type input")

        # ensure start vertex is valid
        invalid_vertex_err = ValueError('A provided vertex was not valid')
        if is_nx_graph_type(G_type):
            if start not in G:
                raise invalid_vertex_err
        else:
            if not isinstance(start, cudf.DataFrame):
                if not isinstance(start, dask_cudf.DataFrame):
                    start = cudf.DataFrame(
                        {'starts': cudf.Series(start)}
                    )

            if G.is_renumbered():
                validlen = len(
                    G.renumber_map.to_internal_vertex_id(
                        start,
                        start.columns
                    ).dropna()
                )
                if validlen < len(start):
                    raise invalid_vertex_err
            else:
                el = G.edgelist.edgelist_df[["src", "dst"]]
                col = start.columns[0]
                null_l = el \
                    .merge(
                        start[col].rename('src'),
                        on='src',
                        how='right'
                    ) \
                    .dst.isnull() \
                    .sum()
                null_r = el \
                    .merge(
                        start[col].rename('dst'),
                        on='dst',
                        how='right'
                    ) \
                    .src.isnull() \
                    .sum()
                if null_l + null_r > 0:
                    raise invalid_vertex_err

    if directed is None:
        directed = True

    return (start, directed)


def _convert_df_to_output_type(df, input_type):
    """
    Given a cudf.DataFrame df, convert it to a new type appropriate for the
    graph algos in this module, based on input_type.
    """
    if input_type in [Graph, DiGraph]:
        return df

    elif is_nx_graph_type(input_type):
        return df.to_pandas()

    elif is_matrix_type(input_type):
        # A CuPy/SciPy input means the return value will be a 2-tuple of:
        #   distance: cupy.ndarray
        #   predecessor: cupy.ndarray
        sorted_df = df.sort_values("vertex")
        if is_cp_matrix_type(input_type):
            distances = cp.fromDlpack(sorted_df["distance"].to_dlpack())
            preds = cp.fromDlpack(sorted_df["predecessor"].to_dlpack())
            return (distances, preds)
        else:
            distances = sorted_df["distance"].to_numpy()
            preds = sorted_df["predecessor"].to_numpy()
            return (distances, preds)
    else:
        raise TypeError(f"input type {input_type} is not a supported type.")


def _call_plc_bfs(G, sources, depth_limit, do_expensive_check=False,
                  direction_optimizing=False, return_predecessors=True):
    handle = ResourceHandle()

    srcs = G.edgelist.edgelist_df['src']
    dsts = G.edgelist.edgelist_df['dst']
    weights = G.edgelist.edgelist_df['weights'] \
        if 'weights' in G.edgelist.edgelist_df \
        else cudf.Series((srcs + 1) / (srcs + 1), dtype='float32')

    sg = SGGraph(
        resource_handle=handle,
        graph_properties=GraphProperties(is_multigraph=G.is_multigraph()),
        src_array=srcs,
        dst_array=dsts,
        weight_array=weights,
        store_transposed=False,
        renumber=False,
        do_expensive_check=do_expensive_check
    )

    distances, predecessors, vertices = \
        pylibcugraph_bfs(
            handle,
            sg,
            sources,
            direction_optimizing,
            depth_limit if depth_limit is not None else -1,
            return_predecessors,
            do_expensive_check
        )

    return cudf.DataFrame({
        'distance': cudf.Series(distances),
        'vertex': cudf.Series(vertices),
        'predecessor': cudf.Series(predecessors),
    })


def bfs(G,
        start=None,
        depth_limit=None,
        i_start=None,
        directed=None,
        return_predecessors=True):
    """
    Find the distances and predecessors for a breadth first traversal of a
    graph.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    start : Integer or list, optional (default=None)
        The id of the graph vertex from which the traversal begins, or
        if a list, the vertex from which the traversal begins in each
        component of the graph.  Only one vertex per connected
        component of the graph is allowed.

    depth_limit : Integer or None, optional (default=None)
        Limit the depth of the search

    i_start : Integer, optional (default=None)
        Identical to start, added for API compatibility. Only start or i_start
        can be set, not both.

    directed : bool, optional (default=None)
        NOTE
            For non-Graph-type (eg. sparse matrix) values of G only. Raises
            TypeError if used with a Graph object.

        If True, then convert the input matrix to a directed cugraph.Graph,
        otherwise an undirected cugraph.Graph object will be used.

    return_predecessors : bool, optional (default=True)
        Whether to return the predecessors for each vertex (returns -1
        for each vertex otherwise)

    Returns
    -------
    Return value type is based on the input type.  If G is a cugraph.Graph,
    returns:

       cudf.DataFrame
          df['vertex'] vertex IDs

          df['distance'] path distance for each vertex from the starting vertex

          df['predecessor'] for each i'th position in the column, the vertex ID
          immediately preceding the vertex at position i in the 'vertex' column

    If G is a networkx.Graph, returns:

       pandas.DataFrame with contents equivalent to the cudf.DataFrame
       described above.

    If G is a CuPy or SciPy matrix, returns:
       a 2-tuple of CuPy ndarrays (if CuPy matrix input) or Numpy ndarrays (if
       SciPy matrix input) representing:

       distance: cupy or numpy ndarray
          ndarray of shortest distances between source and vertex.

       predecessor: cupy or numpy ndarray
          ndarray of predecessors of a vertex on the path from source, which
          can be used to reconstruct the shortest paths.

       ...or if return_sp_counter is True, returns a 3-tuple with the above two
       arrays plus:

       sp_counter: cupy or numpy ndarray
          ndarray of number of shortest paths leading to each vertex.

    Examples
    --------
    >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
    >>> df = cugraph.bfs(G, 0)

    """
    (start, directed) = \
        _ensure_args(G, start, i_start, directed)

    # FIXME: allow nx_weight_attr to be specified
    (G, input_type) = ensure_cugraph_obj(
        G, nx_weight_attr="weight",
        matrix_graph_type=Graph(directed=directed)
    )

    # The BFS C++ extension assumes the start vertex is a cudf.Series object,
    # and operates on internal vertex IDs if renumbered.
    is_dataframe = isinstance(start, cudf.DataFrame) or \
        isinstance(start, dask_cudf.DataFrame)
    if G.renumbered is True:
        if is_dataframe:
            start = G.lookup_internal_vertex_id(start, start.columns)
        else:
            start = G.lookup_internal_vertex_id(cudf.Series(start))

    else:
        if is_dataframe:
            start = start[start.columns[0]]
        else:
            start = cudf.Series(start, name='starts')

    df = _call_plc_bfs(
        G,
        start,
        depth_limit,
        return_predecessors=return_predecessors
    )
    if G.renumbered:
        df = G.unrenumber(df, "vertex")
        df = G.unrenumber(df, "predecessor")
        df.fillna(-1, inplace=True)

    return _convert_df_to_output_type(df, input_type)


def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    """
    Find the distances and predecessors for a breadth first traversal of a
    graph.

    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    source : Integer
        The starting vertex index

    reverse : boolean, optional (default=False)
        If a directed graph, then process edges in a reverse direction
        Currently not implemented

    depth_limit : Int or None, optional (default=None)
        Limit the depth of the search

    sort_neighbors : None or Function, optional (default=None)
        Currently not implemented

    Returns
    -------
    Return value type is based on the input type.  If G is a cugraph.Graph,
    returns:

       cudf.DataFrame
          df['vertex'] vertex IDs

          df['distance'] path distance for each vertex from the starting vertex

          df['predecessor'] for each i'th position in the column, the vertex ID
          immediately preceding the vertex at position i in the 'vertex' column

    If G is a networkx.Graph, returns:

       pandas.DataFrame with contents equivalent to the cudf.DataFrame
       described above.

    If G is a CuPy or SciPy matrix, returns:
       a 2-tuple of CuPy ndarrays (if CuPy matrix input) or Numpy ndarrays (if
       SciPy matrix input) representing:

       distance: cupy or numpy ndarray
          ndarray of shortest distances between source and vertex.

       predecessor: cupy or numpy ndarray
          ndarray of predecessors of a vertex on the path from source, which
          can be used to reconstruct the shortest paths.

       ...or if return_sp_counter is True, returns a 3-tuple with the above two
       arrays plus:

       sp_counter: cupy or numpy ndarray
          ndarray of number of shortest paths leading to each vertex.

    Examples
    --------
    >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
    >>> df = cugraph.bfs_edges(G, 0)

    """

    if reverse is True:
        raise NotImplementedError(
            "reverse processing of graph is currently not supported"
        )

    return bfs(G, source, depth_limit)
