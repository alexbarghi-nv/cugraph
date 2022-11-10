# Copyright (c) 2022, NVIDIA CORPORATION.
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

# cuGraph or cuGraph-Service is required; each has its own version of
# import_optional and we need to select the correct one.
try:
    from cugraph_service_client.remote_graph_utils import import_optional
except ModuleNotFoundError:
    try:
        from cugraph.utilities.utils import import_optional
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "cuGraph extensions for PyG require cuGraph"
            "or cuGraph-Service to be installed."
        )

_transform_to_backend_dtype_1d = import_optional(
    "cugraph_service_client.remote_graph_utils._transform_to_backend_dtype_1d"
)
cudf = import_optional("cudf")
pandas = import_optional("pandas")


def call_cugraph_algorithm(name, graph, *args, backend="numpy", **kwargs):
    """
    Calls a cugraph algorithm for a remote, sg, or mg graph.
    Requires either cuGraph or cuGraph-Service to be installed.

    name : string
        The name of the cuGraph algorithm to run (i.e. uniform_neighbor_sample)
    graph : Graph (cuGraph) or RemoteGraph (cuGraph-Service)
        The graph to call the algorithm on.
    backend : ('cudf', 'pandas', 'cupy', 'numpy', 'torch', 'torch:<device>')
              [default = 'numpy']
        The backend where the algorithm results will be stored.  Only used
        if the graph is a remote graph.
    """

    if graph.is_remote():
        # If the graph is remote, cuGraph-Service must be installed
        # Therefore we do not explicitly check that it is available
        if name != "uniform_neighbor_sample":
            raise ValueError(
                f"cuGraph algorithm {name} is not yet supported for RemoteGraph"
            )
        else:
            # TODO eventually replace this with a "call_algorithm call"
            sample_result = graph._client.uniform_neighbor_sample(
                *args, **kwargs, graph_id=graph._graph_id
            )

            if backend == "cudf":
                df = cudf.DataFrame()
            elif backend == "pandas":
                df = pandas.DataFrame()
            else:
                # handle cupy, numpy, torch as dict of arrays/tensors
                df = {}

            # _transform_to_backend_dtype_1d handles array/Series conversion
            for k, v in sample_result.__dict__.items():
                df[k] = _transform_to_backend_dtype_1d(
                    v, series_name=k, backend=backend
                )

            return df

    # TODO check using graph property in a future PR
    elif graph.is_multi_gpu():
        import cugraph.dask

        return getattr(cugraph.dask, name)(graph, *args, **kwargs)

    # TODO check using graph property in a future PR
    else:
        import cugraph

        return getattr(cugraph, name)(graph, *args, **kwargs)
