from libc.stdint cimport uintptr_t
from libc.stdint cimport int64_t
from libc.limits cimport INT_MAX

from pylibcugraph.resource_handle cimport ResourceHandle
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_bfs,
    cugraph_paths_result_t,
    cugraph_paths_result_get_vertices,
    cugraph_paths_result_get_predecessors,
    cugraph_paths_result_get_distances,
    cugraph_paths_result_free,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_view_free,
)
from pylibcugraph._cugraph_c.resource_handle cimport (
    bool_t,
    data_type_id_t,
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    assert_CAI_type,
    get_c_type_from_numpy_type,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)

def bfs(ResourceHandle handle, _GPUGraph graph, 
        sources, bool_t direction_optimizing, int64_t depth_limit, 
        bool_t compute_predecessors, bool_t do_expensive_check):
    try:
        import cupy
    except ModuleNotFoundError:
        raise RuntimeError("bfs requires the cupy package, which could not "
                           "be imported")
    assert_CAI_type(sources, "sources")

    if depth_limit < 0:
        depth_limit = INT_MAX

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef uintptr_t cai_sources_ptr = \
        sources.__cuda_array_interface__["data"][0]
    cdef cugraph_type_erased_device_array_view_t* sources_view_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_sources_ptr,
            len(sources),
            get_c_type_from_numpy_type(sources.dtype))
    
    cdef cugraph_paths_result_t* result_ptr

    error_code = cugraph_bfs(
        c_resource_handle_ptr,
        c_graph_ptr,
        sources_view_ptr,
        direction_optimizing,
        depth_limit,
        compute_predecessors,
        do_expensive_check,
        &result_ptr,
        &error_ptr
    )
    assert_success(error_code, error_ptr, "cugraph_bfs")

    # deallocate the no-longer needed sources array
    cugraph_type_erased_device_array_view_free(sources_view_ptr)

    # Extract individual device array pointers from result
    cdef cugraph_type_erased_device_array_view_t* distances_ptr = \
        cugraph_paths_result_get_distances(result_ptr)

    cdef cugraph_type_erased_device_array_view_t* predecessors_ptr = \
        cugraph_paths_result_get_predecessors(result_ptr)
    
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_paths_result_get_vertices(result_ptr)

    # copy to cupy arrays
    cupy_distances = copy_to_cupy_array(c_resource_handle_ptr, distances_ptr)
    cupy_predecessors = copy_to_cupy_array(c_resource_handle_ptr, predecessors_ptr)
    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    
    # deallocate the no-longer needed result struct
    cugraph_paths_result_free(result_ptr)

    return (cupy_distances, cupy_predecessors, cupy_vertices)