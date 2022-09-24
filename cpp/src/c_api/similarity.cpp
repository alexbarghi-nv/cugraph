/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cugraph_c/algorithms.h>

#include <c_api/abstract_functor.hpp>
#include <c_api/graph.hpp>
#include <c_api/graph_functions.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace cugraph {
namespace c_api {

struct cugraph_similarity_result_t {
  cugraph_type_erased_device_array_t* similarity_coefficients_;
};

}  // namespace c_api
}  // namespace cugraph

namespace {

struct similarity_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_;
  cugraph::c_api::cugraph_vertex_pairs_t const* vertex_pairs_;
  bool use_weight_;
  bool do_expensive_check_;
  cugraph::c_api::cugraph_similarity_result_t* result_{};

  similarity_functor(::cugraph_resource_handle_t const* handle,
                     ::cugraph_graph_t* graph,
                     ::cugraph_vertex_pairs_t const* vertex_pairs,
                     bool use_weight,
                     bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      vertex_pairs_(reinterpret_cast<cugraph::c_api::cugraph_vertex_pairs_t const*>(vertex_pairs)),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // similarity algorithms expect store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, weight_t, false, multi_gpu>*>(
          graph_->graph_);

      auto graph_view = graph->view();

      CUGRAPH_FAIL("Not implemented");
    }
  }
};

}  // namespace

extern "C" cugraph_type_erased_device_array_view_t* cugraph_similarity_result_get_similarity(
  cugraph_similarity_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_similarity_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->similarity_coefficients_->view());
}

extern "C" void cugraph_similarity_result_free(cugraph_similarity_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_similarity_result_t*>(result);
  delete internal_pointer->similarity_coefficients_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_jaccard_coefficients(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_vertex_pairs_t* vertex_pairs,
  bool_t use_weight,
  bool_t do_expensive_check,
  cugraph_similarity_result_t** result,
  cugraph_error_t** error)
{
  similarity_functor functor(handle, graph, vertex_pairs, use_weight, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_sorensen_coefficients(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_vertex_pairs_t* vertex_pairs,
  bool_t use_weight,
  bool_t do_expensive_check,
  cugraph_similarity_result_t** result,
  cugraph_error_t** error)
{
  similarity_functor functor(handle, graph, vertex_pairs, use_weight, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_error_code_t cugraph_overlap_coefficients(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_vertex_pairs_t* vertex_pairs,
  bool_t use_weight,
  bool_t do_expensive_check,
  cugraph_similarity_result_t** result,
  cugraph_error_t** error)
{
  similarity_functor functor(handle, graph, vertex_pairs, use_weight, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
