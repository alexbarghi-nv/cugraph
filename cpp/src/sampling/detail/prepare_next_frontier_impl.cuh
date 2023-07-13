/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <sampling/detail/sampling_utils.hpp>

#include <raft/core/handle.hpp>

#include <thrust/optional.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename label_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<label_t>>,
           std::optional<std::tuple<rmm::device_uvector<vertex_t>,
                                    std::optional<rmm::device_uvector<label_t>>>>>
prepare_next_frontier(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> sampled_src_vertices,
  std::optional<raft::device_span<label_t const>> sampled_src_vertex_labels,
  raft::device_span<vertex_t const> sampled_dst_vertices,
  std::optional<raft::device_span<label_t const>> sampled_dst_vertex_labels,
  std::optional<std::tuple<rmm::device_uvector<vertex_t>,
                           std::optional<rmm::device_uvector<label_t>>>>&& vertex_used_as_source,
  vertex_partition_view_t<vertex_t, multi_gpu> vertex_partition,
  std::vector<vertex_t> const& vertex_partition_range_lasts,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check)
{
  vertex_partition_device_view_t<vertex_t, multi_gpu> d_vertex_partition(vertex_partition);

  size_t frontier_size = sampled_dst_vertices.size();
  if (prior_sources_behavior == prior_sources_behavior_t::CARRY_OVER) {
    frontier_size += sampled_src_vertices.size();
  }

  rmm::device_uvector<vertex_t> frontier_vertices(frontier_size, handle.get_stream());
  auto frontier_vertex_labels =
    sampled_dst_vertex_labels
      ? std::make_optional<rmm::device_uvector<label_t>>(frontier_size, handle.get_stream())
      : std::nullopt;

  raft::copy(frontier_vertices.begin(),
             sampled_dst_vertices.data(),
             sampled_dst_vertices.size(),
             handle.get_stream());

  if (prior_sources_behavior == prior_sources_behavior_t::CARRY_OVER) {
    raft::copy(frontier_vertices.begin() + sampled_dst_vertices.size(),
               sampled_src_vertices.data(),
               sampled_src_vertices.size(),
               handle.get_stream());
  }

  if (frontier_vertex_labels) {
    raft::copy(frontier_vertex_labels->begin(),
               sampled_dst_vertex_labels->data(),
               sampled_dst_vertex_labels->size(),
               handle.get_stream());

    if (prior_sources_behavior == prior_sources_behavior_t::CARRY_OVER) {
      raft::copy(frontier_vertex_labels->begin() + sampled_dst_vertices.size(),
                 sampled_src_vertex_labels->data(),
                 sampled_src_vertex_labels->size(),
                 handle.get_stream());
    }
  }

  if constexpr (multi_gpu) {
    if (frontier_vertex_labels) {
      std::tie(frontier_vertices, *frontier_vertex_labels) =
        shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
          handle,
          std::move(frontier_vertices),
          std::move(*frontier_vertex_labels),
          vertex_partition_range_lasts);
    } else {
      frontier_vertices = shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
        handle, std::move(frontier_vertices), vertex_partition_range_lasts);
    }
  }

  if (frontier_vertex_labels) {
    auto begin_iter =
      thrust::make_zip_iterator(frontier_vertices.begin(), frontier_vertex_labels->begin());
    thrust::sort(handle.get_thrust_policy(), begin_iter, begin_iter + frontier_vertices.size());
  } else {
    thrust::sort(handle.get_thrust_policy(), frontier_vertices.begin(), frontier_vertices.end());
  }

  if (vertex_used_as_source) {
    auto& [verts, labels] = *vertex_used_as_source;

    // add sources from this expansion to the vertex_used_as_source
    size_t current_verts_size = verts.size();
    size_t new_verts_size     = current_verts_size + sampled_src_vertices.size();

    verts.resize(new_verts_size, handle.get_stream());

    raft::copy(verts.begin() + current_verts_size,
               sampled_src_vertices.data(),
               sampled_src_vertices.size(),
               handle.get_stream());

    // sort and unique the vertex_used_as_source structures
    if (sampled_src_vertex_labels) {
      labels->resize(new_verts_size, handle.get_stream());

      raft::copy(labels->begin() + current_verts_size,
                 sampled_src_vertex_labels->data(),
                 sampled_src_vertex_labels->size(),
                 handle.get_stream());

      auto begin_iter = thrust::make_zip_iterator(verts.begin(), labels->begin());

      thrust::sort(handle.get_thrust_policy(), begin_iter, begin_iter + new_verts_size);

      auto end_iter =
        thrust::unique(handle.get_thrust_policy(), begin_iter, begin_iter + new_verts_size);

      verts.resize(thrust::distance(begin_iter, end_iter), handle.get_stream());
      labels->resize(thrust::distance(begin_iter, end_iter), handle.get_stream());
    } else {
      thrust::sort(handle.get_thrust_policy(), verts.begin(), verts.end());

      auto end_iter = thrust::unique(handle.get_thrust_policy(), verts.begin(), verts.end());

      verts.resize(thrust::distance(verts.begin(), end_iter), handle.get_stream());
    }

    // Now with the updated verts/labels we can filter the next frontier
    std::tie(frontier_vertices, frontier_vertex_labels) = remove_visited_vertices_from_frontier(
      handle,
      std::move(frontier_vertices),
      std::move(frontier_vertex_labels),
      raft::device_span<vertex_t const>{verts.data(), verts.size()},
      labels ? std::make_optional(raft::device_span<label_t const>{labels->data(), labels->size()})
             : std::nullopt);
  }

  if (dedupe_sources) {
    if (frontier_vertex_labels) {
      auto begin_iter =
        thrust::make_zip_iterator(frontier_vertices.begin(), frontier_vertex_labels->begin());

      auto new_end = thrust::unique(
        handle.get_thrust_policy(), begin_iter, begin_iter + frontier_vertices.size());

      frontier_vertices.resize(thrust::distance(begin_iter, new_end), handle.get_stream());
      frontier_vertex_labels->resize(thrust::distance(begin_iter, new_end), handle.get_stream());
    } else {
      auto new_end = thrust::unique(
        handle.get_thrust_policy(), frontier_vertices.begin(), frontier_vertices.end());

      frontier_vertices.resize(thrust::distance(frontier_vertices.begin(), new_end),
                               handle.get_stream());
    }
  }

  return std::make_tuple(std::move(frontier_vertices),
                         std::move(frontier_vertex_labels),
                         std::move(vertex_used_as_source));
}

}  // namespace detail
}  // namespace cugraph
