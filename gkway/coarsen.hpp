#pragma once
#include<iostream>
#include<algorithm>
#include"cuda_check.hpp"
#include <limits>
#include "../declarations.h"
#include "moderngpu/src/moderngpu/kernel_segsort.hxx"
#include "moderngpu/src/moderngpu/kernel_mergesort.hxx"
#include "moderngpu/src/moderngpu/kernel_scan.hxx"
#define MATCH_STEP_THREAD 32
#define WARP_SIZE 32
#define MATCH_STEP_WARP_PER_BLOCK 16
namespace gk { // begin of namespace gk ============================================

  // ======================================================
  //
  // Declaration/Definition of GPU kernels
  //
  // ======================================================
  //
  __device__
  long long get_value_combo(int val1, int val2) {
    long long combo = (long long) val1 << 32;
    combo += val2;
    return combo;
  }

  __device__
  unsigned get_first_val_from_value_combo(long long combo) {
    return (unsigned) (combo >> 32);
  }

  __device__
  unsigned get_second_val_from_val_combo(long long combo) {
    return combo & 0xFFFFFFFF;
  }


  __global__
  void matching_with_score(
    unsigned* d_vwgt,
    unsigned* d_adjp,
    unsigned* d_adjncy,
    unsigned* d_adjwgt,
    unsigned* d_hec_match_candidate,
    const unsigned NUM_VERTICES) {

    //each thread find the heaviest neighbir
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= NUM_VERTICES) {
      return;
    }
    unsigned neighbor_start = d_adjp[gid];
    unsigned neighbor_end = d_adjp[gid + 1];

    //loop through neighbor list to find the heaviest edge neighbor
    unsigned heaviest_edge_wgt = 0;
    unsigned heaviest_degree = 0;
    unsigned neighbor_with_heaviest_edge = 0;
    unsigned edge_wgt, current_neighbor, neighbor_degree;
    for(int i = neighbor_start; i < neighbor_end; i++) {
      edge_wgt = d_adjwgt[i];
      if(heaviest_edge_wgt < edge_wgt) {
        heaviest_edge_wgt = edge_wgt;
        neighbor_with_heaviest_edge = d_adjncy[i];
        heaviest_degree = d_adjp[neighbor_with_heaviest_edge] - d_adjp[neighbor_with_heaviest_edge - 1];
      }
      else if(heaviest_edge_wgt == edge_wgt) {
        current_neighbor = d_adjncy[i];
        neighbor_degree =  d_adjp[current_neighbor] - d_adjp[current_neighbor - 1];
        if(neighbor_degree < heaviest_degree) {
          heaviest_degree = neighbor_degree;
          neighbor_with_heaviest_edge = current_neighbor;
        }
      }
    }
    d_hec_match_candidate[gid] = neighbor_with_heaviest_edge;
 }

  __global__
  void HEC_union_find(
    unsigned* d_match_candidate,
    unsigned long long* d_group_id,
    unsigned* d_if_change,
    unsigned it,
    const unsigned NUM_VERTICES) {

    __shared__ int s_if_change[1];

    unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= NUM_VERTICES) {
      return;
    }

    long long new_group_id;
    s_if_change[0] = 0;

    __syncthreads();
    unsigned heaviest_neighbor = d_match_candidate[gid];
    unsigned nbr_group_id = get_first_val_from_value_combo(d_group_id[heaviest_neighbor - 1]);
    unsigned cur_group_id = get_first_val_from_value_combo(d_group_id[gid]);

    if(cur_group_id > nbr_group_id) {
      new_group_id = get_value_combo(cur_group_id, it);
      atomicMax(&d_group_id[heaviest_neighbor - 1], new_group_id);
      s_if_change[0] = 1;
    }
    else if(cur_group_id < nbr_group_id) {
      new_group_id = get_value_combo(nbr_group_id, it);
      atomicMax(&d_group_id[gid], new_group_id);
      s_if_change[0] = 1;
    }
    __syncthreads();

    if(threadIdx.x == 0 && s_if_change[0] == 1){
      atomicMax(&d_if_change[0], 1);
    }
  }

  __global__
  void count_num_coarsen_vertex(
    unsigned long long* d_group_id,
    unsigned* d_vertex_id,
    unsigned* d_group_head,
    const unsigned NUM_VERTICES){

    unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= NUM_VERTICES) {
      return;
    }

    unsigned group_id = get_first_val_from_value_combo(d_group_id[gid]);
    unsigned vertex_id = d_vertex_id[gid];
 
    if(vertex_id == group_id){
      d_group_head[gid] = 1;
    }
  }

  __global__
  void construct_cmap(
    unsigned* d_group_head,
    unsigned* d_vertex_id,
    unsigned* d_cmap,
    const unsigned NUM_VERTICES) {

    unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= NUM_VERTICES) {
      return;
    }
    unsigned new_id = d_group_head[gid];
    unsigned vertex_id = d_vertex_id[gid];
    //printf("gid:%d, new_id:%d, vertex_id;:%d \n", gid, new_id, vertex_id);
    d_cmap[vertex_id - 1] = new_id;
  }

  __global__
  void cal_coarsen_vwgt(
    unsigned* d_vwgt,
    unsigned* d_coarsen_vwgt,
    unsigned* d_cmap,
    const unsigned NUM_VERTICES) {

    unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= NUM_VERTICES) {
      return;
    }

    unsigned coarsen_id = d_cmap[gid];
    atomicAdd(&d_coarsen_vwgt[coarsen_id - 1], d_vwgt[gid]);
  }

  __global__
  void construct_tmp_adjp(
    unsigned* d_adjp,
    unsigned* d_tmp_adjp,
    unsigned* d_cmap,
    const unsigned NUM_VERTICES) {

    unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= NUM_VERTICES) {
      return;
    }
    //record maximal number of neighbirs for each coarsen vertex
    unsigned num_neighbor = d_adjp[gid + 1] - d_adjp[gid];
    atomicAdd(&d_tmp_adjp[d_cmap[gid]], num_neighbor);
  }

  __global__
  void copied_adjncy(
    unsigned* d_adjp,
    unsigned* d_max_neighbor_cnt,
    unsigned* d_tmp_adjp,
    unsigned* d_adjncy,
    unsigned* d_tmp_adjncy,
    unsigned* d_adjwgt,
    unsigned* d_tmp_adjwgt,
    unsigned* d_cmap,
    const unsigned NUM_VERTICES) {

    unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= NUM_VERTICES) {
      return;
    }
    unsigned coarsen_vertex = d_cmap[gid];
    int neighbor_start = d_adjp[gid];
    int neighbor_end = d_adjp[gid + 1];
    int num_neighbor = neighbor_end - neighbor_start;

    //int pos = atomicAdd(&d_max_neighbor_cnt[coarsen_vertex - 1], num_neighbor);
    int pos = atomicAdd(&d_max_neighbor_cnt[coarsen_vertex - 1], num_neighbor);
    for(int i = 0; i < num_neighbor; i++) {
      //d_tmp_adjncy[pos+i] = d_adjncy[neighbor_start+i];
      int neighbor = d_adjncy[neighbor_start + i] - 1;
      d_tmp_adjncy[pos + i] = d_cmap[neighbor];
      d_tmp_adjwgt[pos + i] = d_adjwgt[neighbor_start+i];
    }
  }

  __global__
  void construct_coarsen_graph(
    unsigned* d_coarsen_adjncy,
    unsigned* d_coarsen_adjwgt,
    unsigned* d_tmp_adjncy,
    unsigned* d_tmp_adjwgt,
    unsigned* d_coarsen_adjp,
    unsigned* d_tmp_adjp,
    unsigned num_coarsen_vertex) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= num_coarsen_vertex) {
      return;
    }

    int neighbor_start = d_coarsen_adjp[gid];
    int neighbor_end = d_coarsen_adjp[gid + 1];
    int num_neighbor = neighbor_end - neighbor_start;
    int old_neighbor_start = d_tmp_adjp[gid];

    for(int i = 0; i < num_neighbor; i++) {
      d_coarsen_adjncy[neighbor_start + i] = d_tmp_adjncy[old_neighbor_start + i];
      d_coarsen_adjwgt[neighbor_start + i] = d_tmp_adjwgt[old_neighbor_start + i];
    }
  }

  __global__
  void HEC_remove_duplicate_edge(
    unsigned* d_adjwgt,
    unsigned* d_tmp_adjp,
    unsigned* d_tmp_coarsen_adjncy,
    unsigned* d_tmp_coarsen_adjwgt,
    unsigned* d_coarsen_adjp,
    size_t num_coarsen_vertex) {

    unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= num_coarsen_vertex) {
      return;
    }

    int start_idx = d_tmp_adjp[gid];
    int end_idx = d_tmp_adjp[gid + 1];

    int i, j, l, r, idx, sum;
    i = start_idx;
    idx = 0;
    sum = 0;

    //printf("gid:%d, start_idx:%d, end_idx:%d \n", gid, start_idx, end_idx);

    while(i < end_idx) {
      //check if neighbor is equal to itself
      //int val = d_tmp_coarsen_adjncy[i];
      l = d_tmp_coarsen_adjncy[i];
      //printf("l:%d, i:%d \n", l, i);
      //remove itself
      if(l == gid + 1) {
        i++;
      }
      else {
        j = i + 1;
        sum = d_tmp_coarsen_adjwgt[i];
        while(j < end_idx) {
          r = d_tmp_coarsen_adjncy[j];
          if(l == r) {
            sum += d_tmp_coarsen_adjwgt[j];
            j++;
          }
          else {
            break;
          }
        }
        d_tmp_coarsen_adjncy[start_idx+idx] = l;
        d_tmp_coarsen_adjwgt[start_idx+idx] = sum;
        idx++;
        i = j;
      }
    }

    d_coarsen_adjp[gid + 1] = idx;
  }

  void contraction(
    const unsigned NUM_VERTICES,
    const unsigned NUM_EDGES,
    unsigned* d_cmap,
    unsigned* d_adjp,
    unsigned* d_adjncy,
    unsigned* d_adjwgt,
    unsigned* d_coarsen_adjp,
    unsigned* d_tmp_adjp,
    unsigned* d_coarsen_adjncy,
    unsigned* d_coarsen_adjwgt,
    unsigned* d_tmp_adjncy,
    unsigned* d_tmp_adjwgt,
    unsigned* d_cv,
    //unsigned* d_tmp_adjp,
    unsigned num_coarsen_vertex,
    unsigned& num_coarsen_edge,
    cudaStream_t stream1,
    mgpu::context_t& context) {
 
    unsigned num_block = (NUM_VERTICES + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    construct_tmp_adjp <<< num_block, THREAD_PER_BLOCK, 0, stream1 >>> (d_adjp, d_tmp_adjp, d_cmap, NUM_VERTICES);

    mgpu::scan<mgpu::scan_type_inc>(d_tmp_adjp, (num_coarsen_vertex + 1), d_tmp_adjp, mgpu::plus_t<unsigned>(), d_cv, context);
    //thrust::inclusive_scan(thrust::cuda::par.on(stream1), d_tmp_adjp, d_tmp_adjp + (num_coarsen_vertex + 1), d_tmp_adjp);

    //check_cuda(cudaMemcpyAsync(&num_coarsen_edge, d_tmp_adjp + num_coarsen_vertex, sizeof(int), cudaMemcpyDeviceToHost, stream1));
    check_cuda(cudaMemcpy(&num_coarsen_edge, d_cv, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "num_coarsen_edge:" << num_coarsen_edge << '\n';
    std::cout << "num_coarsen_vertex:" << num_coarsen_vertex << '\n';

    check_cuda(cudaMemcpyAsync(d_coarsen_adjp, d_tmp_adjp, sizeof(int) * (num_coarsen_vertex+1), cudaMemcpyDeviceToDevice, stream1));

    copied_adjncy <<< num_block, THREAD_PER_BLOCK, 0, stream1 >>> (d_adjp, d_coarsen_adjp, d_tmp_adjp, d_adjncy, d_tmp_adjncy, d_adjwgt, d_tmp_adjwgt, d_cmap, NUM_VERTICES);

    ////segmented sort
    mgpu::segmented_sort(d_tmp_adjncy, d_tmp_adjwgt, NUM_EDGES, d_tmp_adjp, num_coarsen_vertex, []MGPU_DEVICE(int a, int b){return a < b;}, context);
    check_cuda(cudaMemset(d_coarsen_adjp, 0, sizeof(int) * (num_coarsen_vertex + 1)));
    HEC_remove_duplicate_edge <<< num_block, THREAD_PER_BLOCK, 0, stream1 >>> (d_adjwgt, d_tmp_adjp, d_tmp_adjncy, d_tmp_adjwgt, d_coarsen_adjp, num_coarsen_vertex);
    
    mgpu::scan<mgpu::scan_type_inc>(d_coarsen_adjp, (num_coarsen_vertex + 1), d_coarsen_adjp, mgpu::plus_t<unsigned>(), d_cv, context);
    //thrust::inclusive_scan(thrust::cuda::par.on(stream1), d_coarsen_adjp, d_coarsen_adjp + (num_coarsen_vertex + 1), d_coarsen_adjp);
    //check_cuda(cudaMemcpyAsync(&num_coarsen_edge, d_coarsen_adjp + num_coarsen_vertex, sizeof(int), cudaMemcpyDeviceToHost, stream1));
    check_cuda(cudaMemcpy(&num_coarsen_edge, d_cv, sizeof(unsigned), cudaMemcpyDeviceToHost));
    std::cout << "** num_coarsen_edge:" << num_coarsen_edge << "\n";

    construct_coarsen_graph <<< num_block, THREAD_PER_BLOCK, 0, stream1 >>> (d_coarsen_adjncy, d_coarsen_adjwgt, d_tmp_adjncy, d_tmp_adjwgt, d_coarsen_adjp, d_tmp_adjp, num_coarsen_vertex);
  }

  __global__
  void init_group_id(unsigned long long* d_group_id, unsigned* d_vertex_id, const unsigned NUM_VERTICES) {
    unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= NUM_VERTICES) {
      return;
    }
    d_group_id[gid] = get_value_combo(gid + 1, 0);
    d_vertex_id[gid] = gid + 1;
  }

  __global__
  void count_coarsen_group(unsigned long long* d_group_id, unsigned* d_vertex_id, unsigned* d_group_head, const unsigned NUM_VERTICES) {
    unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= NUM_VERTICES) {
      return;
    }
    if(get_first_val_from_value_combo(d_group_id[gid]) == d_vertex_id[gid]) {
      d_group_head[gid] = 1;
    }
    else {
      d_group_head[gid] = 0;
    }
  }

  __global__
  void init_group_ptr(unsigned long long* d_group_id, unsigned* d_vertex_id, unsigned* d_group_head, unsigned* d_group_ptr,
                      const unsigned NUM_COARSEN_VERTICES, const unsigned NUM_VERTICES) {
    unsigned gid = threadIdx.x + blockDim.x * blockIdx.x;
    if(gid >= NUM_VERTICES) {
      return;
    }
    if(gid == NUM_COARSEN_VERTICES) {
      d_group_ptr[NUM_COARSEN_VERTICES] = NUM_VERTICES;
    }
    unsigned group_id = get_first_val_from_value_combo(d_group_id[gid]);
    unsigned new_idx;
    if(group_id == d_vertex_id[gid]) {
      new_idx = d_group_head[group_id - 1];
      //printf("gid:%d, group_id:%d, d_vertex_id[gid]:%d, new_idx:%d \n", gid, group_id, d_vertex_id[gid], new_idx);
      d_group_ptr[new_idx] = gid;
    }
  }

  __global__
  void breakdown_large_group(unsigned long long* d_group_id, unsigned* d_group_head, unsigned* d_vertex_id, unsigned* d_group_ptr, const unsigned NUM_VERTICES) {
    unsigned gid = threadIdx.x + blockDim.x * blockIdx.x;
    if(gid >= NUM_VERTICES) {
      return;
    }
    unsigned group_id = get_first_val_from_value_combo(d_group_id[gid]);
    unsigned group_idx = d_group_head[group_id - 1];
    unsigned left_idx_in_group = gid - d_group_ptr[group_idx];
    unsigned subgroup_id = left_idx_in_group / D_MAX_COARSEN_GROUP;
    //printf("gid:%d, group_id:%d, group_idx:%d, left_idx_in_group:%d, subgroup_id:%d \n", gid, group_id, group_idx, left_idx_in_group, subgroup_id);
    d_group_id[gid] = get_value_combo(d_vertex_id[d_group_ptr[group_idx] + subgroup_id * D_MAX_COARSEN_GROUP], 0);
  }

  void constraint_group_size(unsigned long long* d_group_id, unsigned* d_vertex_id, unsigned* d_vwgt, unsigned* d_group_head, unsigned* d_group_ptr,
                             unsigned* d_hec_match_candidate, unsigned* hp_cv, unsigned* d_sum, cudaStream_t stream1, mgpu::context_t& context, const unsigned NUM_VERTICES) {
    const unsigned NUM_BLOCK_VERTEX_PER_THREAD = (NUM_VERTICES + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    count_coarsen_group <<< NUM_BLOCK_VERTEX_PER_THREAD, THREAD_PER_BLOCK, 0, stream1 >>> (d_group_id, d_vertex_id, d_group_head, NUM_VERTICES); 
    mgpu::scan<mgpu::scan_type_exc>(d_group_head, NUM_VERTICES, d_group_head, mgpu::plus_t<unsigned>(), d_sum, context);
    cudaMemcpy(hp_cv, d_sum, sizeof(unsigned), cudaMemcpyDeviceToHost);
    unsigned num_coarsen_vertices = *hp_cv;
    mgpu::mergesort(d_group_id, d_vertex_id, NUM_VERTICES, mgpu::less_t<unsigned long long>(), context);

    init_group_ptr <<< NUM_BLOCK_VERTEX_PER_THREAD, THREAD_PER_BLOCK, 0, stream1 >>> (d_group_id, d_vertex_id, d_group_head, d_group_ptr, num_coarsen_vertices, NUM_VERTICES);
  
    breakdown_large_group <<< NUM_BLOCK_VERTEX_PER_THREAD, THREAD_PER_BLOCK, 0, stream1 >>> (d_group_id, d_group_head, d_vertex_id, d_group_ptr, NUM_VERTICES);
 
  }

  void coarsening(
    const unsigned NUM_VERTICES,
    const unsigned NUM_EDGES,
    unsigned long long* d_group_id,
    unsigned* d_vertex_id,
    unsigned* d_group_head,
    unsigned* d_group_ptr,
    unsigned* d_ulabel,
    unsigned* d_tmp_cmap,
    unsigned* d_tmp_ps,
    unsigned* d_vwgt,
    unsigned* d_adjp,
    unsigned* d_adjncy,
    unsigned* d_adjwgt,
    unsigned* d_coarsen_vwgt,
    unsigned* d_coarsen_adjp,
    unsigned* d_coarsen_adjncy,
    unsigned* d_coarsen_adjwgt,
    unsigned* d_tmp_coarsen_adjncy,
    unsigned* d_tmp_coarsen_adjwgt,
    unsigned* d_hec_match_candidate,
    unsigned* d_hec_cmap,
    unsigned& num_coarsen_vertex,
    unsigned& num_coarsen_edge,
    unsigned* hp_cv,
    unsigned* d_cv,
    cudaStream_t stream1,
    mgpu::context_t& context) {

    check_cuda(cudaMemset(d_cv, 0, sizeof(int)));
    check_cuda(cudaMemset(d_tmp_cmap, 0, sizeof(unsigned) * NUM_VERTICES));

    const unsigned NUM_BLOCKS = (NUM_VERTICES + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    matching_with_score <<< NUM_BLOCKS, THREAD_PER_BLOCK, 0 , stream1 >>> (d_vwgt, d_adjp, d_adjncy, d_adjwgt, d_hec_match_candidate,  NUM_VERTICES);

    init_group_id <<< NUM_BLOCKS, THREAD_PER_BLOCK, 0, stream1 >>> (d_group_id, d_vertex_id, NUM_VERTICES);
    
    *hp_cv = 1;
    unsigned it = 1;
    while(*hp_cv == 1){
      HEC_union_find <<< NUM_BLOCKS, THREAD_PER_BLOCK, 0, stream1 >>> (d_hec_match_candidate, d_group_id, d_cv, it, NUM_VERTICES);
      cudaMemcpy(hp_cv, d_cv, sizeof(unsigned), cudaMemcpyDeviceToHost);
      std::cout << "if change:" << *hp_cv << '\n';
      it++;
      check_cuda(cudaMemset(d_cv, 0, sizeof(unsigned)));
    }

    constraint_group_size(d_group_id, d_vertex_id, d_vwgt, d_group_head, d_group_ptr, d_hec_match_candidate, hp_cv, d_cv, stream1, context, NUM_VERTICES);
    cudaMemset(d_group_head, 0, sizeof(unsigned) * NUM_VERTICES);

    count_num_coarsen_vertex <<< NUM_BLOCKS, THREAD_PER_BLOCK, 0, stream1 >>> (d_group_id, d_vertex_id, d_group_head,  NUM_VERTICES);

    mgpu::scan<mgpu::scan_type_inc>(d_group_head, NUM_VERTICES, d_group_head, mgpu::plus_t<unsigned>(), d_cv, context);
    cudaMemcpy(hp_cv, d_cv, sizeof(unsigned), cudaMemcpyDeviceToHost);
    num_coarsen_vertex = *hp_cv;
    std::cout << "*** num_coarsen_vertex:" << num_coarsen_vertex << '\n';

    construct_cmap <<< NUM_BLOCKS, THREAD_PER_BLOCK, 0, stream1 >>> (d_group_head, d_vertex_id, d_hec_cmap, NUM_VERTICES);

    cal_coarsen_vwgt <<< NUM_BLOCKS, THREAD_PER_BLOCK, 0, stream1 >>> (d_vwgt, d_coarsen_vwgt, d_hec_cmap, NUM_VERTICES);

    contraction(NUM_VERTICES, NUM_EDGES, d_hec_cmap, d_adjp, d_adjncy, d_adjwgt, d_coarsen_adjp, d_tmp_ps, d_coarsen_adjncy, 
                    d_coarsen_adjwgt, d_tmp_coarsen_adjncy, d_tmp_coarsen_adjwgt, d_cv, num_coarsen_vertex, num_coarsen_edge, 
                    stream1, context); 

 }

} // end of namespace gk ==========================================
