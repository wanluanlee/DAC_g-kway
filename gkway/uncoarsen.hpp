#pragma once

#include"cuda_check.hpp"
#include"move_request.hpp"
#include<iostream>
#include "moderngpu/src/moderngpu/kernel_mergesort.hxx"
namespace gk { // begin of namespace gk ============================================

 //check if a single move can result in a balance partition 
  __device__
  bool is_valid_move(
    unsigned des_partition_wgt,  
    unsigned vwgt) {

    float des_balance = des_partition_wgt + vwgt;
    
    if(des_balance > D_MAX_PARTITION_WGT) {
      return false;
    }
    return true;
  }

  __device__
  unsigned atomicAddInc(unsigned* d_cnt, unsigned lane_id) {
    unsigned int active = __activemask();
    unsigned leader = __ffs(active) -1;
    unsigned change = __popc(active);
    unsigned int rank = __popc(active & (((int64_t)1 << lane_id) - 1));
    unsigned warp_res;
    if(rank == 0) {
      warp_res = atomicAdd(d_cnt, change);
    }
    warp_res = __shfl_sync(active, warp_res, leader);
    return warp_res + rank;
  }

 //each block handle one partition delta wgt and max array size if 1024
 __global__
 void scan(int* d_in, int N) {
  
   unsigned gid = threadIdx.x + blockIdx.x * N;
   unsigned tid = threadIdx.x;
   __shared__ int tmp[WARP_SIZE];
   int tmp1, tmp2, tmp3;
   if(tid >= N) {
     return;
   }
   tmp1 = d_in[gid];
   for(int off = 1; off < 32; off <<= 1) {
     tmp2 = __shfl_up_sync(FULL_MASK, tmp1, off);
     if(tid % WARP_SIZE >= off) {
       tmp1 += tmp2;
     }
   }
   if(tid % WARP_SIZE == 31) {
     tmp[tid / WARP_SIZE] = tmp1;
   }
   __syncthreads();
   if(tid < WARP_SIZE) {
     tmp2 = 0;
     if(tid < blockDim.x / WARP_SIZE) {
       tmp2 = tmp[tid];
     }
     for(int off = 1; off < 32; off <<= 1) {
       tmp3 = __shfl_up_sync(FULL_MASK, tmp2, off);
       if(tid % WARP_SIZE >= off) {
         tmp2 += tmp3;
       }
     }
     if(tid < blockDim.x / 32) {
       tmp[tid] = tmp2;
     }
   }
   //printf("second blockIdx.x%d, tid:%d, tmp2:%d \n", blockIdx.x, tid, tmp2);
   __syncthreads();
   if(tid >= WARP_SIZE) {
     tmp1 += tmp[tid/WARP_SIZE - 1]; 
   }
   d_in[gid] = tmp1;
 }

  __device__
  unsigned calculate_gain(
    unsigned* d_partition, 
    unsigned vertex_id, 
    unsigned vertex_partition,
    unsigned* d_adjncy, 
    unsigned* d_adjwgt,
    unsigned des_partition,
    unsigned neighbor_start_idx, 
    unsigned neighbor_end_idx) {

    unsigned id = 0; //internal degree
    unsigned ed = 0; //external degree
    if(vertex_partition == des_partition) {
      return 0;
    }
    for(int s = neighbor_start_idx; s < neighbor_end_idx; s++) {
      unsigned neighbor_vertex_idx = d_adjncy[s] - 1;    
      unsigned neighbor_partition = d_partition[neighbor_vertex_idx];
      if(vertex_partition == neighbor_partition) {
        id += d_adjwgt[s];
      }
      if(neighbor_partition == des_partition) {
        ed += d_adjwgt[s];
      }
    }
    return ed - id;
  }

  __global__ 
  void create_independent_move_buffer(
    unsigned* d_partition,   
    unsigned* d_partition_wgt,
    unsigned* d_vwgt,
    unsigned* d_adjncy,
    unsigned* d_adjp,
    unsigned* d_if_boundary,
    int* d_vertex_gain,
    unsigned* d_max_gain_partition,
    unsigned* d_if_update,
    mvRequest* d_mv_buffer, 
    unsigned* d_buffer_size,
    const unsigned NUM_VERTICES) {

    unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned lane_id = gid % WARP_SIZE;
    unsigned vertex_id = gid + 1;

    if(gid >= NUM_VERTICES) {
      return;
    }
    d_if_update[gid] = 0;
    if(d_if_boundary[gid] == 0) {
      return;
    }
    unsigned vertex_partition = d_partition[gid];
    unsigned vertex_wgt = d_vwgt[gid]; 
    int gain = d_vertex_gain[gid];
    if(gain <= 0) {
      return;
    }
    unsigned max_gain_partition = d_max_gain_partition[gid];
    unsigned des_partition_wgt = d_partition_wgt[max_gain_partition];
    bool balance = is_valid_move(des_partition_wgt, vertex_wgt);
    if(!balance) {
      return;
    }
    unsigned start = d_adjp[gid];  
    unsigned end = d_adjp[gid+1];
    for(unsigned i = start; i < end; i++) {
       //check if the neighbor has positive and is move
      unsigned neighbor_idx = d_adjncy[i] - 1;
      unsigned neighbor_max_gain_partition = d_max_gain_partition[neighbor_idx];
      unsigned neighbor_vwgt = d_vwgt[neighbor_idx];
      if(d_vertex_gain[neighbor_idx] > 0 && is_valid_move(d_partition_wgt[neighbor_max_gain_partition], neighbor_vwgt))    {
     //if(d_vertex_gain[neighbor_idx] > 0) {
        if(gid + 1 > neighbor_idx + 1) {
          return;
        }
      } 
    } 
    __syncwarp();

    unsigned pos = atomicAddInc(d_buffer_size, lane_id);
    //int pos = atomicAdd(d_buffer_size,1);
    mvRequest mv_request;
    mv_request.vertex_id = vertex_id; 
    mv_request.source_partition = vertex_partition;
    mv_request.des_partition = max_gain_partition;
    mv_request.gain = gain;
    d_mv_buffer[pos] = mv_request;
  }

  __global__
  void update_vertex_info(
    unsigned* d_partition, 
    unsigned* d_if_boundary, 
    int* d_vertex_gain,
    unsigned* d_max_gain_partition,
    unsigned* d_adjncy,
    unsigned* d_adjwgt,
    unsigned* d_adjp,
    unsigned* d_if_updated_vertex,
    const unsigned NUM_VERTICES) {

    unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= NUM_VERTICES || d_if_updated_vertex[gid] == 0) {
      return;
    }
    //loop through to find if neighbor 
    __shared__ int s_neighbor_partition[MAX_SHARE_SIZE];
    unsigned start = d_adjp[gid];  
    unsigned end = d_adjp[gid + 1];
    unsigned share_start = threadIdx.x * D_NUM_PARTITIONS;
    d_if_boundary[gid] = 0; //false

    unsigned partition = d_partition[gid];
    unsigned if_boundary{0};
    
    //init assigned s_neighbor_partition array to 0
    for(int i = 0; i < D_NUM_PARTITIONS; i++) {
      s_neighbor_partition[share_start+i] = 0;
    }
    for(int i = start; i < end; i++) {
      unsigned neighbor_vertex_idx = d_adjncy[i] - 1;  
      unsigned neighbor_partition = d_partition[neighbor_vertex_idx];
      s_neighbor_partition[share_start+neighbor_partition] = 1;
      if(neighbor_partition != partition) {
        if_boundary = 1;
      }
    }

    d_if_boundary[gid] = if_boundary;

    if(if_boundary == 0) {
      d_vertex_gain[gid] = 0;
      d_max_gain_partition[gid] = partition;
      return;
    }

    unsigned max_gain = 0;
    unsigned max_partition = partition;
      //calculate gain
    for(int i = 0; i < D_NUM_PARTITIONS; i++) {
      if(s_neighbor_partition[share_start+i] == 1 && i != partition) { // there are neighbors at this partition
        int gain = calculate_gain(d_partition, gid, partition, d_adjncy, d_adjwgt, i, start, end);
        //printf("gid:%d, parition:%d, gain:%d \n", gid, i, gain);
        if(gain > max_gain) {
          max_gain = gain;
          max_partition = i;
        }
      }
    }
    d_vertex_gain[gid] = max_gain;
    d_max_gain_partition[gid] = max_partition;
  }

 __global__
 void find_mv_wgt_sequence(
   mvRequest* d_sorted_buffer,
   int* d_mv_delta_partition_wgt,
   unsigned* d_vwgt,
   unsigned buffer_size) {

   unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
   if(gid >= buffer_size) {
     return;
   }

   mvRequest mv = d_sorted_buffer[gid];
   unsigned src_partition_idx = mv.source_partition * buffer_size + gid;
   unsigned des_partition_idx = mv.des_partition * buffer_size + gid; 
   unsigned vwgt = d_vwgt[mv.vertex_id - 1];
   d_mv_delta_partition_wgt[src_partition_idx] = vwgt * -1;
   d_mv_delta_partition_wgt[des_partition_idx] = vwgt;
 }

 __global__
 void find_balance_sequence(
   mvRequest* d_sorted_mv,
   int* d_mv_delta_partition_wgt,
   unsigned* d_partition_wgt,
   unsigned* d_mv_balance_sequence,
   int* d_op_result,
   unsigned buffer_size) {

   unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
   if(gid >= buffer_size) {
     return;
   }
   
   //mvRequest mv = d_sorted_mv[gid];
   int if_balance = 1;
   for(int i = 0; i < D_NUM_PARTITIONS; i++) {
     int delta_partition_wgt = d_mv_delta_partition_wgt[gid + i * buffer_size];
     if(delta_partition_wgt > 0) {
       unsigned partition_wgt = delta_partition_wgt + d_partition_wgt[i];
       if(partition_wgt > D_MAX_PARTITION_WGT) {
         if_balance = 0;
       }
     }
   }
   if(if_balance == 1) {
    atomicMax(d_op_result, gid);
   }
   d_mv_balance_sequence[gid] = if_balance;   
 }

 __global__
 void apply_sequence_move(
   mvRequest* d_sorted_buffer,
   int* d_mv_delta_partition_wgt,
   unsigned* d_partition,
   unsigned* d_partition_wgt,
   unsigned* d_if_updated_vertex,
   unsigned* d_vwgt,
   unsigned* d_adjp,
   unsigned* d_adjncy,
   unsigned* d_cutsize,
   int* d_pos,
   unsigned buffer_size) {

   unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
   if(gid < D_NUM_PARTITIONS) {
     int delta_partition_wgt = d_mv_delta_partition_wgt[gid * buffer_size + *d_pos];
     d_partition_wgt[gid] += delta_partition_wgt;
   }
   if(gid > *d_pos) {
     return;
   }

   mvRequest mv = d_sorted_buffer[gid];
   d_partition[mv.vertex_id - 1] = mv.des_partition;
   d_if_updated_vertex[mv.vertex_id - 1] = 1;
   atomicSub(d_cutsize, mv.gain);
   unsigned start = d_adjp[mv.vertex_id - 1];
   unsigned end = d_adjp[mv.vertex_id];
   for(unsigned i = start; i < end; i++) {
     unsigned neighbor_vertex_id = d_adjncy[i] - 1;
     d_if_updated_vertex[neighbor_vertex_id]  = 1;
   }
 }

  __global__
  void expand_partition(
    unsigned* d_partition, 
    unsigned* d_tmp_partition,   
    unsigned* d_if_boundary,
    unsigned* d_tmp_if_boundary,
    unsigned* d_cmap, 
    unsigned* d_if_updated_vertex,
    int* d_max_gain,
    unsigned* d_max_gain_partition,
    const unsigned FINER_NUM_VERTICES) {

    unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= FINER_NUM_VERTICES) {
      return;
    }
    //check idx in coarsen graph
    unsigned coarsen_idx = d_cmap[gid] - 1;
    //printf("gid=%d, coarsen_idx=%d, d_partition:%d \n", gid, coarsen_idx, d_partition[coarsen_idx]);    
    d_tmp_partition[gid] = d_partition[coarsen_idx];
    d_max_gain[gid] = 0;
    d_max_gain_partition[gid] = d_partition[coarsen_idx];
    unsigned if_boundary = d_if_boundary[coarsen_idx];
    d_tmp_if_boundary[gid] = if_boundary;
    //printf("gid:%d, num:%f \n", gid, d_vertex_random_idx[gid]);
    if(if_boundary == 1) {
      d_if_updated_vertex[gid] = 1;
    }
    else {
      d_if_updated_vertex[gid] = 0;
    }
  }

  void uncoarsen(
    unsigned* d_partition, 
    unsigned* d_tmp_if_boundary,
    unsigned* d_if_boundary, 
    int* d_vertex_gain,
    unsigned* d_max_gain_partition,
    unsigned* d_adjncy,
    unsigned* d_adjwgt,
    unsigned* d_adjp,
    unsigned* d_cmap, 
    unsigned* d_tmp_partition,
    unsigned* d_if_updated_vertex,
    unsigned* d_partition_wgt,
    const unsigned FINER_NUM_VERTICES,
    const unsigned FINER_NUM_EDGES,
    const unsigned COARSER_NUM_VERTICES,
    const unsigned COARSER_NUM_EDGES,
    cudaStream_t stream1) {

    const unsigned NUM_BLOCK = (FINER_NUM_VERTICES + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    expand_partition <<< NUM_BLOCK, THREAD_PER_BLOCK, 0, stream1 >>> (d_partition, d_tmp_partition, d_if_boundary, d_tmp_if_boundary, 
                                                                      d_cmap, d_if_updated_vertex, d_vertex_gain, d_max_gain_partition, 
                                                                      FINER_NUM_VERTICES);
    check_cuda(cudaMemcpy(d_partition, d_tmp_partition, sizeof(unsigned) * FINER_NUM_VERTICES, cudaMemcpyDeviceToDevice));
    check_cuda(cudaMemcpy(d_if_boundary, d_tmp_if_boundary, sizeof(unsigned) * FINER_NUM_VERTICES, cudaMemcpyDeviceToDevice)); 
    const unsigned MAX_VERTEX_PER_BLOCK = 128;
    const unsigned VERTEX_GROUP_BLOCK = (FINER_NUM_VERTICES + MAX_VERTEX_PER_BLOCK - 1)/ MAX_VERTEX_PER_BLOCK;
    update_vertex_info <<< VERTEX_GROUP_BLOCK, MAX_VERTEX_PER_BLOCK, 0, stream1 >>> (d_partition, d_if_boundary, d_vertex_gain, d_max_gain_partition, 
                                                                                     d_adjncy, d_adjwgt, d_adjp, d_if_updated_vertex, FINER_NUM_VERTICES);
  }

  __global__
  void print_buffer(mvRequest* d_mv_buffer, unsigned* d_vwgt, unsigned buffer_size) {
    //printf("buffer size=%d \n", buffer_size);
    for(int i = 0; i < buffer_size; i++) {
      printf("print_buffer: i=%d, vertex_id=%d, source_partition=%d, des_partition=%d, gain=%d, vwgt:%d\n", i, d_mv_buffer[i].vertex_id, 
        d_mv_buffer[i].source_partition, d_mv_buffer[i].des_partition, d_mv_buffer[i].gain, d_vwgt[d_mv_buffer[i].vertex_id-1]); 
    }
  }

  __global__
  void calculate_cutsize(
    unsigned* d_partition,
    unsigned* d_adjncy,
    unsigned* d_adjwgt,
    unsigned* d_adjp,
    unsigned* d_cutsize,
    const unsigned NUM_VERTICES) {

    unsigned gid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned vertex_id = gid + 1;
    if(gid >= NUM_VERTICES) {
      return;
    }
    unsigned start = d_adjp[gid];
    unsigned end = d_adjp[gid+1];
    unsigned partition = d_partition[gid];
    for(unsigned i = start; i < end; i++){
      unsigned neighbor_vertex = d_adjncy[i];
      unsigned neighbor_vertex_idx = neighbor_vertex - 1;
      //check if edge is a cute edge
      unsigned neighbor_partition = d_partition[neighbor_vertex_idx];
      if(partition != neighbor_partition && vertex_id < neighbor_vertex) {
        atomicAdd(d_cutsize, d_adjwgt[i]);
      }
    }
  }


  void refinement( 
    unsigned* d_adjp, 
    unsigned* d_vwgt, 
    unsigned* d_partition,
    unsigned* d_partition_wgt,
    unsigned* d_adjncy, 
    unsigned* d_adjwgt,
    unsigned* d_buffer_size,
    unsigned* hp_buffer_size,
    unsigned* d_find_idx,
    int* d_op_result,
    unsigned* hp_pos,
    int* hp_op_result,
    unsigned* d_if_boundary,
    int* d_vertex_gain,
    unsigned* d_max_gain_partition,
    unsigned* d_if_updated_vertex,
    unsigned* d_cutsize,
    mvRequest* d_mv_buffer,
    int* d_mv_delta_partition_wgt,
    unsigned* d_mv_balance_sequence,
    const unsigned NUM_VERTICES,
    const unsigned NUM_EDGES,
    const int NUM_PARTITIONS,
    cudaStream_t stream1,
    mgpu::context_t& context) {
    
    //std::cout << "************************************************\n";
    //std::cout << "@In refinement, NUM_VERTICES: " << NUM_VERTICES << ", NUM_EDGES: " << NUM_EDGES <<'\n';
    //std::cout << "************************************************\n";
    const unsigned NUM_BLOCKS = (NUM_VERTICES + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    while(true) {
      auto start = std::chrono::system_clock::now();
      check_cuda(cudaMemset(d_buffer_size, 0, sizeof(unsigned)));
      create_independent_move_buffer <<< NUM_BLOCKS, THREAD_PER_BLOCK, 0, stream1 >>> (d_partition, d_partition_wgt, d_vwgt, d_adjncy, 
                                                                                      d_adjp, d_if_boundary, d_vertex_gain, d_max_gain_partition, 
                                                                                      d_if_updated_vertex, d_mv_buffer, d_buffer_size, NUM_VERTICES);
      check_cuda(cudaMemcpy(hp_buffer_size, d_buffer_size, sizeof(unsigned), cudaMemcpyDeviceToHost));
      if(*hp_buffer_size == 0) {
        break;
      }
      const unsigned NUM_BLOCK_FOR_MOVE_BUFFER = (*hp_buffer_size + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
      
      if(*hp_buffer_size > 1) {
        mgpu::mergesort(d_mv_buffer, *hp_buffer_size, mgpu::less_t<mvRequest>(), context);
      }
      //std::cout << "buffer_size:" << buffer_size << '\n';
      unsigned tmp_buffer_size = std::min(*hp_buffer_size, (unsigned)1024); 
      *hp_buffer_size = tmp_buffer_size;
      unsigned memset_size = *hp_buffer_size * NUM_PARTITIONS;
      check_cuda(cudaMemset(d_mv_delta_partition_wgt, 0, sizeof(int) * memset_size));
      find_mv_wgt_sequence <<< NUM_BLOCK_FOR_MOVE_BUFFER, THREAD_PER_BLOCK, 0, stream1 >>>(d_mv_buffer, d_mv_delta_partition_wgt, d_vwgt, *hp_buffer_size);
      scan <<< NUM_PARTITIONS, 1024, 0, stream1 >>>(d_mv_delta_partition_wgt, *hp_buffer_size);
      cudaMemset(d_op_result, 0, sizeof(int));
      find_balance_sequence <<< NUM_BLOCK_FOR_MOVE_BUFFER, THREAD_PER_BLOCK, 0, stream1 >>>(d_mv_buffer, d_mv_delta_partition_wgt, d_partition_wgt, 
                                                                                   d_mv_balance_sequence, d_op_result, *hp_buffer_size);
      cudaMemcpy(hp_op_result, d_op_result, sizeof(int), cudaMemcpyDeviceToHost);
      if(*hp_op_result == -1) { //no balance move
        std::cout << "!!! break: *h_find_if_pos:" << *hp_op_result  << '\n';
        break;
      }
      apply_sequence_move <<< NUM_BLOCK_FOR_MOVE_BUFFER, THREAD_PER_BLOCK >>> (d_mv_buffer, d_mv_delta_partition_wgt, d_partition, d_partition_wgt, 
                                                                      d_if_updated_vertex, d_vwgt, d_adjp, d_adjncy, d_cutsize, d_op_result, 
                                                                      *hp_buffer_size); 

      unsigned h_cutsize;
      check_cuda(cudaMemcpy(&h_cutsize, d_cutsize, sizeof(unsigned), cudaMemcpyDeviceToHost));
      const unsigned MAX_VERTEX_PER_BLOCK = 128;
      const unsigned VERTEX_GROUP_BLOCK = (NUM_VERTICES + MAX_VERTEX_PER_BLOCK - 1) / MAX_VERTEX_PER_BLOCK;
      update_vertex_info <<< VERTEX_GROUP_BLOCK, MAX_VERTEX_PER_BLOCK, 0, stream1 >>> (d_partition, d_if_boundary, d_vertex_gain, d_max_gain_partition, 
                                                                                       d_adjncy, d_adjwgt, d_adjp, d_if_updated_vertex, NUM_VERTICES); 

    }

    //std::cout << "************************************************\n";
    //std::cout << "@ Finish refinment, NUM_VERTICES: " << NUM_VERTICES << ", NUM_EDGES: " << NUM_EDGES << '\n';;
    ////std::cout << "move it: " << move_it << '\n';
    //std::cout << "************************************************\n";
  }
}
