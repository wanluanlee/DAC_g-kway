#ifndef CUDA_CHECK_HPP
#define CUDA_CHECK_HPP

#include<string>
#include<sstream>
#include<stdexcept>
#include"move_request.hpp"

cudaError_t check_cuda(cudaError_t result) {
  if(result!= cudaSuccess) {
    using namespace std::literals::string_literals;
    throw std::runtime_error("Cuda Runtime Error: "s + cudaGetErrorString(result));
  }
  return result;
}

template <typename Input>
__global__
void print_array(Input a, int length, int label) {
  printf("***********************************************\n");
  for(int i = 0; i < length; ++i) {
    printf("label=%d, at i=%d, val=%d \n",label, i, a[i]);
  }
  printf("***********************************************\n");
}

__global__
void print_mvRequest(mvRequest* a, int length) {
  printf("***********************************************\n");
  for(int i = 0; i < length; ++i) {
    printf("at i=%d, mv.vertex_id=%d, mv.source=%d, mv.des=%d, mv.gain=%d \n", i, a[i].vertex_id, 
    a[i].source_partition, a[i].des_partition, a[i].gain);
  }
  printf("***********************************************\n");
}

__global__
void print_candidate_combo(long long* d_candidate, int length) {

  printf("***********************************************\n");
  for(int i = 0; i < length; i++) {
    int candidate_wgt = (int)(d_candidate[i] >> 32); 
    int candidate_id = d_candidate[i] & 0xFFFFFFFF;
    printf("at i=%d, combo:%lld, d_candiate_wgt:%d, d_candidate_id:%d \n", i, d_candidate[i], candidate_wgt, candidate_id);
  }
  printf("***********************************************\n");
}

__global__
void print_group_id(unsigned long long* d_group_id, unsigned length) {

  printf("***********************************************\n");
  for(int i = 0; i < length; i++) {
    unsigned group_id = (int)(d_group_id[i] >> 32); 
    unsigned it = d_group_id[i] & 0xFFFFFFFF;
    printf("at i=%d, combo:%lld, d_candiate_wgt:%d, d_candidate_id:%d \n", i, d_group_id[i], group_id, it);
  }
  printf("***********************************************\n");
}

  __global__
  void calculate_partition_wgt(
    unsigned* d_partition,
    unsigned* d_partition_wgt,
    unsigned* d_vwgt,
    const unsigned NUM_VERTICES) {
    const unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= NUM_VERTICES) {
      return;
    }
    unsigned partition = d_partition[gid];
    atomicAdd(&d_partition_wgt[partition], d_vwgt[gid]);
  }

__global__
void degree_analysis(unsigned* d_adjp, unsigned* d_max_degree, unsigned* d_min_degree, unsigned* d_sum_degree, const unsigned NUM_VERTICES) {

  unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid >= NUM_VERTICES) { 
    return;
  }
  unsigned num_degree = d_adjp[gid + 1] - d_adjp[gid];
  atomicMin(d_min_degree, num_degree);
  atomicMax(d_max_degree, num_degree);
  atomicAdd(d_sum_degree, num_degree);
}

__device__
int find_min(const int& val1, const int& val2) {
  if(val1 <= val2) {
    return val1;
  }
  else {
    return val2;
  }
}

__global__
void print_device_const() {
  printf("D_NUM_PARTITIONS:%d \n", D_NUM_PARTITIONS);
  printf("D_TOTAL_VWGT:%zu \n", D_TOTAL_VWGT);
  printf("D_MAX_PARTITION_WGT:%f \n", D_MAX_PARTITION_WGT);
}

__global__
void check_partition_wgt(unsigned* d_partition_wgt) {

  unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned partition_wgt = d_partition_wgt[gid];
  if(partition_wgt > D_MAX_PARTITION_WGT) {
    printf("!!!@@@ ERROR, partition:%d is out of bound \n", gid);
  }
}

__global__
void check_max_vwgt(unsigned* d_max_wgt, unsigned* d_vwgt, const unsigned NUM_VERTICES) {

  unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid >= NUM_VERTICES) {
    return;
  }

  unsigned vwgt = d_vwgt[gid];
  atomicMax(d_max_wgt, vwgt);
}

#endif
