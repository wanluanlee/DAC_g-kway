#include "uncoarsen.hpp"
#include "coarsen.hpp"
#include "graph.hpp"
#include "move_request.hpp"
#include "cuda_check.hpp"
#include <chrono>
#include <fstream>
#include <vector>
#include "metis_partition.hpp"
#include "moderngpu/src/moderngpu/kernel_segsort.hxx"
#include "moderngpu/src/moderngpu/kernel_segreduce.hxx"
#include "../declarations.h"

void coarsening(std::vector<unsigned>& coarsen_num_vertices_vec, std::vector<unsigned>& coarsen_num_ptrs_vec, 
                std::vector<unsigned>& coarsen_num_edges_vec, std::vector<unsigned>& global_vertex_offset_vec,
                std::vector<unsigned>& global_ptr_offset_vec, std::vector<unsigned>& global_edge_offset_vec,
                int& coarsen_it, unsigned* d_vwgt, unsigned* d_adjp, unsigned* d_adjncy, unsigned* d_adjwgt, 
                unsigned* d_cmap, mgpu::context_t& context, const int NUM_PARTITIONS, const unsigned MAX_NUM_VERTICES, 
                const unsigned MAX_NUM_EDGES, const size_t MAX_VERTEX_ARRAY_SIZE, const size_t MAX_EDGE_ARRAY_SIZE, cudaStream_t stream1) {

  //const int COARSEN_THRESHOLD = std::min(int(MAX_NUM_VERTICES / (COARSEN_RATIO * std::log2(NUM_PARTITION))), 262159);
  const int COARSEN_THRESHOLD = 20 * NUM_PARTITIONS;
  std::cout << "!!### coarsen threadhold " << COARSEN_THRESHOLD << '\n';
  //temporaray array for match stage
  unsigned* d_match_candidate;
  unsigned* d_tmp_cmap;
  unsigned* d_tmp_ps;
  unsigned* d_ulabel;
  unsigned* d_tmp_num_neighbor;
  unsigned* d_tmp_coarsen_adjncy;
  unsigned* d_tmp_coarsen_adjwgt;
  unsigned* d_reduce_sum;
  unsigned* hp_reduce_sum;
  unsigned long long* d_group_id;
  unsigned* d_vertex_id;
  unsigned* d_group_head;
  unsigned* d_group_ptr;

  //allocate space on GPU global memory
  check_cuda(cudaMalloc((void **)&d_group_id, sizeof(unsigned long long) * MAX_NUM_VERTICES));
  check_cuda(cudaMalloc((void **)&d_vertex_id, sizeof(unsigned) * MAX_NUM_VERTICES));
  check_cuda(cudaMalloc((void **)&d_group_head, sizeof(unsigned) * MAX_NUM_VERTICES));
  check_cuda(cudaMalloc((void **)&d_group_ptr, sizeof(unsigned) * MAX_NUM_VERTICES));
  check_cuda(cudaMalloc((void **)&d_tmp_cmap, sizeof(unsigned) * MAX_NUM_VERTICES));
  check_cuda(cudaMalloc((void **)&d_ulabel, sizeof(unsigned) * MAX_NUM_VERTICES));
  check_cuda(cudaMalloc((void **)&d_tmp_ps, sizeof(unsigned) * (MAX_NUM_VERTICES + 1)));
  check_cuda(cudaMalloc((void **)&d_tmp_num_neighbor, sizeof(unsigned) * MAX_NUM_VERTICES));
  check_cuda(cudaMalloc((void **)&d_tmp_coarsen_adjncy, sizeof(unsigned) * MAX_NUM_EDGES));
  check_cuda(cudaMalloc((void **)&d_tmp_coarsen_adjwgt, sizeof(unsigned) * MAX_NUM_EDGES));
  check_cuda(cudaMalloc((void**)&d_match_candidate, sizeof(unsigned) * MAX_NUM_VERTICES));
  check_cuda(cudaMalloc((void**)&d_reduce_sum, sizeof(unsigned)));
  check_cuda(cudaMallocHost((void**)&hp_reduce_sum, sizeof(unsigned))); 
  cudaStreamSynchronize(stream1);

  unsigned num_vertices = MAX_NUM_VERTICES;
  unsigned num_edges = MAX_NUM_EDGES;
  unsigned coarsen_num_vertices{0};
  unsigned coarsen_num_edges{0};
  unsigned global_vertex_offset{0};
  unsigned global_ptr_offset{0};
  unsigned global_edge_offset{0};
  unsigned finer_global_vertex_offset{0};
  unsigned finer_global_ptr_offset{0};
  unsigned finer_global_edge_offset{0};

  while(num_vertices > COARSEN_THRESHOLD ) {
    //init output data
    check_cuda(cudaMemset(d_tmp_cmap, 0, sizeof(int) * num_vertices));
    check_cuda(cudaMemset(d_tmp_ps, 0, sizeof(int) * (num_vertices + 1)));
    global_vertex_offset += coarsen_num_vertices_vec.back();
    global_ptr_offset += (coarsen_num_ptrs_vec.back());
    global_edge_offset += coarsen_num_edges_vec.back();
    finer_global_vertex_offset = global_vertex_offset - coarsen_num_vertices_vec.back();
    finer_global_ptr_offset = global_ptr_offset  - coarsen_num_ptrs_vec.back();
    finer_global_edge_offset = global_edge_offset - coarsen_num_edges_vec.back();
    if(global_vertex_offset + num_vertices > MAX_VERTEX_ARRAY_SIZE) {
      std::cout << "ERROR, global_vertex_offset execess the max, increase the size to larger than " << (global_vertex_offset + num_vertices) << "\n";
    }
    if(global_ptr_offset + (num_vertices + 1) > MAX_VERTEX_ARRAY_SIZE) {
      std::cout << "ERROR, global_ptr_offset execess the max, increase the size to larger than " << (global_ptr_offset + num_vertices + 1) << "\n";
    }
    if(global_edge_offset + num_edges > MAX_EDGE_ARRAY_SIZE) {
      std::cout << "ERROR, global_edge_offset excess the max, increase the size to larger than " << (global_edge_offset + num_edges) << "\n";
    }

    gk::coarsening(num_vertices, num_edges, d_group_id, d_vertex_id, d_group_head, d_group_ptr, d_ulabel, d_tmp_cmap, d_tmp_ps, 
                   d_vwgt+finer_global_vertex_offset, d_adjp+finer_global_ptr_offset, 
                   d_adjncy+finer_global_edge_offset, d_adjwgt+finer_global_edge_offset, d_vwgt+global_vertex_offset, d_adjp+global_ptr_offset, 
                   d_adjncy+global_edge_offset, d_adjwgt+global_edge_offset, d_tmp_coarsen_adjncy, d_tmp_coarsen_adjwgt, d_match_candidate, 
                   d_cmap+finer_global_vertex_offset, coarsen_num_vertices, coarsen_num_edges, hp_reduce_sum, d_reduce_sum, stream1, context);
    coarsen_it++;

    if((num_vertices == coarsen_num_vertices) || (coarsen_num_vertices < (NUM_PARTITIONS + 1))) {
      
      break;
    }
    coarsen_num_vertices_vec.push_back(coarsen_num_vertices);
    coarsen_num_ptrs_vec.push_back(coarsen_num_vertices + 1);
    coarsen_num_edges_vec.push_back(coarsen_num_edges);

    global_vertex_offset_vec.push_back(global_vertex_offset);
    global_ptr_offset_vec.push_back(global_ptr_offset);
    global_edge_offset_vec.push_back(global_edge_offset);
    num_vertices = coarsen_num_vertices;
    num_edges = coarsen_num_edges;
    //std::cout << "coarsen_it: " << coarsen_it << ", num_vertices: " << num_vertices << ", num_edges: " << num_edges << '\n';
    std::cout << "coarsen_it: " << coarsen_it << ", coarsen_num_vertices: " << coarsen_num_vertices << ", coarsen_num_edges: " << coarsen_num_edges << '\n';
    //break;
  }
  cudaFree(d_match_candidate);
  cudaFree(d_tmp_cmap);
  cudaFree(d_tmp_ps);
  cudaFree(d_ulabel);
  cudaFree(d_tmp_num_neighbor);
  cudaFree(d_tmp_coarsen_adjncy);
  cudaFree(d_tmp_coarsen_adjwgt);
  cudaFree(d_reduce_sum);
  cudaFree(d_group_id);
  cudaFreeHost(hp_reduce_sum);
  cudaFree(d_vertex_id);
  cudaFree(d_group_head);
  cudaFree(d_group_ptr);
}

void init_partition(unsigned* d_vwgt, unsigned* d_adjp, unsigned* d_adjncy, unsigned* d_adjwgt, unsigned* d_partition, unsigned* d_partition_wgt, 
                    unsigned* d_if_boundary, unsigned* d_cutsize, const unsigned NUM_COARSEN_VERTEX, const unsigned NUM_COARSEN_EDGE, const unsigned NUM_PARTITIONS, 
                    cudaStream_t stream1) {

   metis_init_partition(d_vwgt, d_adjp, d_adjncy, d_adjwgt, d_partition, d_partition_wgt, d_if_boundary, d_cutsize, 
                         NUM_COARSEN_VERTEX, NUM_COARSEN_EDGE, NUM_PARTITIONS, stream1);
}

void uncoarsening(std::vector<unsigned>& coarsen_num_vertices_vec, std::vector<unsigned>& coarsen_num_edges_vec,
                  std::vector<unsigned>& global_vertex_offset_vec, std::vector<unsigned>& global_ptr_offset_vec,
                  std::vector<unsigned>& global_edge_offset_vec, unsigned* d_partition, unsigned* d_if_boundary,
                  unsigned* d_adjncy, unsigned* d_adjwgt, unsigned* d_adjp, unsigned* d_cmap, unsigned* d_partition_wgt, 
                  unsigned* d_cutsize, unsigned* d_vwgt, const int NUM_PARTITIONS, const unsigned MAX_NUM_VERTICES, 
                  const unsigned MAX_NUM_EDGES, cudaStream_t stream1, mgpu::context_t& context) {
   
   unsigned* d_if_updated_vertex;
   unsigned* d_max_gain_partition;
   int* d_vertex_gain;
   int* d_mv_delta_partition_wgt;
   unsigned* d_mv_balance_sequence;
   int* d_op_result;

   //uncoarsening variables
   mvRequest* d_mv_buffer;
   unsigned* d_tmp_partition;
   unsigned* d_tmp_if_boundary;
   unsigned* d_buffer_size;
   unsigned* hp_buffer_size;
   int* hp_op_result;
   unsigned* d_pos;
   unsigned* hp_pos;
   //unsigned long long* d_vertex_adj_partition;
   check_cuda(cudaMalloc((void**)&d_mv_buffer, sizeof(mvRequest) * MAX_NUM_VERTICES));
   check_cuda(cudaMalloc((void**)&d_tmp_partition, sizeof(unsigned) * MAX_NUM_VERTICES));
   check_cuda(cudaMalloc((void**)&d_tmp_if_boundary, sizeof(unsigned) * MAX_NUM_VERTICES));
   check_cuda(cudaMalloc((void**)&d_buffer_size, sizeof(unsigned)));
   check_cuda(cudaMalloc((void**)&d_if_updated_vertex, sizeof(unsigned) * MAX_NUM_VERTICES));
   check_cuda(cudaMalloc((void**)&d_vertex_gain, sizeof(int) * MAX_NUM_VERTICES));
   check_cuda(cudaMalloc((void**)&d_max_gain_partition, sizeof(unsigned) * MAX_NUM_VERTICES));
   check_cuda(cudaMalloc((void**)&d_mv_balance_sequence, sizeof(unsigned) * 1024));
   check_cuda(cudaMalloc((void**)&d_mv_delta_partition_wgt, sizeof(int) * 1024 * NUM_PARTITIONS));
   check_cuda(cudaMalloc((void**)&d_op_result, sizeof(int)));
   check_cuda(cudaMallocHost((void**)&hp_buffer_size, sizeof(unsigned))); 
   check_cuda(cudaMallocHost((void**)&hp_op_result, sizeof(int))); 
   check_cuda(cudaMallocHost((void**)&hp_pos, sizeof(unsigned))); 
   check_cuda(cudaMalloc((void**)&d_pos, sizeof(unsigned))); 
   unsigned coarsen_it = coarsen_num_vertices_vec.size();
   unsigned num_finer_vertex, num_finer_edge, num_coarsen_vertex, num_coarsen_edge, global_vertex_offset, global_ptr_offset, global_edge_offset;

   for(unsigned i = 0; i < (coarsen_it - 1); ++i) {

    num_finer_vertex = coarsen_num_vertices_vec[coarsen_it - 2 - i];
    num_finer_edge = coarsen_num_edges_vec[coarsen_it - 2 - i];
    num_coarsen_vertex = coarsen_num_vertices_vec[coarsen_it - 1 - i];
    num_coarsen_edge = coarsen_num_edges_vec[coarsen_it - 1 - i];
    global_vertex_offset = global_vertex_offset_vec[coarsen_it - 2 - i];
    global_ptr_offset = global_ptr_offset_vec[coarsen_it - 2 - i];
    global_edge_offset = global_edge_offset_vec[coarsen_it - 2 - i];

    gk::uncoarsen(d_partition, d_tmp_if_boundary, d_if_boundary, d_vertex_gain, d_max_gain_partition, d_adjncy+global_edge_offset, 
                  d_adjwgt+global_edge_offset, d_adjp+global_ptr_offset, d_cmap+global_vertex_offset, d_tmp_partition, 
                  d_if_updated_vertex, d_partition_wgt, num_finer_vertex, num_finer_edge, num_coarsen_vertex, num_coarsen_edge, stream1);

    gk::refinement(d_adjp+global_ptr_offset, d_vwgt+global_vertex_offset, d_partition, d_partition_wgt, d_adjncy+global_edge_offset, 
                   d_adjwgt+global_edge_offset, d_buffer_size, hp_buffer_size, d_pos, d_op_result, hp_pos, hp_op_result, 
                   d_if_boundary, d_vertex_gain, d_max_gain_partition, d_if_updated_vertex, d_cutsize, d_mv_buffer, 
                   d_mv_delta_partition_wgt, d_mv_balance_sequence, num_finer_vertex, num_finer_edge, NUM_PARTITIONS, stream1, context);
  }
  
   unsigned num_block = (num_finer_vertex + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
   unsigned h_cutsize;
   std::cout << "num_finer_vertex:" << num_finer_vertex << '\n';
   check_cuda(cudaMemset(d_partition_wgt, 0, sizeof(unsigned) * NUM_PARTITIONS));
   calculate_partition_wgt <<< num_block, THREAD_PER_BLOCK, 0, stream1 >>> (d_partition, d_partition_wgt, d_vwgt, num_finer_vertex);
   check_partition_wgt <<< 1, NUM_PARTITIONS, 0, stream1 >>> (d_partition_wgt);
   check_cuda(cudaMemset(d_cutsize, 0, sizeof(unsigned)));
   gk::calculate_cutsize <<< num_block, THREAD_PER_BLOCK, 0, stream1 >>> (d_partition, d_adjncy, d_adjwgt, d_adjp, d_cutsize, num_finer_vertex);
   check_cuda(cudaMemcpy(&h_cutsize, d_cutsize, sizeof(unsigned), cudaMemcpyDeviceToHost));

   //std::cout << "***** h_cutsize " << h_cutsize << '\n';

  check_cuda(cudaFree(d_mv_buffer));
  check_cuda(cudaFree(d_tmp_partition));
  check_cuda(cudaFree(d_tmp_if_boundary));
  check_cuda(cudaFree(d_buffer_size));
  check_cuda(cudaFree(d_if_updated_vertex));
  check_cuda(cudaFree(d_vertex_gain));
  check_cuda(cudaFree(d_max_gain_partition));
  check_cuda(cudaFree(d_mv_balance_sequence));
  check_cuda(cudaFree(d_mv_delta_partition_wgt));
  check_cuda(cudaFree(d_op_result));
  cudaFreeHost(hp_buffer_size);
  cudaFreeHost(hp_op_result);
  cudaFreeHost(hp_pos);
  cudaFree(d_pos);
}

void graph_partitioner(const std::string& GRAPH_FILE, const std::string& OUT_FILE, const int NUM_PARTITIONS) {
  
  //call graph parser
  gk::Graph graph_parser(GRAPH_FILE);
  const unsigned MAX_NUM_VERTICES = graph_parser.get_num_vertex();
  const unsigned MAX_NUM_EDGES = graph_parser.get_num_edge();
  const unsigned TOTAL_VWGT = graph_parser.get_total_vertex_weight();
  const unsigned MAX_COARSEN_GROUP = 6;
  const float MAX_PARTITION_WGT = (float) TOTAL_VWGT / (float) NUM_PARTITIONS * (1.0f + IMBALANCE_RATIO) + 1;
  //calculate max memory required for storing corasen graphs
  const unsigned MAX_COARSEN_LEVEL = 6;
  const float COARSEN_RATE = 0.6f;
  const size_t MAX_VERTEX_ARRAY_SIZE = static_cast<size_t>(std::ceil(MAX_NUM_VERTICES * (1 - std::pow(COARSEN_RATE, MAX_COARSEN_LEVEL)) / (1 - COARSEN_RATE)));
  const size_t MAX_EDGE_ARRAY_SIZE = static_cast<size_t>(std::ceil(MAX_NUM_EDGES * (1 - std::pow(COARSEN_RATE, MAX_COARSEN_LEVEL)) / (1 - COARSEN_RATE)));

  unsigned* d_partition;
  unsigned* d_partition_wgt;
  unsigned* d_if_boundary;
  unsigned* d_cutsize;

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  mgpu::standard_context_t context(false, stream1);

  cudaMemcpyToSymbol(D_NUM_PARTITIONS, &NUM_PARTITIONS, sizeof(int));
  cudaMemcpyToSymbol(D_TOTAL_VWGT, &TOTAL_VWGT, sizeof(int));
  cudaMemcpyToSymbol(D_MAX_PARTITION_WGT, &MAX_PARTITION_WGT, sizeof(float));
  cudaMemcpyToSymbol(D_MAX_COARSEN_GROUP, &MAX_COARSEN_GROUP, sizeof(unsigned));


  //uncoarsening variables
  check_cuda(cudaMalloc((void**)&d_partition, sizeof(unsigned) * MAX_NUM_VERTICES));
  check_cuda(cudaMalloc((void**)&d_partition_wgt, sizeof(unsigned) * NUM_PARTITIONS));
  check_cuda(cudaMalloc((void**)&d_if_boundary, sizeof(unsigned) * MAX_NUM_VERTICES));
  check_cuda(cudaMalloc((void**)&d_cutsize, sizeof(unsigned)));
  check_cuda(cudaMemset(d_partition, 0, sizeof(unsigned) * MAX_NUM_VERTICES));

  //**********************************
   //start coarsening 
  //********************************
  //malloc space for storing coarsen result
  unsigned* d_adjp;
  unsigned* d_vwgt;
  unsigned* d_adjncy;
  unsigned* d_adjwgt;
  unsigned* d_cmap;
  //malloc global memeory for storing 6 array for each coarsening it 
  check_cuda(cudaMallocAsync((void **)&d_adjp, sizeof(unsigned) * MAX_VERTEX_ARRAY_SIZE, stream1));
  check_cuda(cudaMallocAsync((void **)&d_vwgt, sizeof(unsigned) * MAX_VERTEX_ARRAY_SIZE, stream1));
  check_cuda(cudaMallocAsync((void **)&d_adjncy, sizeof(unsigned) * MAX_EDGE_ARRAY_SIZE, stream1));
  check_cuda(cudaMallocAsync((void **)&d_adjwgt, sizeof(unsigned) * MAX_EDGE_ARRAY_SIZE, stream1));
  check_cuda(cudaMallocAsync((void **)&d_cmap, sizeof(unsigned) * MAX_VERTEX_ARRAY_SIZE, stream1));
 
  std::cout << "MAX_NUM_VERTICES:" << MAX_NUM_VERTICES << '\n';

  //copy the orginal graph to
  if(MAX_NUM_VERTICES + 1 >  MAX_VERTEX_ARRAY_SIZE) {
    std::cout << "ERROR, increase MAX_VERTEX_ARRAY_SIZE to:%d" << MAX_NUM_VERTICES + 1 << '\n';
  }
  if(MAX_NUM_EDGES >  MAX_EDGE_ARRAY_SIZE) {
    std::cout << "ERROR, increase MAX_VERTEX_ARRAY_SIZE to:%d" << MAX_NUM_VERTICES << '\n';
  }

  check_cuda(cudaMemcpy(d_adjp, graph_parser.get_adjp(), sizeof(unsigned) * (MAX_NUM_VERTICES + 1), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_vwgt, graph_parser.get_vwgt(), sizeof(unsigned) * MAX_NUM_VERTICES, cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_adjncy, graph_parser.get_adjncy(), sizeof(unsigned) * MAX_NUM_EDGES, cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_adjwgt, graph_parser.get_adjwgt(), sizeof(unsigned) * MAX_NUM_EDGES, cudaMemcpyHostToDevice));

  std::vector<unsigned> coarsen_num_vertices_vec(1, MAX_NUM_VERTICES);
  std::vector<unsigned> coarsen_num_ptrs_vec(1, MAX_NUM_VERTICES + 1);
  std::vector<unsigned> coarsen_num_edges_vec(1, MAX_NUM_EDGES);
  std::vector<unsigned> global_vertex_offset_vec(1, 0);
  std::vector<unsigned> global_ptr_offset_vec(1, 0);
  std::vector<unsigned> global_edge_offset_vec(1, 0);
  int coarsen_it{0};
  auto start = std::chrono::system_clock::now();
  auto coarsen_start = std::chrono::system_clock::now();

  std::cout << "!!### MAX_PARTITION_WGT " << MAX_PARTITION_WGT << '\n';
  
  coarsening(coarsen_num_vertices_vec, coarsen_num_ptrs_vec, coarsen_num_edges_vec, global_vertex_offset_vec, 
             global_ptr_offset_vec, global_edge_offset_vec, coarsen_it, 
             d_vwgt, d_adjp, d_adjncy, d_adjwgt, d_cmap, context, 
             NUM_PARTITIONS, MAX_NUM_VERTICES, MAX_NUM_EDGES, MAX_VERTEX_ARRAY_SIZE, MAX_EDGE_ARRAY_SIZE, stream1);

  std::cout << "@@@ coarsen it: " << coarsen_it << '\n';

  cudaStreamSynchronize(stream1);
  auto coarsen_end = std::chrono::system_clock::now();

  cudaStreamSynchronize(stream1);
  auto init_start = std::chrono::system_clock::now();
  unsigned coarsen_num_vertices = coarsen_num_vertices_vec.back();
  unsigned coarsen_num_edges = coarsen_num_edges_vec.back();
  unsigned global_vertex_offset = global_vertex_offset_vec.back();
  unsigned global_ptr_offset = global_ptr_offset_vec.back();
  unsigned global_edge_offset = global_edge_offset_vec.back();
  check_cuda(cudaMemset(d_partition_wgt, 0, sizeof(unsigned) * NUM_PARTITIONS));
  std::cout <<"start init partition, num_vertices: " << coarsen_num_vertices << ", coarsen_num_edges: " << coarsen_num_edges << '\n';

  init_partition(d_vwgt+global_vertex_offset, d_adjp+global_ptr_offset, d_adjncy+global_edge_offset, 
                  d_adjwgt+global_edge_offset, d_partition, d_partition_wgt, d_if_boundary, d_cutsize, coarsen_num_vertices, 
                  coarsen_num_edges, NUM_PARTITIONS, stream1); 
  std::cout << "coarsen_num_vertices: " << coarsen_num_vertices << '\n';
  cudaStreamSynchronize(stream1);
  auto init_end = std::chrono::system_clock::now();
  cudaStreamSynchronize(stream1);
  auto refine_start = std::chrono::system_clock::now();
  uncoarsening(coarsen_num_vertices_vec, coarsen_num_edges_vec, global_vertex_offset_vec, global_ptr_offset_vec, global_edge_offset_vec,
                d_partition, d_if_boundary, d_adjncy, d_adjwgt, d_adjp, d_cmap, d_partition_wgt, d_cutsize,
                d_vwgt, NUM_PARTITIONS, MAX_NUM_VERTICES, MAX_NUM_EDGES, stream1, context);

  auto refine_end = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();

  //copy partiiton result back to CPU
  std::cout << "#################### Execution Time ################# \n";
  std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000000.0f << "s\n"; 
  std::cout << "Coarsen time: " << std::chrono::duration_cast<std::chrono::microseconds>(coarsen_end-coarsen_start).count()/1000000.0f << "s\n"; 
  std::cout << "Init partition time: " << std::chrono::duration_cast<std::chrono::microseconds>(init_end-init_start).count()/1000000.0f << "s\n"; 
  std::cout << "Refinement time: " << std::chrono::duration_cast<std::chrono::microseconds>(refine_end-refine_start).count() /1000000.0f << "s\n"; 
  std::cout << "#################### Partition Summary ################# \n";

  //check_cuda(cudaMemsetAsync(d_cutsize, 0, sizeof(int), stream1));
  //gk::calculate_cutsize <<< num_block, THREAD_PER_BLOCK, 0, stream1 >>> (d_partition, d_adjncy, d_adjwgt, d_adjp, d_cutsize, finer_num_vertices);
  unsigned h_cutsize;
  check_cuda(cudaMemcpy(&h_cutsize, d_cutsize, sizeof(int), cudaMemcpyDeviceToHost));

  std::cout << "***** h_cutsize " << h_cutsize << '\n';
  std::vector<int> vertex_partition(MAX_NUM_VERTICES, 0);
  check_cuda(cudaMemcpy(vertex_partition.data(), d_partition, sizeof(int) * MAX_NUM_VERTICES, cudaMemcpyDeviceToHost));
  graph_parser.dump_result(vertex_partition, OUT_FILE);

  cudaFree(d_partition);
  cudaFree(d_partition_wgt);
  cudaFree(d_adjp);
  cudaFree(d_vwgt);
  cudaFree(d_adjncy);
  cudaFree(d_adjwgt);
  cudaFree(d_cmap);
  cudaFree(d_if_boundary);
  cudaFree(d_cutsize);
  check_cuda(cudaStreamDestroy(stream1));
}
