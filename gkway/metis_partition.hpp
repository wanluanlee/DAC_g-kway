#include "metis/metis.h"


void metis_init_partition(unsigned* d_vwgt, unsigned* d_adjp, unsigned* d_adjncy, unsigned* d_adjwgt, unsigned* d_partition, unsigned* d_partition_wgt, 
  unsigned* d_if_boundary, unsigned* d_cutsize, const unsigned NUM_VERTICES, const unsigned NUM_EDGES, const unsigned NUM_PARTITIONS, cudaStream_t stream1) {

  //copy data to host
  unsigned* h_vwgt = (unsigned*) malloc(sizeof(unsigned) * NUM_VERTICES);
  unsigned* h_adjp = (unsigned*) malloc(sizeof(unsigned) * (NUM_VERTICES + 1));
  unsigned* h_adjncy = (unsigned*) malloc(sizeof(unsigned) * NUM_EDGES);
  unsigned* h_adjwgt = (unsigned*) malloc(sizeof(unsigned) * NUM_EDGES);
  unsigned* h_partition = (unsigned*) malloc(sizeof(unsigned) * NUM_VERTICES);
  unsigned* h_if_boundary = (unsigned*) malloc(sizeof(unsigned) * NUM_VERTICES);
  unsigned* h_partition_wgt = (unsigned*) malloc(sizeof(unsigned) * NUM_PARTITIONS);
  memset(h_partition_wgt, 0, sizeof(unsigned) * NUM_PARTITIONS);
  memset(h_if_boundary, 0, sizeof(unsigned) * NUM_VERTICES);
  check_cuda(cudaMemcpy(h_vwgt, d_vwgt, sizeof(int) * NUM_VERTICES, cudaMemcpyDeviceToHost));
  check_cuda(cudaMemcpy(h_adjp, d_adjp, sizeof(int) * (NUM_VERTICES + 1), cudaMemcpyDeviceToHost));
  check_cuda(cudaMemcpy(h_adjncy, d_adjncy, sizeof(int) * NUM_EDGES, cudaMemcpyDeviceToHost));
  check_cuda(cudaMemcpy(h_adjwgt, d_adjwgt, sizeof(int) * NUM_EDGES, cudaMemcpyDeviceToHost));

 

  std::vector<idx_t> mt_vwgt(NUM_VERTICES);
  std::vector<idx_t> mt_adjp(NUM_VERTICES + 1);
  std::vector<idx_t> mt_adjncy(NUM_EDGES * 2);
  std::vector<idx_t> mt_adjwgt(NUM_EDGES * 2);
  std::vector<idx_t> mt_partition(NUM_VERTICES);

  idx_t mt_num_vertex = NUM_VERTICES;
  idx_t ncon = 1;
  idx_t mt_num_part = NUM_PARTITIONS;
  idx_t mt_cutsize;
  //std::cout << "start copy vwgt and adjp \n";
  for(int i = 0; i < NUM_VERTICES; i++) {
    mt_vwgt[i] = (idx_t) h_vwgt[i];
    mt_adjp[i] = (idx_t) h_adjp[i];
    h_if_boundary[i] = 0;
  }
  mt_adjp[NUM_VERTICES] = (idx_t) h_adjp[NUM_VERTICES];

  for(int i = 0; i < NUM_EDGES; i++) {
    mt_adjncy[i] = (idx_t) (h_adjncy[i] - 1);
    mt_adjwgt[i] = (idx_t) h_adjwgt[i];  
  }
  int res = METIS_PartGraphKway(&mt_num_vertex, &ncon, mt_adjp.data(), mt_adjncy.data(), mt_vwgt.data(), NULL, mt_adjwgt.data(), 
                                &mt_num_part, NULL, NULL, NULL, &mt_cutsize, mt_partition.data());

  printf ( "\n" );
  printf ( "  Return code = %d\n", res );
  printf ( "  Edge cuts for partition = %d\n", ( int ) mt_cutsize );
  for ( unsigned part_i = 0; part_i < NUM_VERTICES; part_i++ )
  {
    //printf ( "     %d     %d\n", part_i, ( int ) mt_partition[part_i] );
    h_partition[part_i] = (int) mt_partition[part_i];
    h_partition_wgt[h_partition[part_i]] += h_vwgt[part_i];
  }

  //set if boundary array
  for(int i = 0; i < NUM_VERTICES; i++) {
    int start = h_adjp[i];
    int end = h_adjp[i+1];
    int partition = h_partition[i];
    for(int j = start; j < end; j++) {
      int neighbor_partition = h_partition[h_adjncy[j] - 1];
      if(partition != neighbor_partition) {
        h_if_boundary[i] = 1;
        break;
      }
    }
  }

  //for(int i = 0; i < num_vertex; i++) {
    //std::cout << "id:" << i << ", if_boundary: " << h_if_boundary[i] << '\n';
  //}

  ////std::cout << "######### graph info ############# \n";
  //for(unsigned i = 0; i < num_vertex; i++) {
    //std::cout << "vertex_id:" << (i + 1) << ", vertex partition:" << h_partition[i] << '\n';
  //}

  for(int i = 0; i < NUM_PARTITIONS; i++) {
    std::cout << "partition: " << i << ", wgt: " << h_partition_wgt[i] << '\n';
  }

  //copy partition result back to device
  cudaMemcpy(d_partition, h_partition, sizeof(unsigned) * NUM_VERTICES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_partition_wgt, h_partition_wgt, sizeof(unsigned) * NUM_PARTITIONS, cudaMemcpyHostToDevice);
  cudaMemcpy(d_if_boundary, h_if_boundary, sizeof(unsigned) * NUM_VERTICES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cutsize, &mt_cutsize, sizeof(unsigned), cudaMemcpyHostToDevice);

  free(h_vwgt);
  free(h_adjp);
  free(h_adjwgt);
  free(h_adjncy);
  free(h_partition);
  free(h_if_boundary);
  free(h_partition_wgt);
}
