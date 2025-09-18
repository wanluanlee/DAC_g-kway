#include "../declarations.h"
#include "../gkway/graph_partitioner.hpp"
#include<string>
#include <sys/resource.h>
int main(int argc, char** argv) {

  if(argc != 3) {
    std::cerr << "usage: ./gkway graph_file num_partition \n";
  }

  const std::string GRAPH_FILE = argv[1];
  //std::cout << "now process:" << GRAPH_FILE << '\n';
  const int NUM_PARTITIONS = std::stoi(argv[2]);
  const std::string OUT_FILE = argv[3];

  //graph_partitioner(GRAPH_FILE, NUM_PARTITION, COARSEN_RATIO, OUT_FILE, PATH_FILE);
  graph_partitioner(GRAPH_FILE, OUT_FILE, NUM_PARTITIONS);

  return 0;
}


