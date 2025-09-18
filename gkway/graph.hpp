#pragma once
#include<numeric>
#include "../declarations.h"
//#include <filesystem>
#include <experimental/filesystem>

namespace gk { // begin of namespace gk ============================================

// ======================================================
//
// Declaration of Graph
//
// This class is used to grap graph information.
//
// ======================================================

  class Graph {

    public:

        // TODO: const std::string&
        Graph(const std::string& input_path);
        ~Graph();

        // TODO: std::vector<int>&
        const int* get_adjncy();
        const int* get_adjncy_source();
        const int* get_adjp();
        const int* get_adjwgt();
        const int* get_vwgt();
        size_t get_num_vertex();
        size_t get_num_edge();
        int get_total_vertex_weight();
        void allocate_gpu_memory();
        size_t cal_cutsize(std::vector<int> partitions);
        void create_input(const std::string& inbput_path);
        void dump_result(const std::vector<int>& vertex_partition, const std::string& file_name);

    private:

        void _parse();

        size_t _num_vertex = 0;
        size_t _num_edge = 0;
        std::string _input_path;
        std::vector<int> _adjncy; //adjacency list
        std::vector<int> _adjp; //adjacency pointer
        std::vector<int> _adjwgt; //edge weight
        std::vector<int> _vwgt; //vertx weight
        std::vector<int> _adjncy_source;
  };

// ======================================================
//
// Definition of Graph
//
// ======================================================

  // TODO: understand what the difference is between this and yours
  // in-place construction
  Graph::Graph(const std::string& input_path) : 
    _input_path {input_path} {
     _parse();
  }

  Graph::~Graph() {
  }

  // TODO: tabe size = 2
  void Graph::_parse() {
    std::ifstream file(_input_path);
    std::string line;
    int line_number = 0;
    int adjncy_count = 0;
    int vertex_count = 1;
    int format_count = 0;
    int if_weighted_edge = false;
    if(file.is_open()) {
      //std::cout << "file open " << std::endl;
      while (std::getline(file,line)) {
          //std::cout << "line number " << line_number << std::endl;
        std::istringstream ss(line);
        std::string word;
        if(line_number == 0) {
          while(ss >> word) {
            if(format_count == 0) {
             _num_vertex = std::stoi(word); 
            }
            else if(format_count == 1) {
             _num_edge = std::stoi(word) * 2; 
            }
            else if(format_count == 2) {
             if(word == "11") {
               printf("weighted graph \n");
               if_weighted_edge = true;
             }
            }
            format_count++;
          }
          printf("number of vertex %zu \n", _num_vertex);
          printf("number of edge %zu \n", _num_edge);
          //ss >> _num_vertex;
          //ss >> _num_edge;
          _adjp.resize(_num_vertex + 1, 0);
          _vwgt.resize(_num_vertex, 1);
          _adjncy.resize(_num_edge, 0);
          _adjncy_source.resize(_num_edge, 0);
          _adjwgt.resize(_num_edge, 0);
        }
        else {
          int token_count = 0;
          while (ss >> word) {
            if(if_weighted_edge == false) {
              // TODO: _adjncy[adjncy_count] vs _adjncy.at(adjncy_count)
              _adjncy[adjncy_count] = std::stoi(word);
              _adjncy_source[adjncy_count] = line_number;
              _adjwgt[adjncy_count] = 1;
              adjncy_count++;
            }
             /*
             * If the Graph's edges have weightes, each line will have the following format
             * v1 e1 v2 e2 ...
             * Where v1 is the first connected vertex and e1 is the edge weight betwwen v1 and source vertex
             * */
            else {
              if(token_count %2 == 0) {
              //std::cout << "count: " << token_count << ", word: " << word << "\n";
                _adjncy[adjncy_count] = std::stoi(word);
                _adjncy_source[adjncy_count] = line_number;
                //printf("adjncy at %d is %d \n", adjncy_count, _adjncy[adjncy_count]);
              }
              else {
                _adjwgt[adjncy_count] = std::stoi(word);
                adjncy_count++;
              }
              token_count++;
            }
          }
          if(vertex_count < _num_vertex) {
            _adjp[vertex_count] = adjncy_count;
            //_vwgt[vertex_count] = 1;
            //_adjwgt[vertex_count] = 1;
            vertex_count++;
          }
        }
        line_number++;
      }
    }
    file.close();
    //std::cout << "vertex_count " << vertex_count << ", adjncy_count "<< adjncy_count << '\n';
    _adjp[vertex_count] = adjncy_count;
  }

  const int* Graph::get_adjncy() {
    return _adjncy.data();
  }

  const int* Graph::get_adjncy_source() {
    return _adjncy_source.data();
  }

  const int* Graph::get_adjp() {
    return _adjp.data();
  }

  const int* Graph::get_adjwgt() {
    return _adjwgt.data();
  }

  const int* Graph::get_vwgt() {
    return _vwgt.data();
  }

  int Graph::get_total_vertex_weight() {
    return std::accumulate(_vwgt.begin(), _vwgt.end(), 0);
  }
  size_t Graph::get_num_vertex() {
    return _num_vertex;
  }

  size_t Graph::get_num_edge() {
    return _num_edge;
  }

  size_t Graph::cal_cutsize(std::vector<int> partitions) {
    size_t cutsize = 0;
    for(int i = 0; i < _num_vertex; ++i) {
      int start_adj = _adjp[i];
      int end_adj = _adjp[i+1];
      int vertex_partition = partitions[i];
      for(int j = start_adj; j < end_adj; ++j) {
        if(i < _adjncy[j]) {
          //chcek if the edge is cut
          if(vertex_partition != partitions[_adjncy[j]]){
            //std::cout << "vertex_id: " << i << ", adjncy: " << _adjncy[j] << ", _adjw: " << _adjwgt[j] << '\n';
            //cutsize += _adjwgt[j]
            cutsize += 1;
          }
        }
      }
    }

    return cutsize;
  }

  void Graph::create_input(const std::string& input_name) {
    std::string file_path = "/home/wanluanlee/Documents/G-kway/build/exec/out";
    std::ofstream out_file(file_path);
    if(!out_file) {
      std::cerr << "can't open output file \n";
    }
    std::cout << "write file \n";
    out_file << _num_edge << " " << _num_vertex << '\n';

    for(int i = 0; i < _vwgt.size(); i++) {
      int start = _adjp[i];
      int end = _adjp[i+1];
      for(int j = start; j < end; j++) {
        out_file << i << " " << j << '\n';
      }
    }
    out_file.flush();
    out_file.close();   

  }

  void Graph::dump_result(const std::vector<int>& vertex_partition, const std::string& file_name) {
    std::experimental::filesystem::path cwd = std::experimental::filesystem::current_path();
    std::string out_path = cwd.string() + "/" + file_name + ".out";
    std::cout << "out_path:" << out_path << "\n";
    std::ofstream out_file(out_path);
    if(!out_file) {
      std::cerr << "can't open output file \n";
    }
    for(int i = 0; i < vertex_partition.size(); i++) {
      out_file << vertex_partition[i]; 
      out_file << '\n';
    }
    out_file.close();
  }

} // end of namespace gk ==========================================
