#include "../declarations.h"
#pragma once 
  struct mvRequest {
    unsigned vertex_id;
    unsigned source_partition;
    unsigned des_partition;
    int gain;

    mvRequest() = default;

    __host__ __device__
    mvRequest(unsigned vertex_id, 
              unsigned source_partition, 
              unsigned des_partition, 
              int gain) : 
              vertex_id(vertex_id), 
              source_partition(source_partition), 
              des_partition(source_partition), 
              gain(gain) {};

    __host__ __device__
    mvRequest& operator=(const mvRequest& other) {
      vertex_id = other.vertex_id;
      source_partition = other.source_partition;
      des_partition = other.des_partition;
      gain = other.gain;
      return *this;
    };

    __host__ __device__
    bool operator==(const mvRequest& other) const {
      return (vertex_id == other.vertex_id);
    };

    __host__ __device__
    bool operator <(mvRequest& rhs) const {
      if(gain == rhs.gain) {
        return vertex_id < rhs.vertex_id;
      }
      return gain > rhs.gain;
    }
  };

