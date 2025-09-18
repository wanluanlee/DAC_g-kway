#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <bits/stdc++.h>
#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define MAX_SHARE_SIZE 12000
#define THREAD_PER_BLOCK 512
#define MAX_MV_BUFFER_SIZE 1024
#define VERTEX_PER_BLOCK 32
#define WARP_VERTEX_THREAD_PER_BLOCK 1024
#define IMBALANCE_RATIO 0.03
#define WARP_SIZE_SHIFT 5
__constant__ unsigned D_NUM_PARTITIONS; 
__constant__ unsigned D_MAX_COARSEN_GROUP;
__constant__ size_t D_TOTAL_VWGT;   
__constant__ float D_MAX_PARTITION_WGT;
namespace gk {
}

