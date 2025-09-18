# G-kway: Multilevel GPU-Accelerated *k*-way Graph Partitioner

G-kway is a GPU-accelerated multilevel *k*-way graph partitioner designed for very large graphs. It introduces (1) a **union-find–based coarsening with scoring** that merges many vertices per level, and (2) an **independent-set–based refinement** that moves many boundary vertices in parallel—both implemented with modern CUDA warp-level primitives and CSR graph storage. On large industrial and DIMACS graphs, G-kway achieved strong speedups over 32-thread CPU baselines and prior GPU partitioners, with comparable or better cut quality.

## Features
- **Union-find coarsening with scoring** – merges multiple vertices per level to reduce the number of levels while preserving structure; prioritizes neighbors by edge weight and degree to avoid imbalanced supernodes.
- **Independent-set refinement** – selects a parallelizable set of boundary vertices, sorts moves by gain, and applies a balanced subsequence of moves without exponential enumeration.
- **GPU-friendly implementation** – CUDA 12 kernels using warp collectives (`__shfl*`, `__ballot*`, `__reduce_*_sync`), pinned memory, and CSR storage.
- **Scales to industry-sized graphs** – evaluated on large circuit timing graphs and classic DIMACS graphs (e.g., *delaunay*, *ldoor*, *asia.osm*).

## Paper
**G-kway: Multilevel GPU-Accelerated *k*-way Graph Partitioner** (DAC 2024)  
Wan-Luan Lee, Dian-Lun Lin, Tsung-Wei Huang, Shui Jiang, Tsung-Yi Ho, Yibo Lin, Bei Yu.  
If you use this code, please cite the paper—see [Citing](#citing).

## Requirements
- **CUDA**: 12.0+  
- **GPU**: NVIDIA GPU with compute capability ≥ 7.0 recommended (e.g., Turing/Ampere)  
- **Compiler**: GCC 8+ / Clang 10+ (host), `nvcc` (device)  
- **CMake**: 3.18+ (if using the provided CMake build)  
- **OS**: Linux x86_64 (tested)

## Getting Started
Clone the repository **with submodules** (ModernGPU):
```bash
git clone --recurse-submodules https://github.com/wanluanlee/DAC_g-kway.git
cd DAC_g-kway
# If you forgot --recurse-submodules:
git submodule update --init --recursive
```
## Build
Using CMake (recommended):
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

This will produce the main executable /exec/g-kway

## Run
```bash
./g-kway  /path/to/graph  #partitions /path/to/out
```

## Input Format
We follow the academic-standard graph format used by **METIS** and **mt-metis** for inputs.

## Citing
If you use, compare against, or build upon G-kway, please cite:

```
@inproceedings{Lee2024Gkway,
  title     = {G-kway: Multilevel GPU-Accelerated k-way Graph Partitioner},
  author    = {Wan-Luan Lee and Dian-Lun Lin and Tsung-Wei Huang and Shui Jiang and Tsung-Yi Ho and Yibo Lin and Bei Yu},
  booktitle = {Proceedings of the 61st ACM/IEEE Design Automation Conference (DAC '24)},
  year      = {2024}
}
```

## License
Specify your license here (e.g., MIT/BSD/Apache-2.0). Include third-party licenses for ModernGPU in `3rd-party/moderngpu/`.

## Acknowledgments
This work benefited from prior open-source efforts in graph processing and GPU programming, and from support acknowledged in the paper.
