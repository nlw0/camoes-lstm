#!/bin/bash
# RunPod's image puts system CUDA on LD_LIBRARY_PATH. CUDA.jl then mixes
# its own libcublas with the system libcublasLt and matmul crashes.
# Keep the NVIDIA driver libs, drop /usr/local/cuda/lib64.
set -euo pipefail
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
cd "$(dirname "$0")"
exec julia --project=/root/.julia/environments/v1.13 camoes.jl "$@"
