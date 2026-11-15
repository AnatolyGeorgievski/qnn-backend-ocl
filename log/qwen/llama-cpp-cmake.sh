#!/usr/bin/env bash
# Снимок параметров сборки llama.cpp на arc3 (2026-11-14).
# Источник: ~/src/llama.cpp/build/CMakeCache.txt · git 0cce758
# Не запускать вслепую с других машин — пути CUDA/arch под Blackwell.

set -euo pipefail
SRC="${SRC:-$HOME/src/llama.cpp}"
BUILD="${BUILD:-$SRC/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_NCCL=ON \
  -DGGML_CUDA_COMPRESSION_MODE=size \
  -DGGML_OPENMP=ON \
  -DGGML_NATIVE=ON \
  -DGGML_RPC=OFF \
  -DGGML_OPENCL=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_BLAS=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_OPENSSL=ON \
  -DBUILD_SHARED_LIBS=ON

cmake --build "$BUILD" -j"$(nproc)"
