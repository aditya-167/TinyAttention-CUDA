#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#pragma once
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8

# define BD 1

#define tilesize 32

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}


__global__ void sgemm_naive(int M, int N, int K, float *A,
                            float *B, float *C) {
    // this kernel uses x as the row index and y as the column index (which leads to bad coalescing)
  // compute position in C that this thread is responsible for
// A is MxK, B is KxN, C is MxN

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];

    }
    // C = α*(A@B)+β*C
    C[x * N + y] = tmp;
  }
}


__global__ void sgemm_naive_batched(int L, int M, int N, int K, float *A,
                            float *B, float *C) {
    // this kernel uses x as the row index and y as the column index (which leads to bad coalescing)
  // compute position in C that this thread is responsible for
// A is MxK, B is KxN, C is MxN

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint batch = blockIdx.z * blockDim.z + threadIdx.z;

  const uint offset_A = batch * M * K;
  const uint offset_B = batch * K * N;
  const uint offset_C = batch * M * N;

  if (batch >= L) {
    return;
  }
  

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[offset_A + x * K + i] * B[offset_B + i * N + y];

    }
    // C = α*(A@B)+β*C
    C[offset_C + x * N + y] = tmp;
  }
}


__global__ void sgemm_naive_coalesced(int M, int N, int K, float *A,
                            float *B, float *C) {
    // this kernel uses x as the col index and y as the row index (which leads to better coalescing) and a decent speedup
  // compute position in C that this thread is responsible for
// A is MxK, B is KxN, C is MxN

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < N && y < M) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[y*K+i] * B[i*N+x];

    }
    // C = α*(A@B)+β*C
    C[y*N+x] = tmp;
  }
}


__global__ void sgemm_naive_coalesced_batched(int L, int M, int N, int K, float *A,
                            float *B, float *C) {
    // this kernel uses x as the col index and y as the row index (which leads to better coalescing) and a decent speedup
  // compute position in C that this thread is responsible for
// A is MxK, B is KxN, C is MxN

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint batch = blockIdx.z * blockDim.z + threadIdx.z;

  const uint offset_A = batch * M * K;
  const uint offset_B = batch * K * N;
  const uint offset_C = batch * M * N;

  if (batch >= L) {
    return;
  }
  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < N && y < M) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[offset_A+y*K+i] * B[offset_B+i*N+x];

    }
    // C = α*(A@B)+β*C
    C[offset_C+y*N+x] = tmp;
  }
}


__global__ void sgemm_naive_coalesced_tiled(int M, int N, int K, float *A,
                            float *B, float *C) {
    // assign smem
    __shared__ float As[tilesize][tilesize];
    __shared__ float Bs[tilesize][tilesize];

    // this kernel uses x as the col index and y as the row index (which leads to better coalescing) and a decent speedup
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;


    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;

    const uint blocksize=blockDim.x;

    
    float val = 0.0;
    // loop over phases of tiling 
    for(int i=0;i<K;i+=blocksize) {
        //load to shared mem
        
        if (y<M && i+tx<K){

            As[ty][tx] = A[y*K+i+tx];
        }
        else{
            As[ty][tx] = 0;
        }
        if (x<N && i+ty<K){
            Bs[ty][tx] = B[(i+ty)*N+x];
        }
        else{
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        for(int j=0;j<blocksize;j++) {
            val += As[ty][j]*Bs[j][tx];
        }
        __syncthreads();
    }
    C[y*N+x] = val;
}




__global__ void sgemm_naive_coalesced_tiled_batched(int L, int M, int N, int K, float *A,
                            float *B, float *C) {
    // assign smem
    __shared__ float As[tilesize][tilesize];
    __shared__ float Bs[tilesize][tilesize];

    

    // this kernel uses x as the col index and y as the row index (which leads to better coalescing) and a decent speedup
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint batch = blockIdx.z * blockDim.z + threadIdx.z;

    const uint offset_A = batch * M * K;
    const uint offset_B = batch * K * N;
    const uint offset_C = batch * M * N;

    if (batch >= L) {
      return;
    }

    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;

    const uint blocksize=blockDim.x;

    
    float val = 0.0;
    // loop over phases of tiling 
    for(int i=0;i<K;i+=blocksize) {
        //load to shared mem
        
        if (y<M && i+tx<K){

            As[ty][tx] = A[offset_A+y*K+i+tx];
        }
        else{
            As[ty][tx] = 0;
        }
        if (x<N && i+ty<K){
            Bs[ty][tx] = B[offset_B+(i+ty)*N+x];
        }
        else{
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        for(int j=0;j<blocksize;j++) {
            val += As[ty][j]*Bs[j][tx];
        }
        __syncthreads();
    }
    C[offset_C+y*N+x] = val;
}


__global__ void sgemm_naive_coalesced_tiled_2dcoarsened(int M, int N, int K, float *A,
                            float *B, float *C) {
    // assign smem
    __shared__ float As[tilesize][tilesize];
    __shared__ float Bs[tilesize][tilesize];

    // this kernel uses x as the col index and y as the row index (which leads to better coalescing) and a decent speedup
    // compute position in C that this thread is responsible for
    // Each kernel is responsible for 
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;


    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;

    const uint blocksize=blockDim.x;

    
    float val = 0.0;
    // loop over phases of tiling 
    for(int i=0;i<K;i+=blocksize) {
        //load to shared mem
        
        if (y<M && i+tx<K){

            As[ty][tx] = A[y*K+i+tx];
        }
        else{
            As[ty][tx] = 0;
        }
        if (x<N && i+ty<K){
            Bs[ty][tx] = B[(i+ty)*N+x];
        }
        else{
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        for(int j=0;j<blocksize;j++) {
            val += As[ty][j]*Bs[j][tx];
        }
        __syncthreads();
    }
    C[y*N+x] = val;


}
__global__ void sgemmVectorize(int M, int N, int K, float *A,
                               float *B, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = threadResults[resIdxM * TN + resIdxN];
      tmp.y =  threadResults[resIdxM * TN + resIdxN + 1];
      tmp.z =  threadResults[resIdxM * TN + resIdxN + 2];
      tmp.w =  threadResults[resIdxM * TN + resIdxN + 3];
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, const float *A,
                       const float *B,float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  
  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;
  const uint strideA = numThreadsBlocktile / BK;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
  float a;
    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
             regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    //a =     regM[2];
    //printf("a: %f\n", a);
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          threadResults[resIdxM * TN + resIdxN];
    }
  }
}


__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling_batched(int L, int M, int N, int K, const float *A,
                       const float *B,float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  
  const uint batch = blockIdx.z * blockDim.z + threadIdx.z;

  const uint offset_A = batch * M * K;
  const uint offset_B = batch * K * N;
  const uint offset_C = batch * M * N;
  

  if (batch >= L) {
    return;
  }
  
  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K + offset_A;
  B += cCol * BN + offset_B;
  C += cRow * BM * N + cCol * BN + offset_C;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;
  const uint strideA = numThreadsBlocktile / BK;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
  
  
    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
             regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    //a =     regM[2];
    //printf("a: %f\n", a);
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          threadResults[resIdxM * TN + resIdxN];
    }
  }
}

void run_sgemm_naive(torch::Tensor A, torch::Tensor B, torch::Tensor C){
    dim3 gridDim(CEIL_DIV(B.size(3), 32), CEIL_DIV(A.size(2), 32));
    dim3 blockDim(32,32);

    // loop over batchsize and head
    for (int i = 0; i < A.size(0); i++) {
        for (int j = 0; j < A.size(1); j++) {
            // get the i-th batch and j-th head
            torch::Tensor Aij = A[i][j];
            torch::Tensor Bij = B[i][j];
            torch::Tensor Cij = C[i][j];
            // compute the matrix multiplication
            sgemm_naive<<<gridDim, blockDim>>>(A.size(2), B.size(3), A.size(3), Aij.data_ptr<float>(), Bij.data_ptr<float>(), Cij.data_ptr<float>());
            // allocate memory for output on GPU in cuda
        }
    }

}

void run_sgemm_naive_batched(torch::Tensor A, torch::Tensor B, torch::Tensor C){
    
    dim3 gridDim(CEIL_DIV(B.size(3), 32), CEIL_DIV(A.size(2), 32), CEIL_DIV(A.size(0)*A.size(1), BD));
    dim3 blockDim(32,32,BD);
    int L = A.size(0)*A.size(1);
    sgemm_naive_batched<<<gridDim, blockDim>>>(L, A.size(2), B.size(3), A.size(3), A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
    return;

}

void run_sgemm_coalesced(torch::Tensor A, torch::Tensor B, torch::Tensor C){
    dim3 gridDim(CEIL_DIV(B.size(3), 32), CEIL_DIV(A.size(2), 32));
    dim3 blockDim(32,32);

    // loop over batchsize and head
    for (int i = 0; i < A.size(0); i++) {
        for (int j = 0; j < A.size(1); j++) {
            // get the i-th batch and j-th head
            torch::Tensor Aij = A[i][j];
            torch::Tensor Bij = B[i][j];
            torch::Tensor Cij = C[i][j];
            // compute the matrix multiplication
            sgemm_naive_coalesced<<<gridDim, blockDim>>>(A.size(2), B.size(3), A.size(3), Aij.data_ptr<float>(), Bij.data_ptr<float>(), Cij.data_ptr<float>());
            // allocate memory for output on GPU in cuda
        }
    }
}

void run_sgemm_coalesced_batched(torch::Tensor A, torch::Tensor B, torch::Tensor C){
    dim3 gridDim(CEIL_DIV(B.size(3), 32), CEIL_DIV(A.size(2), 32), CEIL_DIV(A.size(0)*A.size(1),BD));
    dim3 blockDim(32,32,BD);
    int L = A.size(0)*A.size(1);
    sgemm_naive_coalesced_batched<<<gridDim, blockDim>>>(L, A.size(2), B.size(3), A.size(3), A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
    return;

}

void run_sgemm_coalesced_tiled(torch::Tensor A, torch::Tensor B, torch::Tensor C){
    dim3 gridDim(CEIL_DIV(B.size(3), 32), CEIL_DIV(A.size(2), 32));
    dim3 blockDim(32,32);

    // loop over batchsize and head
    for (int i = 0; i < A.size(0); i++) {
        for (int j = 0; j < A.size(1); j++) {
            // get the i-th batch and j-th head
            torch::Tensor Aij = A[i][j];
            torch::Tensor Bij = B[i][j];
            torch::Tensor Cij = C[i][j];
            // compute the matrix multiplication
            sgemm_naive_coalesced_tiled<<<gridDim, blockDim>>>(A.size(2), B.size(3), A.size(3), Aij.data_ptr<float>(), Bij.data_ptr<float>(), Cij.data_ptr<float>());
            // allocate memory for output on GPU in cuda
        }
    }

}

void run_sgemm_coalesced_tiled_batched(torch::Tensor A, torch::Tensor B, torch::Tensor C){
    dim3 gridDim(CEIL_DIV(B.size(3), 32), CEIL_DIV(A.size(2), 32), CEIL_DIV(A.size(0)*A.size(1),BD));
    dim3 blockDim(32,32,BD);
    int L = A.size(0)*A.size(1);
    sgemm_naive_coalesced_tiled_batched<<<gridDim, blockDim>>>(L, A.size(2), B.size(3), A.size(3), A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
    return;

}

void run_sgemm_blocktiling(torch::Tensor A, torch::Tensor B, torch::Tensor C){
    dim3 gridDim(CEIL_DIV(B.size(3), BN), CEIL_DIV(A.size(2), BM));
    dim3 blockDim(CEIL_DIV(BM * BN, (TM * TN)));

    // loop over batchsize and head
    for (int i = 0; i < A.size(0); i++) {
        for (int j = 0; j < A.size(1); j++) {
            // get the i-th batch and j-th head
            torch::Tensor Aij = A[i][j];
            torch::Tensor Bij = B[i][j];
            torch::Tensor Cij = C[i][j];
            // compute the matrix multiplication
            sgemm_naive_coalesced<<<gridDim, blockDim>>>(A.size(2), B.size(3), A.size(3), Aij.data_ptr<float>(), Bij.data_ptr<float>(), Cij.data_ptr<float>());
            // allocate memory for output on GPU in cuda
        }
    }

}

void run_sgemm_blocktiling_batched(torch::Tensor A, torch::Tensor B, torch::Tensor C){
    dim3 gridDim(CEIL_DIV(B.size(3), BN), CEIL_DIV(A.size(2), BM), CEIL_DIV(A.size(0)*A.size(1),BD));
    dim3 blockDim((BM * BN)/(TM * TN), BD);
    int L = A.size(0)*A.size(1);
    sgemm2DBlocktiling_batched<<<gridDim, blockDim>>>(L, A.size(2), B.size(3), A.size(3), A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
    return;
    }



torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    double start, end;
    start = getTimeStamp();
    torch::Tensor C = torch::zeros({A.size(0), A.size(1), A.size(2), B.size(3)}, torch::kCUDA);
    


    
    // A = (batchsize, head, M, K)
    // B = (batchsize, head, K, N)
    // C = (batchsize, head, M, N)

    // Goal is for each batch, head compute A[batch,head] @ B[batch,head]
    // + 20% speedup over nontiled coalesced
    // + 500% speedup by using threadcoarsening and tiled together! (code is a freakin mess tho)
    
    // further small speed up by using he bathced versions of the kernels

    // streaming not really beneficial probably as we dont have any data loading happening
    
    run_sgemm_blocktiling_batched(A, B, C);
    cudaDeviceSynchronize();
    end = getTimeStamp();
    printf("Time taken: %lf\n", (end-start));
    return C;
}
