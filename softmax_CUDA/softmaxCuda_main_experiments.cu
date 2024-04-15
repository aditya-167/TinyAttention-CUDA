#include <torch/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include <ctime>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

//Optimized
const int TILE_DIM_Y = 32;  // Tile dimension for rows
const int TILE_DIM_X = 32;  // Tile dimension for columns// must be 32 for this method
const int BLOCK_SIZE = 32;


#include <iostream>
#include <math.h>
/*
__global__ void softmax_kernel_coalesced_coarsened(float *input, float *output, int rows, int cols, int coarsening_factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows) {
        float max_val = input[idx * cols];
        for (int i = 1; i < cols; i++) {
            float val = input[idx * cols + i];
            max_val = (val > max_val) ? val : max_val;
        }

        float sum_exp = 0.0f;
        // Thread coarsening: each thread handles multiple elements
        for (int i = 0; i < cols; i += coarsening_factor) {
            float exp_sum = 0.0f;
            // Compute the sum of exponentials for the coarsened group
            for (int j = 0; j < coarsening_factor && i + j < cols; j++) {
                float exp_val = expf(input[idx * cols + i + j] - max_val);
                output[idx * cols + i + j] = exp_val;
                exp_sum += exp_val;
            }
            // Accumulate the sum of exponentials for normalization
            sum_exp += exp_sum;
        }
        // Normalize the softmax values
        for (int i = 0; i < cols; i += coarsening_factor) {
            for (int j = 0; j < coarsening_factor && i + j < cols; j++) {
                output[idx * cols + i + j] /= sum_exp;
            }
        }
    }
}

__global__ void softmax_kernel_naive(float *input, float *output, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows) {
        float max_val = input[idx * cols];
        for (int i = 1; i < cols; i++) {
            if (input[idx * cols + i] > max_val) {
                max_val = input[idx * cols + i];
            }
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum_exp += expf(input[idx * cols + i] - max_val);
        }
        for (int i = 0; i < cols; i++) {
            output[idx * cols + i] = expf(input[idx * cols + i] - max_val) / sum_exp;
        }
    }
}

__global__ void softmax2D_kernel(float *d_in, float *d_out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Find max value in the row
        float max_val = d_in[row * N];
        for (int i = 1; i < N; ++i) {
            max_val = fmaxf(max_val, d_in[row * N + i]);
        }
        
        // Subtract max value from each element for numerical stability
        float sum_exp = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum_exp += expf(d_in[row * N + i] - max_val);
        }

        // Calculate softmax for each element in the row
        d_out[row * N + col] = expf(d_in[row * N + col] - max_val) / sum_exp;
    }
}



template <typename T>
__global__ void softmaxKernel2D_rows(const T* input, T* exp_sums, int N, int M) {
    int row = blockIdx.y * TILE_DIM_Y + threadIdx.y;
    int col = blockIdx.x * TILE_DIM_X + threadIdx.x;
    T val = 0;
    // Copy data from global memory to shared memory
    if (row < N && col < M) {
        T max_val = input[row * M];  // Initialize max_val with the first element of the row
        for (int i = 1; i < M; ++i) {
            max_val = max(max_val, input[row * M + i]);
        }
        if (sizeof(T) == 8)
            val = exp(input[row * M + col] - max_val);
        else
            val = expf(input[row * M + col] - max_val);
    }
    // warp shuffle reduction
    // Use XOR mode to perform butterfly reduction
    for (int i = 16; i >= 1; i >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, i, 32);
    // update global value for row
    if ((threadIdx.x == 0) && (row < N)) atomicAdd(exp_sums + row, val);
}

template <typename T>
__global__ void softmaxKernel2D_elementwise(const T* input, const T* exp_sums, T* output, int N, int M) {
    int row = blockIdx.y * TILE_DIM_Y + threadIdx.y;
    int col = blockIdx.x * TILE_DIM_X + threadIdx.x;
    // Compute the softmax values
    if (row < N && col < M) {
        T max_val = input[row * M];  // Initialize max_val with the first element of the row
        for (int i = 1; i < M; ++i) {
            max_val = max(max_val, input[row * M + i]);
        }
        T exp_sum_row = exp_sums[row];
        if (sizeof(T) == 8)
            output[row * M + col] = exp(input[row * M + col] - max_val) / exp_sum_row;
        else
            output[row * M + col] = expf(input[row * M + col] - max_val) / exp_sum_row;
    }
}




__global__ void softmax_kernel_naive_batched(float *input, float *output, int batch_size, int n_head, int seq_len, int head_embd) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batch_size * n_head) {
        int batch_idx = idx / n_head;
        int head_idx = idx % n_head;

        for (int i = 0; i < seq_len; i++) {
            // Compute max value within the sequence for each batch and head
            float max_val = input[(batch_idx * n_head * seq_len * head_embd) + (head_idx * seq_len * head_embd) + i * head_embd];
            for (int j = 1; j < head_embd; j++) {
                float val = input[(batch_idx * n_head * seq_len * head_embd) + (head_idx * seq_len * head_embd) + i * head_embd + j];
                if (val > max_val) {
                    max_val = val;
                }
            }

            // Compute softmax for each element in the sequence
            float sum_exp = 0.0f;
            for (int j = 0; j < head_embd; j++) {
                float val = input[(batch_idx * n_head * seq_len * head_embd) + (head_idx * seq_len * head_embd) + i * head_embd + j];
                sum_exp += expf(val - max_val);
            }

            for (int j = 0; j < head_embd; j++) {
                float val = input[(batch_idx * n_head * seq_len * head_embd) + (head_idx * seq_len * head_embd) + i * head_embd + j];
                output[(batch_idx * n_head * seq_len * head_embd) + (head_idx * seq_len * head_embd) + i * head_embd + j] = expf(val - max_val) / sum_exp;
            }
        }
    }
}
*/

void softmax_cudnn(float *input, float *output, int num_samples, int num_classes) {
    // Set up cuDNN
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_samples, num_classes, 1, 1);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_samples, num_classes, 1, 1);
    
    // Perform softmax operation
    float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_desc, input, &beta, output_desc, output);
    
    // Clean up
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroy(cudnn);
}
/*
void run_softmax_naive(torch::Tensor A, torch::Tensor C){
    const int seq_len = A.size(2);
    const int head_embd = A.size(3);
    const int M = seq_len;
    const int N = head_embd;
    int threadsPerBlock = 256;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(ceil((M + blockDim.x - 1) / (float)blockDim.x), ceil((N + blockDim.y - 1) / (float)blockDim.y));

    // loop over batchsize and head
    for (int i = 0; i < A.size(0); i++) {
        for (int j = 0; j < A.size(1); j++) {
            // get the i-th batch and j-th head
            torch::Tensor Aij = A[i][j];
            torch::Tensor Cij = C[i][j];
            // compute the softmax


            softmax_kernel_naive<<<gridDim, blockDim>>>(Aij.data_ptr<float>(), Cij.data_ptr<float>(), M, N);

        }
    }

}
void run_softmax_batched_naive(torch::Tensor A, torch::Tensor C){
    auto A_data = A.data_ptr<float>();
    auto C_data = C.data_ptr<float>();
    const int batch_size = A.size(0);
    const int n_head = A.size(1);
    const int seq_len = A.size(2);
    const int head_embd = A.size(3);
    const int M = seq_len;
    const int N = head_embd;

    int threadsPerBlock = 256;
    dim3 threadsperblock(threadsPerBlock);

    dim3 blocksPerGrid ((batch_size * n_head + threadsPerBlock - 1) / threadsPerBlock);


    softmax_kernel_naive_batched<<<blocksPerGrid, threadsperblock>>>(A_data, C_data, batch_size, n_head, seq_len, head_embd);


}
*/
void run_softmax_cuDNN(torch::Tensor A, torch::Tensor C){
    const int seq_len = A.size(2);
    const int head_embd = A.size(3);
    const int M = seq_len;
    const int N = head_embd;
    int threadsPerBlock = 256;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(ceil((M + blockDim.x - 1) / (float)blockDim.x), ceil((N + blockDim.y - 1) / (float)blockDim.y));

    // loop over batchsize and head
    for (int i = 0; i < A.size(0); i++) {
        for (int j = 0; j < A.size(1); j++) {
            // get the i-th batch and j-th head
            torch::Tensor Aij = A[i][j];
            torch::Tensor Cij = C[i][j];

            // compute the softmax

            softmax_cudnn(Aij.data_ptr<float>(), Cij.data_ptr<float>(), M, N);

            //softmaxKernel2D_rows<<<blocks, threads>>>(Aij.data_ptr<float>(), Cij.data_ptr<float>(), M, N);
            //softmaxKernel2D_elementwise<<<blocks, threads>>>(Aij.data_ptr<float>(), d_sums, Cij.data_ptr<float>(),  M, N);
        }
    }

}
/*

void run_softmax_thread_coarse(torch::Tensor A, torch::Tensor C){
    const int seq_len = A.size(2);
    const int head_embd = A.size(3);
    const int M = seq_len;
    const int N = head_embd;
    int threadsPerBlock = 256;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(ceil((M + blockDim.x - 1) / (float)blockDim.x), ceil((N + blockDim.y - 1) / (float)blockDim.y));

    // loop over batchsize and head
    for (int i = 0; i < A.size(0); i++) {
        for (int j = 0; j < A.size(1); j++) {
            // get the i-th batch and j-th head
            torch::Tensor Aij = A[i][j];
            torch::Tensor Cij = C[i][j];

            // compute the softmax

            softmax_kernel_coalesced_coarsened(Aij.data_ptr<float>(), Cij.data_ptr<float>(), M, N, 4);

            //softmaxKernel2D_rows<<<blocks, threads>>>(Aij.data_ptr<float>(), Cij.data_ptr<float>(), M, N);
            //softmaxKernel2D_elementwise<<<blocks, threads>>>(Aij.data_ptr<float>(), d_sums, Cij.data_ptr<float>(),  M, N);
        }
    }

}

void run_softmax_optimized(torch::Tensor A, torch::Tensor C){
    const int seq_len = A.size(2);
    const int head_embd = A.size(3);
    const int M = seq_len;
    const int N = head_embd;
    float *d_sums;
    cudaMalloc(&d_sums, M * sizeof(float));
    dim3 threads(TILE_DIM_X, TILE_DIM_Y);
    dim3 blocks((M + TILE_DIM_X - 1) / TILE_DIM_X, (N + TILE_DIM_Y - 1) / TILE_DIM_Y);

    // loop over batchsize and head
    for (int i = 0; i < A.size(0); i++) {
        for (int j = 0; j < A.size(1); j++) {
            // get the i-th batch and j-th head
            torch::Tensor Aij = A[i][j];
            torch::Tensor Cij = C[i][j];

            // compute the softmax
            softmaxKernel2D_rows<<<blocks, threads>>>(Aij.data_ptr<float>(), d_sums, M, N);
            softmaxKernel2D_elementwise<<<blocks, threads>>>(Aij.data_ptr<float>(), d_sums, Cij.data_ptr<float>(),  M, N);
        }
    }
    cudaFree(d_sums);
}
*/
/*************************************************************************** Invocations*********************************************************/

torch::Tensor forward(torch::Tensor A) {
    const int batch_size = A.size(0);
    const int n_head = A.size(1);
    const int seq_len = A.size(2);
    const int head_embd = A.size(3);
    const int M = seq_len;
    const int N = head_embd;
    double start, end;
    torch::Tensor C = torch::zeros({batch_size, n_head, M, N}, A.options().device(torch::kCUDA));
    
    run_softmax_cuDNN(A,C);

    //softmax_kernel_naive<<<gridDim, blockDim>>>(A_data, C_data, M, N, softmax_scale);
    cudaDeviceSynchronize();
    //cudaFree(d_sums);
    return C;
}


