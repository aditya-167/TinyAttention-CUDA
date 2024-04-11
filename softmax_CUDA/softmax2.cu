#include <torch/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


#include <iostream>
#include <math.h>


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
__global__ void softmax_parallel_reduction(float *d_in, float *d_out, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;

        // Subtract each element from the maximum within its row to prevent overflow
        float max_val = d_in[row * cols];
        for (int i = 1; i < cols; ++i) {
            max_val = fmaxf(max_val, d_in[row * cols + i]);
        }
        float shifted_val = d_in[idx] - max_val;

        // Compute exponentials
        float exp_val = expf(shifted_val);

        // Perform parallel reduction to compute sum
        __shared__ float sum;
        if (col == 0) sum = 0;
        __syncthreads();

        atomicAdd(&sum, exp_val);
        __syncthreads();

        // Compute softmax for each element
        d_out[idx] = exp_val / sum;
    }
}
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


// normal
/*
torch::Tensor forward(torch::Tensor A) {
    const int batch_size = A.size(0);
    const int n_head = A.size(1);
    const int seq_len = A.size(2);
    const int head_embd = A.size(3);
    const int M = seq_len;
    const int N = head_embd;

    torch::Tensor C = torch::zeros({batch_size, n_head, M, N}, A.options().device(torch::kCUDA));
    auto A_data = A.data_ptr<float>();
    auto C_data = C.data_ptr<float>();
    int threadsPerBlock = 256;
    dim3 blockDim(threadsPerBlock);
    
    dim3 blocksPerGrid ((M + threadsPerBlock - 1) / threadsPerBlock);
    //softmax_kernel_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, M, N);

    //dim3 blockDim(16,16);
    
    //dim3 gridDim(ceil((M + blockDim.x - 1) / (float)blockDim.x), ceil((N + blockDim.y - 1) / (float)blockDim.y));

    // loop over batchsize and head
    for (int i = 0; i < A.size(0); i++) {
        for (int j = 0; j < A.size(1); j++) {
            // get the i-th batch and j-th head
            torch::Tensor Aij = A[i][j];
            torch::Tensor Cij = C[i][j];
            // compute the softmax
            //int shared_memory_size = blockDim.x * sizeof(float);
            softmax_cudnn(Aij.data_ptr<float>(), Cij.data_ptr<float>(), M, N);

            //softmax_kernel_naive<<<gridDim, blockDim>>>(Aij.data_ptr<float>(), Cij.data_ptr<float>(), M, N);

            //sgemm_naive<<<gridDim, blockDim>>>(A.size(2), B.size(3), A.size(3), Aij.data_ptr<float>(), Bij.data_ptr<float>(), Cij.data_ptr<float>());
            // allocate memory for output on GPU in cuda
        }
    }

    //softmax_kernel_naive<<<gridDim, blockDim>>>(A_data, C_data, M, N, softmax_scale);
    cudaDeviceSynchronize();

    return C;
}
*/

/*
torch::Tensor forward(torch::Tensor A) {
    const int batch_size = A.size(0);
    const int n_head = A.size(1);
    const int seq_len = A.size(2);
    const int head_embd = A.size(3);
    const int M = seq_len;
    const int N = head_embd;

    torch::Tensor C = torch::zeros({batch_size, n_head, M, N}, A.options().device(torch::kCUDA));
    auto A_data = A.data_ptr<float>();
    auto C_data = C.data_ptr<float>();
    int threadsPerBlock = 256;
    dim3 blockDim(threadsPerBlock);

    dim3 blocksPerGrid ((batch_size * n_head + threadsPerBlock - 1) / threadsPerBlock);

    batch_softmax_kernel_naive2<<<blocksPerGrid, blockDim>>>(A_data, C_data, batch_size, n_head, seq_len, head_embd);

    cudaDeviceSynchronize();

    return C;
}
*/