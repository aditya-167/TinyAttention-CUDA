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



#include <iostream>
#include <math.h>


__global__ void softmax_kernel_naive(float *x, float *C, int M, int N, float softmax_scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        float max_val = x[i * N];
        for (int k = 1; k < N; ++k) {
            if (x[i * N + k] > max_val) {
                max_val = x[i * N + k];
            }
        }

        float sum_exp = 0.0f;
        for (int k = 0; k < N; ++k) {
            C[i * N + k + j * M * N] = expf((x[i * N + k] - max_val) * softmax_scale);
            sum_exp += C[i * N + k + j * M * N];
        }

        for (int k = 0; k < N; ++k) {
            C[i * N + k + j * M * N] /= sum_exp;
        }
    }
}

void softmax_cpu(float *x, float *C, int M, int N, float softmax_scale) {
    for (int i = 0; i < M; ++i) {
        float max_val = x[i * N];
        for (int j = 1; j < N; ++j) {
            if (x[i * N + j] > max_val) {
                max_val = x[i * N + j];
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = expf((x[i * N + j] - max_val) * softmax_scale); // Applying softmax_scale here
            sum_exp += C[i * N + j];
        }

        for (int j = 0; j < N; ++j) {
            C[i * N + j] /= sum_exp;
        }
    }
}
/*
int main() {
    // Example usage
    const int M = 40; // Number of examples
    const int K = 60; // Number of classes
    const float threshold = 1e-5; // Threshold for verification

    // Random input data
    float input_cpu[M * K];
    float input_gpu[M * K];
    float softmax_cpu_results[M * K];
    float softmax_gpu_results[M * K];
    for (int i = 0; i < M * K; ++i) {
        input_cpu[i] = input_gpu[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Softmax scale
    float softmax_scale = 1.0f / sqrt(K);

    // Launching kernel on GPU
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, M * K * sizeof(float));
    cudaMalloc((void **)&d_output, M * K * sizeof(float));
    cudaMemcpy(d_input, input_gpu, M * K * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (K + blockDim.y - 1) / blockDim.y);

    softmax_kernel_naive<<<gridDim, blockDim>>>(d_input, d_output, M, K, softmax_scale);
    cudaDeviceSynchronize();

    cudaMemcpy(softmax_gpu_results, d_output, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    // Performing softmax on CPU for comparison
    softmax_cpu(input_cpu, softmax_cpu_results, M, K, softmax_scale);

    // Verification
    bool passed = true;
    for (int i = 0; i < M * K; ++i) {
        if (fabs(softmax_cpu_results[i] - softmax_gpu_results[i]) > threshold) {
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Verification passed! GPU and CPU results are within the threshold." << std::endl;
        
        // Print CPU results
        std::cout << "CPU Results:" << std::endl;
        for (int i = 0; i < M * K; ++i) {
            std::cout << softmax_cpu_results[i] << " ";
        }
        std::cout << std::endl;

        // Print GPU results
        std::cout << "GPU Results:" << std::endl;
        for (int i = 0; i < M * K; ++i) {
            std::cout << softmax_gpu_results[i] << " ";
        }
        std::cout << std::endl;
        
    } else {
        std::cout << "Verification failed! GPU and CPU results differ by more than the threshold." << std::endl;
    }
    
    return 0;
}
*/
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
    //dim3 blockDim(threadsPerBlock);
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Softmax scale
    float softmax_scale = 1.0f / sqrt(N);

    softmax_kernel_naive<<<gridDim, blockDim>>>(A_data, C_data, M, N, softmax_scale);
    cudaDeviceSynchronize();

    return C;
}
