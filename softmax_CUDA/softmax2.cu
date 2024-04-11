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


__global__ void softmax_kernel_naive(float *x, float *C, int M, int N) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        float max_val = x[i * N];
        for (int k = 1; k < N; ++k) {
            if (x[i * N + k] > max_val) {
                max_val = x[i * N + k];
            }
        }

        float sum_exp = 0.0f;
        for (int k = 0; k < N; ++k) {
            C[i * N + k + j * M * N] = expf((x[i * N + k] - max_val));
            sum_exp += C[i * N + k + j * M * N];
        }

        for (int k = 0; k < N; ++k) {
            C[i * N + k + j * M * N] /= sum_exp;
        }
    }
}
__global__ void softmax_kernel_naive_batched(float *x, float *C, int M, int N, int B) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint batch = blockIdx.z * blockDim.z + threadIdx.z;
    const uint batch_offset = batch*M*N;
    if (batch<B){
        if (i < M && j < N) {
            float max_val = x[i * N];
            for (int k = 1; k < N; ++k) {
                if (x[batch_offset + i * N + k] > max_val) {
                    max_val = x[batch_offset+i * N + k];
                }
            }

            float sum_exp = 0.0f;
            for (int k = 0; k < N; ++k) {
                C[i * N + k + j * M * N] = expf((x[i * N + k] - max_val));
                sum_exp += C[i * N + k + j * M * N];
            }

            for (int k = 0; k < N; ++k) {
                C[i * N + k + j * M * N] /= sum_exp;
            }
        }

    }

    
}
/*
void softmax_cpu(float *x, float *C, int M, int N) {
    for (int i = 0; i < M; ++i) {
        float max_val = x[i * N];
        for (int j = 1; j < N; ++j) {
            if (x[i * N + j] > max_val) {
                max_val = x[i * N + j];
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = expf((x[i * N + j] - max_val)); // Applying softmax_scale here
            sum_exp += C[i * N + j];
        }

        for (int j = 0; j < N; ++j) {
            C[i * N + j] /= sum_exp;
        }
    }
}
*/
/*
void run_sgemm_naive_batched(torch::Tensor A, torch::Tensor B, torch::Tensor C){
    
    dim3 gridDim(CEIL_DIV(B.size(3), 32), CEIL_DIV(A.size(2), 32), CEIL_DIV(A.size(0)*A.size(1), BD));
    dim3 blockDim(32,32,BD);
    int L = A.size(0)*A.size(1);
    sgemm_naive_batched<<<gridDim, blockDim>>>(L, A.size(2), B.size(3), A.size(3), A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
    return;

}
*/

/*

// Softmax CUDA kernel using cuDNN
__global__ void softmax_kernel_cuDNN(const float *input, float *output, int M, int N, float softmax_scale) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x < M && idx_y < N) {
        int idx = idx_y * M + idx_x;

        // Define cuDNN descriptors
        cudnnHandle_t cudnn;
        cudnnCreate(&cudnn);
        cudnnTensorDescriptor_t desc;
        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, M, N);

        // Softmax operation
        float alpha = 1.0f;
        float beta = 0.0f;
        cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, desc, input + idx, &beta, desc, output + idx);

        // Scale softmax output
        output[idx] *= softmax_scale;

        // Destroy cuDNN descriptors
        cudnnDestroyTensorDescriptor(desc);
        cudnnDestroy(cudnn);
    }
}
*/
/*
int main() {
    // Example usage
    const int M = 200; // Number of examples
    const int K = 100; // Number of classes
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

    //int threadsPerBlock = 256;
    dim3 blockDim(32,32);
    dim3 gridDim(ceil((M + blockDim.x - 1) / (float)blockDim.x), ceil((K + blockDim.y - 1) / (float)blockDim.y));

    //dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (K + blockDim.y - 1) / blockDim.y);

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
    //int threadsPerBlock = 256;
    //dim3 blockDim(threadsPerBlock);
    dim3 blockDim(32,32);
    
    dim3 gridDim(ceil((M + blockDim.x - 1) / (float)blockDim.x), ceil((N + blockDim.y - 1) / (float)blockDim.y));

    // loop over batchsize and head
    for (int i = 0; i < A.size(0); i++) {
        for (int j = 0; j < A.size(1); j++) {
            // get the i-th batch and j-th head
            torch::Tensor Aij = A[i][j];
            torch::Tensor Cij = C[i][j];
            // compute the softmax
            softmax_kernel_naive<<<gridDim, blockDim>>>(Aij.data_ptr<float>(), Cij.data_ptr<float>(), M, N);

            //sgemm_naive<<<gridDim, blockDim>>>(A.size(2), B.size(3), A.size(3), Aij.data_ptr<float>(), Bij.data_ptr<float>(), Cij.data_ptr<float>());
            // allocate memory for output on GPU in cuda
        }
    }

    //softmax_kernel_naive<<<gridDim, blockDim>>>(A_data, C_data, M, N, softmax_scale);
    cudaDeviceSynchronize();

    return C;
}
