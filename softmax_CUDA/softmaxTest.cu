/*
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cuda.h>


const int TILE_DIM_Y = 32;  // Tile dimension for rows
const int TILE_DIM_X = 32;  // Tile dimension for columns// must be 32 for this method

template <typename T>
__global__ void softmaxKernel2D_rows(const T* input, T* exp_sums, int N, int M) {
    int row = blockIdx.y * TILE_DIM_Y + threadIdx.y;
    int col = blockIdx.x * TILE_DIM_X + threadIdx.x;
    T val = 0;
    // Copy data from global memory to shared memory
    if (row < N && col < M) {
        if (sizeof(T) == 8)
          val = exp(input[row * M + col]);
        else
          val = expf(input[row * M + col]);
    }
    // warp shuffle reduction
    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i>>=1)
        val += __shfl_xor_sync(0xffffffff, val, i, 32);
    // update global value for row
    if ((threadIdx.x == 0) && (row < N)) atomicAdd(exp_sums+row, val);
}

template <typename T>
__global__ void softmaxKernel2D_elementwise(const T* input, const T* exp_sums, T *output, int N, int M) {
    int row = blockIdx.y * TILE_DIM_Y + threadIdx.y;
    int col = blockIdx.x * TILE_DIM_X + threadIdx.x;
    // Compute the softmax values
    if (row < N && col < M) {
      if (sizeof(T) == 8)
        output[row * M + col] = exp(input[row * M + col])/ exp_sums[row];
      else
        output[row * M + col] = expf(input[row * M + col])/ exp_sums[row];
    }
}

template <typename T>
void randomInit(std::vector<T> &x) {
    // Pseudo-random float vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> unif(-1, 1);
    for (int i = 0; i < x.size(); i++) {
        x[i] = unif(gen);
    }
}
template <typename T>
bool verifyOutput(const T* output, int N, int M) {
    for (int i = 0; i < N; i++) {
        T softmax_val = 0;
        for (int j = 0; j < M; j++) {
            softmax_val += output[i * M + j];
        }
        // Check if row sums to one
        if (fabs(softmax_val - 1) > 1e-4) {
            std::cout << softmax_val << std::endl;
            return false;
        }
    }
    return true;
}

using mt = float;

int main() {
    int N = 2048; // number of rows in dataset
    int M = 512; // number of columns in dataset
    std::vector<mt> h_input(N * M);
    std::vector<mt> h_output(N * M);
    mt *d_input, *d_output, *d_sums;

    // Randomly initialize input
    randomInit(h_input);

    // Allocate memory on the device
    cudaMalloc(&d_input, N * M * sizeof(mt));
    cudaMalloc(&d_output, N * M * sizeof(mt));
    cudaMalloc(&d_sums, N * sizeof(mt));
    cudaMemset(d_sums, 0, N*sizeof(mt));
    // Copy to device
    cudaMemcpy(d_input,  h_input.data(),  N * M * sizeof(mt), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threads(TILE_DIM_X, TILE_DIM_Y);
    dim3 blocks((M + TILE_DIM_X - 1) / TILE_DIM_X, (N + TILE_DIM_Y - 1) / TILE_DIM_Y);

    // Launch the kernel
    softmaxKernel2D_rows<<<blocks, threads>>>(d_input, d_sums, N, M);
    softmaxKernel2D_elementwise<<<blocks, threads>>>(d_input, d_sums, d_output,  N, M);

    // Copy to host
    cudaMemcpy(h_output.data(), d_output, N * M * sizeof(mt), cudaMemcpyDeviceToHost);

    // Check
    if (verifyOutput(h_output.data(), N, M)) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failure!" << std::endl;
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
*/

/*
#include <stdio.h>
#include <math.h>

__global__ void safe_softmax(float *input, float *output, int rows, int cols) {
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

int main() {
    int rows = 100;
    int cols = 5100;
    float *h_input, *h_output_cpu, *h_output_gpu;
    float *d_input, *d_output;

    // Allocate host memory
    h_input = (float *)malloc(rows * cols * sizeof(float));
    h_output_cpu = (float *)malloc(rows * cols * sizeof(float));
    h_output_gpu = (float *)malloc(rows * cols * sizeof(float));

    // Allocate device memory
    cudaMalloc((void **)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void **)&d_output, rows * cols * sizeof(float));

    // Initialize input matrix
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = i % 7; // Some arbitrary values
    }

    // Copy input matrix to device
    cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch GPU kernel
    int block_size = 256;
    int num_blocks = (rows + block_size - 1) / block_size;
    safe_softmax<<<num_blocks, block_size>>>(d_input, d_output, rows, cols);

    // Copy GPU output to host
    cudaMemcpy(h_output_gpu, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute CPU softmax
    for (int i = 0; i < rows; i++) {
        float max_val = h_input[i * cols];
        for (int j = 1; j < cols; j++) {
            if (h_input[i * cols + j] > max_val) {
                max_val = h_input[i * cols + j];
            }
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum_exp += expf(h_input[i * cols + j] - max_val);
        }
        for (int j = 0; j < cols; j++) {
            h_output_cpu[i * cols + j] = expf(h_input[i * cols + j] - max_val) / sum_exp;
        }
    }

    // Verify results
    float threshold = 1e-5;
    bool passed = true;
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > threshold) {
            passed = false;
            break;
        }
    }

    if (passed) {
        printf("Verification passed!\n");
        // Print values for verification
        printf("CPU Softmax:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%.6f ", h_output_cpu[i * cols + j]);
            }
            printf("\n");
        }
        printf("\n");
        printf("GPU Softmax:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%.6f ", h_output_gpu[i * cols + j]);
            }
            printf("\n");
        }
    } else {
        printf("Verification failed!\n");
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);

    return 0;
}*/



#include <stdio.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cmath>

// Softmax kernel using cuDNN
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

// CPU softmax implementation
void softmax_cpu(float *input, float *output, int num_samples, int num_classes) {
    for (int i = 0; i < num_samples; ++i) {
        float max_val = input[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            max_val = fmaxf(max_val, input[i * num_classes + j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            output[i * num_classes + j] = expf(input[i * num_classes + j] - max_val);
            sum += output[i * num_classes + j];
        }

        for (int j = 0; j < num_classes; ++j) {
            output[i * num_classes + j] /= sum;
        }
    }
}

int main() {
    // Example usage
    const int num_samples = 4000;
    const int num_classes = 3000;
    float *input, *output_gpu, *output_cpu;
    
    // Allocate memory on GPU
    cudaMalloc(&input, num_samples * num_classes * sizeof(float));
    cudaMalloc(&output_gpu, num_samples * num_classes * sizeof(float));
    
    // Initialize input data on CPU
    float *host_input = (float*)malloc(num_samples * num_classes * sizeof(float));
    // Initialize host_input...
    
    // Copy input data to GPU
    cudaMemcpy(input, host_input, num_samples * num_classes * sizeof(float), cudaMemcpyHostToDevice);
    
    // Call the cuDNN softmax function
    softmax_cudnn(input, output_gpu, num_samples, num_classes);
    
    // Allocate memory for CPU output
    output_cpu = (float*)malloc(num_samples * num_classes * sizeof(float));
    
    // Call CPU softmax function
    softmax_cpu(host_input, output_cpu, num_samples, num_classes);
    
    // Copy the result back to host (GPU)
    float *host_output_gpu = (float*)malloc(num_samples * num_classes * sizeof(float));
    cudaMemcpy(host_output_gpu, output_gpu, num_samples * num_classes * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare CPU and GPU results
    bool results_match = true;
    for (int i = 0; i < num_samples * num_classes; ++i) {
        if (fabs(output_cpu[i] - host_output_gpu[i]) > 1e-5) { // Tolerance for floating-point comparison
            results_match = false;
            break;
        }
    }
    
    if (results_match) {
        printf("CPU and GPU softmax results match.\n");
    } else {
        printf("CPU and GPU softmax results do not match.\n");
    }
    
    // Free memory
    cudaFree(input);
    cudaFree(output_gpu);
    free(host_input);
    free(output_cpu);
    free(host_output_gpu);
    
    return 0;
}
