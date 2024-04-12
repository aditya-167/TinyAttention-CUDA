
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cuda.h>

/*
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
    int N = 3000; // number of rows in dataset
    int M = 2000; // number of columns in dataset
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

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <vector>

// Define TILE_DIM_X and TILE_DIM_Y if they're not already defined
#ifndef TILE_DIM_X
#define TILE_DIM_X 32
#endif

#ifndef TILE_DIM_Y
#define TILE_DIM_Y 32
#endif

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
template <typename T>
void softmax2D_rows_cpu(const T* input, T* exp_sums, int N, int M) {
    for (int row = 0; row < N; ++row) {
        T max_val = input[row * M];  // Initialize max_val with the first element of the row
        for (int i = 1; i < M; ++i) {
            max_val = std::max(max_val, input[row * M + i]);
        }
        T sum = 0;
        for (int col = 0; col < M; ++col) {
            T val = exp(input[row * M + col] - max_val);
            exp_sums[row] += val;
            sum += val;
        }
        exp_sums[row] = sum;
    }
}

template <typename T>
void softmax2D_elementwise_cpu(const T* input, const T* exp_sums, T* output, int N, int M) {
    for (int row = 0; row < N; ++row) {
        T max_val = input[row * M];  // Initialize max_val with the first element of the row
        for (int i = 1; i < M; ++i) {
            max_val = std::max(max_val, input[row * M + i]);
        }
        T exp_sum_row = exp_sums[row];
        for (int col = 0; col < M; ++col) {
            output[row * M + col] = exp(input[row * M + col] - max_val) / exp_sum_row;
        }
    }
}
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {
    // Define input dimensions
    const int N = 3000;
    const int M = 2414;

    // Generate random input
    std::vector<float> input(N * M);
    for (int i = 0; i < N * M; ++i) {
        input[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f;
    }

    // Allocate memory for output on CPU
    std::vector<float> output_cpu(N * M);
    std::vector<float> output_gpu(N * M);

    // Allocate device memory
    float *d_input, *d_exp_sums, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_exp_sums, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * M * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), N * M * sizeof(float), cudaMemcpyHostToDevice));

    // Set grid and block dimensions
    dim3 blockDim(TILE_DIM_X, TILE_DIM_Y);
    dim3 gridDim((M + TILE_DIM_X - 1) / TILE_DIM_X, (N + TILE_DIM_Y - 1) / TILE_DIM_Y);

    // Launch softmax kernel for rows on GPU
    softmaxKernel2D_rows<<<gridDim, blockDim>>>(d_input, d_exp_sums, N, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy exp_sums back to host
    std::vector<float> exp_sums(N);
    CUDA_CHECK(cudaMemcpy(exp_sums.data(), d_exp_sums, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute softmax rows CPU for verification
    softmax2D_rows_cpu<float>(input.data(), exp_sums.data(), N, M);

    // Launch softmax kernel elementwise on GPU
    softmaxKernel2D_elementwise<<<gridDim, blockDim>>>(d_input, d_exp_sums, d_output, N, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output from device to host
    CUDA_CHECK(cudaMemcpy(output_gpu.data(), d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute softmax elementwise CPU for verification
    softmax2D_elementwise_cpu<float>(input.data(), exp_sums.data(), output_cpu.data(), N, M);

    // Verify results
    bool passed = true;
    for (int i = 0; i < N * M; ++i) {
        if (std::abs(output_cpu[i] - output_gpu[i]) > 1e-5) {
            std::cout << "Verification failed at index " << i << ": CPU = " << output_cpu[i] << ", GPU = " << output_gpu[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Verification passed!" << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_exp_sums));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}


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
