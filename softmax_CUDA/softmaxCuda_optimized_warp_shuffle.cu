#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cuda.h>
#include <chrono>


#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>

#define TILE_DIM_X 32  // Tile dimension for rows
#define TILE_DIM_Y 32  // Tile dimension for columns

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
        if (fabs(softmax_val - 1) > 1e-5) {
            std::cout << softmax_val << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    int N = 8192;
    int M = 8192;
    std::vector<float> h_input(N * M);
    std::vector<float> h_output(N * M);
    float *d_input, *d_output, *d_exp_sums;

    // Randomly initialize input
    randomInit(h_input);

    // Allocate memory on the device
    cudaMalloc(&d_input, N * M * sizeof(float));
    cudaMalloc(&d_output, N * M * sizeof(float));
    cudaMalloc(&d_exp_sums, N * sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timing for memory transfer from CPU to GPU
    cudaEventRecord(start);
    cudaMemcpy(d_input,  h_input.data(),  N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
        cudaEventSynchronize(stop);

    float cpu_to_gpu_time;
    cudaEventElapsedTime(&cpu_to_gpu_time, start, stop);
    std::cout << "CPU to GPU Memory Transfer Time: " << cpu_to_gpu_time << " ms" << std::endl;

    // Define block and grid dimensions
    dim3 threads(TILE_DIM_X, TILE_DIM_Y);
    dim3 blocks((M + TILE_DIM_X - 1) / TILE_DIM_X, (N + TILE_DIM_Y - 1) / TILE_DIM_Y);

    // Kernel execution time measurement
    cudaEventRecord(start);
    softmaxKernel2D_rows<<<blocks, threads>>>(d_input, d_exp_sums, N, M);
        cudaDeviceSynchronize();

    softmaxKernel2D_elementwise<<<blocks, threads>>>(d_input, d_exp_sums, d_output, N, M);

cudaEventRecord(stop);
cudaDeviceSynchronize();
        cudaEventSynchronize(stop);

            float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);
    std::cout << "GPU Kernel Execution Time: " << kernel_time << " ms" << std::endl;

    // Timing for memory transfer from GPU to CPU
    cudaEventRecord(start);

    // Copy the result back to the host
    cudaMemcpy(h_output.data(), d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    float gpu_to_cpu_time;
    cudaEventElapsedTime(&gpu_to_cpu_time, start, stop);
    std::cout << "GPU to CPU Memory Transfer Time: " << gpu_to_cpu_time << " ms" << std::endl;


    // Check correctness and print result
    if (verifyOutput(h_output.data(), N, M)) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failure!" << std::endl;
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_exp_sums);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
