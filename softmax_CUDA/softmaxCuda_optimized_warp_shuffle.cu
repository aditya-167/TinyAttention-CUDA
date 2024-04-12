#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cuda.h>
#include <chrono>


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

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_rows> <num_cols>" << std::endl;
        return 1;
    }
    const int N = std::stoi(argv[1]);
    const int M = std::stoi(argv[2]);

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

    // Create CUDA events for timing GPU execution
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    // Record start event for GPU
    CUDA_CHECK(cudaEventRecord(start_gpu));

    // Launch softmax kernel for rows on GPU
    softmaxKernel2D_rows<<<gridDim, blockDim>>>(d_input, d_exp_sums, N, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy exp_sums back to host
    std::vector<float> exp_sums(N);
    CUDA_CHECK(cudaMemcpy(exp_sums.data(), d_exp_sums, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Record stop event for GPU
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    // Calculate GPU execution time
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu));

    // Create chrono objects for timing CPU execution
    auto start_cpu = std::chrono::high_resolution_clock::now();

    // Compute softmax rows CPU for verification
    softmax2D_rows_cpu<float>(input.data(), exp_sums.data(), N, M);
    cudaDeviceSynchronize();

    // Compute softmax elementwise CPU for verification
    //softmax2D_elementwise_cpu<float>(input.data(), exp_sums.data(), output_cpu.data(), N, M);
    cudaDeviceSynchronize();

    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);

    // Print CPU execution time
    std::cout << "CPU Execution Time: " << duration_cpu.count() / 1000 << " ms" << std::endl;

    // Create CUDA events for timing GPU execution
    cudaEvent_t start_gpu_elementwise, stop_gpu_elementwise;
    CUDA_CHECK(cudaEventCreate(&start_gpu_elementwise));
    CUDA_CHECK(cudaEventCreate(&stop_gpu_elementwise));

    // Record start event for GPU elementwise computation
    CUDA_CHECK(cudaEventRecord(start_gpu_elementwise));

    // Launch softmax kernel elementwise on GPU
    //softmaxKernel2D_elementwise<<<gridDim, blockDim>>>(d_input, d_exp_sums, d_output, N, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Record stop event for GPU elementwise computation
    CUDA_CHECK(cudaEventRecord(stop_gpu_elementwise));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu_elementwise));

    // Calculate GPU elementwise execution time
    float gpu_elementwise_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_elementwise_time, start_gpu_elementwise, stop_gpu_elementwise));

    // Copy output from device to host
    CUDA_CHECK(cudaMemcpy(output_gpu.data(), d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));

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

    // Print GPU execution time
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;
    //std::cout << "GPU Elementwise Execution Time: " << gpu_elementwise_time << " ms" << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_exp_sums));
    CUDA_CHECK(cudaFree(d_output));

    // Destroy events
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));
    CUDA_CHECK(cudaEventDestroy(start_gpu_elementwise));
    CUDA_CHECK(cudaEventDestroy(stop_gpu_elementwise));

    return 0;
}
