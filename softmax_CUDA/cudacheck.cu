#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// CUDA kernel for naive implementation of softmax with numerical stability and reduction for finding max value
__global__ void softmax_kernel_naive(float *input, float *output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        int col = idx % cols;
        
        __shared__ float max_vals[THREADS_PER_BLOCK]; // Shared memory for storing max values
        max_vals[threadIdx.x] = input[row * cols + col];
        __syncthreads();
        
        // Reduction to find max value in the row
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                max_vals[threadIdx.x] = fmaxf(max_vals[threadIdx.x], max_vals[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        
        // Get max value from shared memory
        float max_val = max_vals[0];
        
        // Subtract max value from each element for numerical stability
        float exp_val = expf(input[idx] - max_val);
        
        float sum_exp = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum_exp += expf(input[row * cols + i] - max_val);
        }
        
        output[idx] = exp_val / sum_exp;
    }
}

// Host function for CPU softmax calculation
void softmax_cpu(float *input, float *output, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        float max_val = input[row * cols];
        for (int col = 1; col < cols; col++) {
            max_val = fmaxf(max_val, input[row * cols + col]);
        }
        
        float sum_exp = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum_exp += expf(input[row * cols + col] - max_val);
        }
        
        for (int col = 0; col < cols; col++) {
            output[row * cols + col] = expf(input[row * cols + col] - max_val) / sum_exp;
        }
    }
}

// Function to verify GPU results against CPU results
bool verify_result(float *gpu_result, float *cpu_result, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(gpu_result[i] - cpu_result[i]) > 1e-5) {
            std::cout << "Verification failed at index " << i << ": GPU - " << gpu_result[i] << ", CPU - " << cpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int rows = 8192;
    const int cols = 8192;
    const int size = rows * cols * sizeof(float);
    
    // Allocate memory on the host
    float *input_host = new float[size];
    float *output_host_cpu = new float[size];
    float *output_host_gpu = new float[size];

    // Initialize input data on the host
    srand(time(NULL));
    for (int i = 0; i < rows * cols; i++) {
        input_host[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f;
    }

    // Allocate memory on the device
    float *input_device, *output_device;
    cudaMalloc((void**)&input_device, size);
    cudaMalloc((void**)&output_device, size);

    // Copy input data from host to device
    cudaMemcpy(input_device, input_host, size, cudaMemcpyHostToDevice);

    // Create CUDA events for timing GPU execution
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Record start event for GPU
    cudaEventRecord(start_gpu);

    // Launch GPU kernel
    int num_blocks = (rows * cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    softmax_kernel_naive<<<num_blocks, THREADS_PER_BLOCK>>>(input_device, output_device, rows, cols);
    cudaDeviceSynchronize();

    // Record stop event for GPU
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    // Calculate GPU execution time
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    // Copy output data from device to host
    cudaMemcpy(output_host_gpu, output_device, size, cudaMemcpyDeviceToHost);

    // Perform softmax calculation on the CPU and measure time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_cpu(input_host, output_host_cpu, rows, cols);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);

    // Verify GPU results against CPU results
    bool result = verify_result(output_host_gpu, output_host_cpu, rows * cols);
    if (result) {
        std::cout << "GPU computation matches CPU computation." << std::endl;
    } else {
        std::cout << "GPU computation does not match CPU computation." << std::endl;
    }

    // Print GPU execution time
    std::cout << "GPU Execution Time: " << gpu_time << " milliseconds" << std::endl;

    // Print CPU execution time
    std::cout << "CPU Execution Time: " << duration_cpu.count() << " milliseconds" << std::endl;

    // Free memory
    delete[] input_host;
    delete[] output_host_cpu;
    delete[] output_host_gpu;
    cudaFree(input_device);
    cudaFree(output_device);

    // Destroy events
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
