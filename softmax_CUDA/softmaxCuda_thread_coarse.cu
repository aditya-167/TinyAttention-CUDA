#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <chrono>

#define THREADS_PER_BLOCK 256

// CUDA kernel for softmax calculation with thread coalescing and coarsening
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

// Host function to calculate softmax
void softmax_cpu(float *input, float *output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float max_val = input[i * cols];
        for (int j = 1; j < cols; j++) {
            float val = input[i * cols + j];
            max_val = (val > max_val) ? val : max_val;
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            float exp_val = expf(input[i * cols + j] - max_val);
            output[i * cols + j] = exp_val;
            sum_exp += exp_val;
        }

        for (int j = 0; j < cols; j++) {
            output[i * cols + j] /= sum_exp;
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

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_rows> <num_cols>" << std::endl;
        return 1;
    }
    const int rows = std::stoi(argv[1]);
    const int cols = std::stoi(argv[2]);
    const int size = rows * cols * sizeof(float);
    const int coarsening_factor = 8; // You can adjust this value as needed

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
    int num_blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    softmax_kernel_coalesced_coarsened<<<num_blocks, THREADS_PER_BLOCK>>>(input_device, output_device, rows, cols, coarsening_factor);
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
