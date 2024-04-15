#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <chrono>
/*
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
     // Create CUDA events for timing GPU execution
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    int num_blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Allocate memory on the device
    float *input_device, *output_device;
    cudaMalloc((void**)&input_device, size);
    cudaMalloc((void**)&output_device, size);

    // Record start event for GPU
    cudaEventRecord(start_gpu);
    // Copy input data from host to device
    cudaMemcpy(input_device, input_host, size, cudaMemcpyHostToDevice);

   


    // Launch GPU kernel
    softmax_kernel_coalesced_coarsened<<<num_blocks, THREADS_PER_BLOCK>>>(input_device, output_device, rows, cols, coarsening_factor);
    cudaDeviceSynchronize();

    // Record stop event for GPU
    cudaEventRecord(stop_gpu);
    //cudaEventSynchronize(stop_gpu);


    // Copy output data from device to host
    cudaMemcpy(output_host_gpu, output_device, size, cudaMemcpyDeviceToHost);

    // Calculate GPU execution time
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

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
*/

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>

#define THREADS_PER_BLOCK 256

__global__ void softmax_kernel_coalesced_coarsened_reduction(float *input, float *output, int rows, int cols, int coarsening_factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows) {
        // Shared memory for partial maximum values
        __shared__ float sdata[THREADS_PER_BLOCK];

        // Initialize shared memory to negative infinity
        float max_val = -INFINITY;

        // Find the maximum value within the row using parallel reduction
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float val = input[idx * cols + i];
            max_val = fmaxf(max_val, val);
        }
        sdata[threadIdx.x] = max_val;

        // Reduce within the block
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
            }
            __syncthreads();
        }

        // Extract the maximum value from shared memory
        max_val = sdata[0];

        // Compute softmax values with coarsening
        float sum_exp = 0.0f;
        for (int i = 0; i < cols; i += coarsening_factor) {
            float exp_sum = 0.0f;
            for (int j = 0; j < coarsening_factor && i + j < cols; j++) {
                float exp_val = expf(input[idx * cols + i + j] - max_val);
                output[idx * cols + i + j] = exp_val;
                exp_sum += exp_val;
            }
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

__global__ void softmax_kernel_coalesced_coarsened_reduction2(float *input, float *output, int rows, int cols, int coarsening_factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows) {
        // Shared memory for partial maximum values
        __shared__ float sdata_max[THREADS_PER_BLOCK];
        // Shared memory for partial sum of exponentials
        __shared__ float sdata_sum[THREADS_PER_BLOCK];

        // Initialize shared memory to negative infinity for max_val
        float max_val = -INFINITY;
        // Initialize shared memory to zero for sum_exp
        float sum_exp = 0.0f;

        // Find the maximum value and compute the sum of exponentials within the row using parallel reduction
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float val = input[idx * cols + i];
            float exp_val = expf(val);
            max_val = fmaxf(max_val, val);
            sum_exp += exp_val;
        }
        sdata_max[threadIdx.x] = max_val;
        sdata_sum[threadIdx.x] = sum_exp;

        // Reduce maximum value within the block
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata_max[threadIdx.x] = fmaxf(sdata_max[threadIdx.x], sdata_max[threadIdx.x + s]);
                sdata_sum[threadIdx.x] += sdata_sum[threadIdx.x + s];
            }
            __syncthreads();
        }

        // Extract the maximum value from shared memory
        max_val = sdata_max[0];
        sum_exp = sdata_sum[0];

        // Compute softmax values with coarsening and normalize
        for (int i = 0; i < cols; i += coarsening_factor) {
            float exp_sum = 0.0f;
            for (int j = 0; j < coarsening_factor && i + j < cols; j++) {
                float exp_val = expf(input[idx * cols + i + j] - max_val);
                output[idx * cols + i + j] = exp_val / sum_exp;
                exp_sum += exp_val;
            }
            sum_exp = exp_sum;
        }
    }
}


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
    const int rows = 8192;
    const int cols = 8192;

    // Allocate memory on host
    float *input_host = (float*)malloc(rows * cols * sizeof(float));
    float *output_host_cpu = (float*)malloc(rows * cols * sizeof(float));
    float *output_host_gpu = (float*)malloc(rows * cols * sizeof(float));
    srand(1234); // Choose any seed value

    // Initialize input data
    for (int i = 0; i < rows * cols; i++) {
        input_host[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Define grid and block size
    int blockSize = 256;
    int numBlocks = (rows + blockSize - 1) / blockSize;

    // Allocate memory on device
    float *input_device, *output_device;
    cudaMalloc(&input_device, rows * cols * sizeof(float));
    cudaMalloc(&output_device, rows * cols * sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start_cpu_to_gpu, stop_cpu_to_gpu;
    cudaEventCreate(&start_cpu_to_gpu);
    cudaEventCreate(&stop_cpu_to_gpu);
    cudaEvent_t start_gpu_to_cpu, stop_gpu_to_cpu;
    cudaEventCreate(&start_gpu_to_cpu);
    cudaEventCreate(&stop_gpu_to_cpu);

    // Record start time for CPU to GPU memory transfer
    cudaEventRecord(start_cpu_to_gpu);
    // Transfer input data from host to device
    cudaMemcpy(input_device, input_host, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    // Record end time for CPU to GPU memory transfer
    cudaEventRecord(stop_cpu_to_gpu);
    cudaEventSynchronize(stop_cpu_to_gpu);

    // Launch kernel
    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventRecord(start_kernel);
    softmax_kernel_coalesced_coarsened<<<numBlocks, blockSize>>>(input_device, output_device, rows, cols, 8);
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    // Record start time for GPU to CPU memory transfer
    cudaEventRecord(start_gpu_to_cpu);
    // Transfer output data from device to host
    cudaMemcpy(output_host_gpu, output_device, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    // Record end time for GPU to CPU memory transfer
    cudaEventRecord(stop_gpu_to_cpu);
    cudaEventSynchronize(stop_gpu_to_cpu);

    // Calculate and print timings for CPU to GPU memory transfer
    float duration_cpu_to_gpu;
    cudaEventElapsedTime(&duration_cpu_to_gpu, start_cpu_to_gpu, stop_cpu_to_gpu);
    std::cout << "CPU to GPU Memory Transfer Time: " << duration_cpu_to_gpu << " ms" << std::endl;

    // Calculate and print timings for GPU kernel execution
    float duration_gpu_execution;
    cudaEventElapsedTime(&duration_gpu_execution, start_kernel, stop_kernel);
    std::cout << "GPU Kernel Execution Time: " << duration_gpu_execution << " ms" << std::endl;

    // Calculate and print timings for GPU to CPU memory transfer
    float duration_gpu_to_cpu;
    cudaEventElapsedTime(&duration_gpu_to_cpu, start_gpu_to_cpu, stop_gpu_to_cpu);
    std::cout << "GPU to CPU Memory Transfer Time: " << duration_gpu_to_cpu << " ms" << std::endl;

    // Total GPU execution time
    float total_gpu_time = duration_cpu_to_gpu + duration_gpu_execution + duration_gpu_to_cpu;
    std::cout << "Total GPU Execution Time: " << total_gpu_time << " ms" << std::endl;

    // Perform softmax on CPU for verification
    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_cpu(input_host, output_host_cpu, rows, cols);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    std::cout << "CPU Execution Time: " << duration_cpu.count() << " ms" << std::endl;

    // Compare CPU and GPU results
    bool passed = true;
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(output_host_cpu[i] - output_host_gpu[i]) > 1e-5) {
            std::cout << "CPU and GPU results mismatch at index " << i << ": "
                      << output_host_cpu[i] << " != " << output_host_gpu[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "CPU and GPU results match." << std::endl;
    } else {
        std::cout << "CPU and GPU results mismatch." << std::endl;
    }

    // Free memory
    free(input_host);
    free(output_host_cpu);
    free(output_host_gpu);
    cudaFree(input_device);
    cudaFree(output_device);

    // Destroy events
    cudaEventDestroy(start_cpu_to_gpu);
    cudaEventDestroy(stop_cpu_to_gpu);
    cudaEventDestroy(start_gpu_to_cpu);
    cudaEventDestroy(stop_gpu_to_cpu);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    return 0;
}
