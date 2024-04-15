/*
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
// Kernel declaration
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

// CPU implementation of softmax
void softmax_cpu(float *input, float *output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float max_val = input[i * cols];
        for (int j = 1; j < cols; j++) {
            if (input[i * cols + j] > max_val) {
                max_val = input[i * cols + j];
            }
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum_exp += expf(input[i * cols + j] - max_val);
        }
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = expf(input[i * cols + j] - max_val) / sum_exp;
        }
    }
}

int main(int argc, char *argv[]) {
    // Parse command-line argume
    const int rows = 4096;
    const int cols = 4096;

    // Allocate memory on host
    float *input_host = (float*)malloc(rows * cols * sizeof(float));
    float *output_host_cpu = (float*)malloc(rows * cols * sizeof(float));
    float *output_host_gpu = (float*)malloc(rows * cols * sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
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

    // Record start event for GPU
    cudaEventRecord(start);
    // Transfer input data from host to device
    cudaMemcpy(input_device, input_host, rows * cols * sizeof(float), cudaMemcpyHostToDevice);





    // Launch kernel
    softmax_kernel_naive<<<numBlocks, blockSize>>>(input_device, output_device, rows, cols);
    cudaDeviceSynchronize();



    // Transfer output data from device to host
    cudaMemcpy(output_host_gpu, output_device, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    // Record stop event for GPU
    cudaEventRecord(stop);
    // Calculate GPU execution time
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Record start time for CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    //cudaEventRecord(start);

    // Perform softmax on CPU for verification
    softmax_cpu(input_host, output_host_cpu, rows, cols);

    // Record stop time for CPU
    //cudaEventRecord(stop);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    // Synchronize event recording
    //cudaEventSynchronize(stop);

    // Calculate CPU execution time
    //float cpu_time;
    //cudaEventElapsedTime(&cpu_time, start, stop);

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

    std::cout << "GPU Execution Time: " << gpu_time << " ms" << std::endl;
    std::cout << "CPU Execution Time: " << duration_cpu.count()  << " ms" << std::endl;

    // Free memory
    free(input_host);
    free(output_host_cpu);
    free(output_host_gpu);
    cudaFree(input_device);
    cudaFree(output_device);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
*/

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>


// Kernel declaration
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

__global__ void softmax_kernel_reduction(float *input, float *output, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows) {
        // Shared memory for partial maximum values
        __shared__ float sdata[256];

        // Find the maximum value within the row using parallel reduction
        float max_val = -INFINITY;
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

        // Compute softmax values
        float sum_exp = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum_exp += expf(input[idx * cols + i] - max_val);
        }
        for (int i = 0; i < cols; i++) {
            output[idx * cols + i] = expf(input[idx * cols + i] - max_val) / sum_exp;
        }
    }
}
__global__ void softmax_kernel_reduction2(float *input, float *output, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows) {
        // Shared memory for partial maximum values
        __shared__ float sdata_max[256];
        // Shared memory for partial sum of exponentials
        __shared__ float sdata_sum[256];

        // Find the maximum value within the row using parallel reduction
        float max_val = -INFINITY;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float val = input[idx * cols + i];
            max_val = fmaxf(max_val, val);
        }
        sdata_max[threadIdx.x] = max_val;

        // Reduce maximum value within the block
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata_max[threadIdx.x] = fmaxf(sdata_max[threadIdx.x], sdata_max[threadIdx.x + s]);
            }
            __syncthreads();
        }

        // Extract the maximum value from shared memory
        max_val = sdata_max[0];

        // Compute sum of exponentials and store it in shared memory
        float sum_exp = 0.0f;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float exp_val = expf(input[idx * cols + i] - max_val);
            sum_exp += exp_val;
        }
        sdata_sum[threadIdx.x] = sum_exp;

        // Reduce sum of exponentials within the block
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata_sum[threadIdx.x] += sdata_sum[threadIdx.x + s];
            }
            __syncthreads();
        }

        // Normalize the softmax values
        for (int i = 0; i < cols; i++) {
            output[idx * cols + i] = expf(input[idx * cols + i] - max_val) / sdata_sum[0];
        }
    }
}


// CPU implementation of softmax
void softmax_cpu(float *input, float *output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float max_val = input[i * cols];
        for (int j = 1; j < cols; j++) {
            if (input[i * cols + j] > max_val) {
                max_val = input[i * cols + j];
            }
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum_exp += expf(input[i * cols + j] - max_val);
        }
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = expf(input[i * cols + j] - max_val) / sum_exp;
        }
    }
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
    softmax_kernel_reduction2<<<numBlocks, blockSize>>>(input_device, output_device, rows, cols);
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
