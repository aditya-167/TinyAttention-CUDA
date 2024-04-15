#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
/*


__global__ void softmax2D_kernel(float *d_in, float *d_out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Find max value in the row
        float max_val = d_in[row * N];
        for (int i = 1; i < N; ++i) {
            max_val = fmaxf(max_val, d_in[row * N + i]);
        }
        
        // Subtract max value from each element for numerical stability
        float sum_exp = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum_exp += expf(d_in[row * N + i] - max_val);
        }

        // Calculate softmax for each element in the row
        d_out[row * N + col] = expf(d_in[row * N + col] - max_val) / sum_exp;
    }
}

void softmax2D_cpu(float *d_in, float *d_out, int M, int N) {
    for (int row = 0; row < M; ++row) {
        // Find max value in the row
        float max_val = d_in[row * N];
        for (int i = 1; i < N; ++i) {
            max_val = fmaxf(max_val, d_in[row * N + i]);
        }
        
        // Subtract max value from each element for numerical stability
        float sum_exp = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum_exp += expf(d_in[row * N + i] - max_val);
        }

        // Calculate softmax for each element in the row
        for (int col = 0; col < N; ++col) {
            d_out[row * N + col] = expf(d_in[row * N + col] - max_val) / sum_exp;
        }
    }
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_rows> <num_cols>" << std::endl;
        return 1;
    }
    const int M = std::stoi(argv[1]);
    const int N = std::stoi(argv[2]);

    // Allocate memory on host
    float *input_host = (float*)malloc(M * N * sizeof(float));
    float *output_host_cpu = (float*)malloc(M * N * sizeof(float));
    float *output_host_gpu = (float*)malloc(M * N * sizeof(float));

    // Initialize input data
    for (int i = 0; i < M * N; i++) {
        input_host[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on device
    float *input_device, *output_device;
    cudaMalloc(&input_device, M * N * sizeof(float));
    cudaMalloc(&output_device, M * N * sizeof(float));

    // Transfer input data from host to device
    cudaMemcpy(input_device, input_host, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Create CUDA events for timing GPU execution
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Record start event for GPU
    cudaEventRecord(start_gpu);

    // Launch kernel
    softmax2D_kernel<<<gridSize, blockSize>>>(input_device, output_device, M, N);
    cudaDeviceSynchronize();

    // Record stop event for GPU
    cudaEventRecord(stop_gpu);

    // Transfer output data from device to host
    cudaMemcpy(output_host_gpu, output_device, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Synchronize GPU

    // Calculate GPU execution time
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    // Create CUDA events for timing CPU execution
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);

    // Record start event for CPU
    cudaEventRecord(start_cpu);

    // Perform softmax on CPU for verification
    softmax2D_cpu(input_host, output_host_cpu, M, N);

    // Record stop event for CPU
    cudaEventRecord(stop_cpu);

    // Synchronize CPU
    cudaDeviceSynchronize();

    // Calculate CPU execution time
    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);

    // Compare CPU and GPU results
    bool passed = true;
    for (int i = 0; i < M * N; i++) {
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
    std::cout << "CPU Execution Time: " << cpu_time << " ms" << std::endl;

    // Free memory
    free(input_host);
    free(output_host_cpu);
    free(output_host_gpu);
    cudaFree(input_device);
    cudaFree(output_device);

    // Destroy events
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);

    return 0;
}
*/
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

// CUDA kernel for 2D softmax calculation
__global__ void softmax2D_kernel(float *d_in, float *d_out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Find max value in the row
        float max_val = d_in[row * N];
        for (int i = 1; i < N; ++i) {
            max_val = fmaxf(max_val, d_in[row * N + i]);
        }
        
        // Subtract max value from each element for numerical stability
        float sum_exp = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum_exp += expf(d_in[row * N + i] - max_val);
        }

        // Calculate softmax for each element in the row
        d_out[row * N + col] = expf(d_in[row * N + col] - max_val) / sum_exp;
    }
}

__global__ void softmax_kernel_reduction2(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Shared memory for partial maximum values
        __shared__ float sdata_max[BLOCK_SIZE_X * BLOCK_SIZE_Y];
        // Shared memory for partial sum of exponentials
        __shared__ float sdata_sum[BLOCK_SIZE_X * BLOCK_SIZE_Y];

        // Find the maximum value within the row using parallel reduction
        float max_val = -INFINITY;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float val = input[row * cols + i];
            max_val = fmaxf(max_val, val);
        }
        sdata_max[threadIdx.x + threadIdx.y * blockDim.x] = max_val;

        // Reduce maximum value within the block
        __syncthreads();
        for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (threadIdx.x + threadIdx.y * blockDim.x < s) {
                sdata_max[threadIdx.x + threadIdx.y * blockDim.x] = fmaxf(sdata_max[threadIdx.x + threadIdx.y * blockDim.x], sdata_max[threadIdx.x + threadIdx.y * blockDim.x + s]);
            }
            __syncthreads();
        }

        // Extract the maximum value from shared memory
        max_val = sdata_max[0];

        // Compute sum of exponentials and store it in shared memory
        float sum_exp = 0.0f;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float exp_val = expf(input[row * cols + i] - max_val);
            sum_exp += exp_val;
        }
        sdata_sum[threadIdx.x + threadIdx.y * blockDim.x] = sum_exp;

        // Reduce sum of exponentials within the block
        __syncthreads();
        for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (threadIdx.x + threadIdx.y * blockDim.x < s) {
                sdata_sum[threadIdx.x + threadIdx.y * blockDim.x] += sdata_sum[threadIdx.x + threadIdx.y * blockDim.x + s];
            }
            __syncthreads();
        }

        // Normalize the softmax values
        output[row * cols + col] = expf(input[row * cols + col] - max_val) / sdata_sum[0];
    }
}


// CPU softmax function for 2D array
void softmax2D_cpu(float *d_in, float *d_out, int M, int N) {
    for (int row = 0; row < M; ++row) {
        // Find max value in the row
        float max_val = d_in[row * N];
        for (int i = 1; i < N; ++i) {
            max_val = fmaxf(max_val, d_in[row * N + i]);
        }
        
        // Subtract max value from each element for numerical stability
        float sum_exp = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum_exp += expf(d_in[row * N + i] - max_val);
        }

        // Calculate softmax for each element in the row
        for (int col = 0; col < N; ++col) {
            d_out[row * N + col] = expf(d_in[row * N + col] - max_val) / sum_exp;
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
    const int rows = 4096;
    const int cols = 4096;

    // Allocate memory on host
    float *input_host = (float*)malloc(rows * cols * sizeof(float));
    float *output_host_cpu = (float*)malloc(rows * cols * sizeof(float));
    float *output_host_gpu = (float*)malloc(rows * cols * sizeof(float));

    // Initialize input data
    for (int i = 0; i < rows * cols; i++) {
        input_host[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Define grid and block size
    dim3 blockSize(32, 32);
    dim3 numBlocks((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

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
    softmax2D_kernel<<<numBlocks, blockSize>>>(input_device, output_device, rows, cols);
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
    softmax2D_cpu(input_host, output_host_cpu, rows, cols);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    std::cout << "CPU Execution Time: " << duration_cpu.count() << " ms" << std::endl;

    // Compare CPU and GPU results
    bool passed = verify_result(output_host_gpu, output_host_cpu, rows * cols);

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
