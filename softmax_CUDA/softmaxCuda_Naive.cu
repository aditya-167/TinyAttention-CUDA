#include <iostream>
#include <cmath>
#include <cstdlib>

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
    // Parse command-line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <rows> <cols>" << std::endl;
        return 1;
    }
    const int rows = std::stoi(argv[1]);
    const int cols = std::stoi(argv[2]);

    // Allocate memory on host
    float *input_host = (float*)malloc(rows * cols * sizeof(float));
    float *output_host_cpu = (float*)malloc(rows * cols * sizeof(float));
    float *output_host_gpu = (float*)malloc(rows * cols * sizeof(float));

    // Initialize input data
    for (int i = 0; i < rows * cols; i++) {
        input_host[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on device
    float *input_device, *output_device;
    cudaMalloc(&input_device, rows * cols * sizeof(float));
    cudaMalloc(&output_device, rows * cols * sizeof(float));

    // Transfer input data from host to device
    cudaMemcpy(input_device, input_host, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size
    int blockSize = 256;
    int numBlocks = (rows + blockSize - 1) / blockSize;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event for GPU
    cudaEventRecord(start);

    // Launch kernel
    softmax_kernel_naive<<<numBlocks, blockSize>>>(input_device, output_device, rows, cols);
    cudaDeviceSynchronize();

    // Record stop event for GPU
    cudaEventRecord(stop);

    // Transfer output data from device to host
    cudaMemcpy(output_host_gpu, output_device, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate GPU execution time
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Record start time for CPU
    cudaEventRecord(start);

    // Perform softmax on CPU for verification
    softmax_cpu(input_host, output_host_cpu, rows, cols);

    // Record stop time for CPU
    cudaEventRecord(stop);

    // Synchronize event recording
    cudaEventSynchronize(stop);

    // Calculate CPU execution time
    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start, stop);

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
    std::cout << "CPU Execution Time: " << cpu_time << " ms" << std::endl;

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
