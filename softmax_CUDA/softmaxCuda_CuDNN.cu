#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CUDNN_CHECK(call) \
do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// Kernel declaration
void softmax_cudnn(float *input, float *output, int num_samples, int num_classes) {
    // Set up cuDNN
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));
    
    cudnnTensorDescriptor_t input_desc, output_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_samples, num_classes, 1, 1));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_samples, num_classes, 1, 1));
    
    // Perform softmax operation
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_desc, input, &beta, output_desc, output));
    
    // Clean up
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

// CPU implementation of softmax using cudnn
void softmax_cudnn_cpu(float *input, float *output, int num_samples, int num_classes) {
    for (int i = 0; i < num_samples; ++i) {
        float max_val = input[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            max_val = std::max(max_val, input[i * num_classes + j]);
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += expf(input[i * num_classes + j] - max_val);
        }
        for (int j = 0; j < num_classes; ++j) {
            output[i * num_classes + j] = expf(input[i * num_classes + j] - max_val) / sum_exp;
        }
    }
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    
    const int num_samples = 8192;//std::stoi(argv[1]);
    const int num_classes = 8192;//std::stoi(argv[2]);

    // Generate random input
    std::vector<float> input(num_samples * num_classes);
    for (int i = 0; i < num_samples * num_classes; ++i) {
        input[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f;
    }

    // Allocate memory for output on CPU
    std::vector<float> output_cpu(num_samples * num_classes);
    std::vector<float> output_gpu(num_samples * num_classes);

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, num_samples * num_classes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_samples * num_classes * sizeof(float)));

    cudaEvent_t start_cpu_to_gpu, stop_cpu_to_gpu;
    cudaEventCreate(&start_cpu_to_gpu);
    cudaEventCreate(&stop_cpu_to_gpu);
    cudaEvent_t start_gpu_to_cpu, stop_gpu_to_cpu;
    cudaEventCreate(&start_gpu_to_cpu);
    cudaEventCreate(&stop_gpu_to_cpu);
cudaEventRecord(start_cpu_to_gpu);

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), num_samples * num_classes * sizeof(float), cudaMemcpyHostToDevice));
cudaEventRecord(stop_cpu_to_gpu);
    cudaEventSynchronize(stop_cpu_to_gpu);
    // Create CUDA events for timing GPU execution
     cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventRecord(start_kernel);

    // Call softmax_cudnn on GPU
    softmax_cudnn(d_input, d_output, num_samples, num_classes);
cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    // Record stop event for GPU
cudaEventRecord(start_gpu_to_cpu);

    // Copy output from device to host
    CUDA_CHECK(cudaMemcpy(output_gpu.data(), d_output, num_samples * num_classes * sizeof(float), cudaMemcpyDeviceToHost));

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
    softmax_cudnn_cpu(input.data(), output_cpu.data(), num_samples, num_classes);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);

    // Verify results
    bool passed = true;
    for (int i = 0; i < num_samples * num_classes; ++i) {
        if (std::abs(output_cpu[i] - output_gpu[i]) > 1e-5) {
            std::cout << "Verification failed at index " << i << ": CPU = " << output_cpu[i] << ", GPU = " << output_gpu[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Verification passed!" << std::endl;
    }

    // Print CPU execution time
    std::cout << "CPU Execution Time: " << duration_cpu.count()/1000 << " ms" << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    // Destroy events
    cudaEventDestroy(start_cpu_to_gpu);
    cudaEventDestroy(stop_cpu_to_gpu);
    cudaEventDestroy(start_gpu_to_cpu);
    cudaEventDestroy(stop_gpu_to_cpu);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    return 0;
}
