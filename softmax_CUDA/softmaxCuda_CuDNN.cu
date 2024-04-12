#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cudnn.h>
void softmax_cudnn(float *input, float *output, int num_samples, int num_classes) {
    // Set up cuDNN
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_samples, num_classes, 1, 1);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_samples, num_classes, 1, 1);
    
    // Perform softmax operation
    float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_desc, input, &beta, output_desc, output);
    
    // Clean up
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroy(cudnn);
}
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

int main() {
    // Define input dimensions
    const int num_samples = 4096;
    const int num_classes = 4096;

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

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), num_samples * num_classes * sizeof(float), cudaMemcpyHostToDevice));

    // Create CUDA events for timing GPU execution
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    // Record start event for GPU
    CUDA_CHECK(cudaEventRecord(start_gpu));

    // Call softmax_cudnn on GPU
    softmax_cudnn(d_input, d_output, num_samples, num_classes);

    // Record stop event for GPU
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    // Calculate GPU execution time
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu));

    // Copy output from device to host
    CUDA_CHECK(cudaMemcpy(output_gpu.data(), d_output, num_samples * num_classes * sizeof(float), cudaMemcpyDeviceToHost));

    // Perform softmax on CPU for verification
    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_cudnn_cpu(input.data(), output_cpu.data(), num_samples, num_classes);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);

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

    // Print GPU execution time
    std::cout << "GPU Execution Time: " << gpu_time << " ms" << std::endl;

    // Print CPU execution time
    std::cout << "CPU Execution Time: " << duration_cpu.count()/1000 << " ms" << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    // Destroy events
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));

    return 0;
}
