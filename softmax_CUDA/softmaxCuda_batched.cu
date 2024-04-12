#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>

// Kernel declaration
__global__ void softmax_kernel_naive_batched(float *input, float *output, int batch_size, int n_head, int seq_len, int head_embd) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batch_size * n_head) {
        int batch_idx = idx / n_head;
        int head_idx = idx % n_head;

        for (int i = 0; i < seq_len; i++) {
            // Compute max value within the sequence for each batch and head
            float max_val = input[(batch_idx * n_head * seq_len * head_embd) + (head_idx * seq_len * head_embd) + i * head_embd];
            for (int j = 1; j < head_embd; j++) {
                float val = input[(batch_idx * n_head * seq_len * head_embd) + (head_idx * seq_len * head_embd) + i * head_embd + j];
                if (val > max_val) {
                    max_val = val;
                }
            }

            // Compute softmax for each element in the sequence
            float sum_exp = 0.0f;
            for (int j = 0; j < head_embd; j++) {
                float val = input[(batch_idx * n_head * seq_len * head_embd) + (head_idx * seq_len * head_embd) + i * head_embd + j];
                sum_exp += expf(val - max_val);
            }

            for (int j = 0; j < head_embd; j++) {
                float val = input[(batch_idx * n_head * seq_len * head_embd) + (head_idx * seq_len * head_embd) + i * head_embd + j];
                output[(batch_idx * n_head * seq_len * head_embd) + (head_idx * seq_len * head_embd) + i * head_embd + j] = expf(val - max_val) / sum_exp;
            }
        }
    }
}

// CPU implementation of softmax for batched data
void softmax_batched_cpu(float *input, float *output, int batch_size, int n_head, int seq_len, int head_embd) {
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < n_head; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                // Compute max value within the sequence for each batch and head
                float max_val = input[(b * n_head * seq_len * head_embd) + (h * seq_len * head_embd) + i * head_embd];
                for (int j = 1; j < head_embd; ++j) {
                    float val = input[(b * n_head * seq_len * head_embd) + (h * seq_len * head_embd) + i * head_embd + j];
                    if (val > max_val) {
                        max_val = val;
                    }
                }

                // Compute softmax for each element in the sequence
                float sum_exp = 0.0f;
                for (int j = 0; j < head_embd; ++j) {
                    float val = input[(b * n_head * seq_len * head_embd) + (h * seq_len * head_embd) + i * head_embd + j];
                    sum_exp += expf(val - max_val);
                }

                for (int j = 0; j < head_embd; ++j) {
                    float val = input[(b * n_head * seq_len * head_embd) + (h * seq_len * head_embd) + i * head_embd + j];
                    output[(b * n_head * seq_len * head_embd) + (h * seq_len * head_embd) + i * head_embd + j] = expf(val - max_val) / sum_exp;
                }
            }
        }
    }
}

int main() {
    const int batch_size = 1;   // Example: number of batches
    const int n_head = 1;       // Example: number of heads
    const int seq_len = 4096;     // Example: sequence length
    const int head_embd = 4096;   // Example: embedding size per head

    // Allocate memory on host
    float *input_host = (float*)malloc(batch_size * n_head * seq_len * head_embd * sizeof(float));
    float *output_host_cpu = (float*)malloc(batch_size * n_head * seq_len * head_embd * sizeof(float));
    float *output_host_gpu = (float*)malloc(batch_size * n_head * seq_len * head_embd * sizeof(float));

    // Initialize input data
    for (int i = 0; i < batch_size * n_head * seq_len * head_embd; ++i) {
        input_host[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on device
    float *input_device, *output_device;
    cudaMalloc(&input_device, batch_size * n_head * seq_len * head_embd * sizeof(float));
    cudaMalloc(&output_device, batch_size * n_head * seq_len * head_embd * sizeof(float));

    // Transfer input data from host to device
    cudaMemcpy(input_device, input_host, batch_size * n_head * seq_len * head_embd * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size
    int blockSize = 256;
    int numBlocks = (batch_size * n_head + blockSize - 1) / blockSize;

    // Launch kernel
    softmax_kernel_naive_batched<<<numBlocks, blockSize>>>(input_device, output_device, batch_size, n_head, seq_len, head_embd);

    // Transfer output data from device to host
    cudaMemcpy(output_host_gpu, output_device, batch_size * n_head * seq_len * head_embd * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform softmax on CPU for verification
    softmax_batched_cpu(input_host, output_host_cpu, batch_size, n_head, seq_len, head_embd);

    // Compare CPU and GPU results
    bool passed = true;
    for (int i = 0; i < batch_size * n_head * seq_len * head_embd; ++i) {
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

    return 0;
}
