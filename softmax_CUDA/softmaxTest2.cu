#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MAX

#define MAX_BLOCK_SIZE 1024 // Define your maximum block size here

__global__ void softmax(float *d_in, float *d_out, int N) {
    __shared__ float shared_exp[MAX_BLOCK_SIZE];
    __shared__ float shared_sum;

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float max_val = -FLT_MAX;
    for (int offset = 0; offset < N; offset += blockDim.x) {
        int idx = offset + threadIdx.x;
        if (idx < N && d_in[idx] > max_val) {
            max_val = d_in[idx];
        }
    }
    __syncthreads();

    if (col < N) {
        printf("Thread %d: Input value: %f, Max value: %f\n", threadIdx.x, d_in[col], max_val);
    }

    float local_exp = 0.0f;
    if (col < N) {
        local_exp = expf(d_in[col] - max_val); // Subtract max_val for numerical stability
        printf("Thread %d: Local exp value: %f\n", threadIdx.x, local_exp);
        shared_exp[threadIdx.x] = local_exp;
    } else {
        shared_exp[threadIdx.x] = 0.0f; // Zero padding for elements beyond N
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_exp[i];
        }
        printf("Block %d: Shared sum: %f\n", blockIdx.x, sum);
        shared_sum = sum;
    }
    __syncthreads();

    if (col < N) {
        d_out[col] = shared_exp[threadIdx.x] / shared_sum;
    }
}

int main() {
    const int N = 1024; // Example array size
    const int BLOCK_SIZE = 256; // Example block size
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *h_input = new float[N];
    float *h_output_cpu = new float[N];
    float *h_output_gpu = new float[N];

    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; // Random values between 0 and 1
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    softmax<<<NUM_BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize(); // Ensure kernel execution is finished before copying data back

    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate softmax on CPU for verification
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += expf(h_input[j]);
        }
        h_output_cpu[i] = expf(h_input[i]) / sum;
    }

    // Compare CPU and GPU results
    bool verification_failed = false;
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_output_cpu[i] - h_output_gpu[i]) > 1e-5) {
            std::cerr << "Verification failed at index " << i << std::endl;
            std::cerr << "CPU result: " << h_output_cpu[i] << ", GPU result: " << h_output_gpu[i] << std::endl;
            verification_failed = true;
            break;
        }
    }

    if (!verification_failed) {
        std::cout << "Verification passed" << std::endl;
    }

    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
