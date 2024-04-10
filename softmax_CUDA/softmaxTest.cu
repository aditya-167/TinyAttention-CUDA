#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cuda.h>


const int TILE_DIM_Y = 32;  // Tile dimension for rows
const int TILE_DIM_X = 32;  // Tile dimension for columns// must be 32 for this method

template <typename T>
__global__ void softmaxKernel2D_rows(const T* input, T* exp_sums, int N, int M) {
    int row = blockIdx.y * TILE_DIM_Y + threadIdx.y;
    int col = blockIdx.x * TILE_DIM_X + threadIdx.x;
    T val = 0;
    // Copy data from global memory to shared memory
    if (row < N && col < M) {
        if (sizeof(T) == 8)
          val = exp(input[row * M + col]);
        else
          val = expf(input[row * M + col]);
    }
    // warp shuffle reduction
    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i>>=1)
        val += __shfl_xor_sync(0xffffffff, val, i, 32);
    // update global value for row
    if ((threadIdx.x == 0) && (row < N)) atomicAdd(exp_sums+row, val);
}

template <typename T>
__global__ void softmaxKernel2D_elementwise(const T* input, const T* exp_sums, T *output, int N, int M) {
    int row = blockIdx.y * TILE_DIM_Y + threadIdx.y;
    int col = blockIdx.x * TILE_DIM_X + threadIdx.x;
    // Compute the softmax values
    if (row < N && col < M) {
      if (sizeof(T) == 8)
        output[row * M + col] = exp(input[row * M + col])/ exp_sums[row];
      else
        output[row * M + col] = expf(input[row * M + col])/ exp_sums[row];
    }
}

template <typename T>
void randomInit(std::vector<T> &x) {
    // Pseudo-random float vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> unif(-1, 1);
    for (int i = 0; i < x.size(); i++) {
        x[i] = unif(gen);
    }
}
template <typename T>
bool verifyOutput(const T* output, int N, int M) {
    for (int i = 0; i < N; i++) {
        T softmax_val = 0;
        for (int j = 0; j < M; j++) {
            softmax_val += output[i * M + j];
        }
        // Check if row sums to one
        if (fabs(softmax_val - 1) > 1e-4) {
            std::cout << softmax_val << std::endl;
            return false;
        }
    }
    return true;
}

using mt = float;

int main() {
    int N = 2048; // number of rows in dataset
    int M = 512; // number of columns in dataset
    std::vector<mt> h_input(N * M);
    std::vector<mt> h_output(N * M);
    mt *d_input, *d_output, *d_sums;

    // Randomly initialize input
    randomInit(h_input);

    // Allocate memory on the device
    cudaMalloc(&d_input, N * M * sizeof(mt));
    cudaMalloc(&d_output, N * M * sizeof(mt));
    cudaMalloc(&d_sums, N * sizeof(mt));
    cudaMemset(d_sums, 0, N*sizeof(mt));
    // Copy to device
    cudaMemcpy(d_input,  h_input.data(),  N * M * sizeof(mt), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threads(TILE_DIM_X, TILE_DIM_Y);
    dim3 blocks((M + TILE_DIM_X - 1) / TILE_DIM_X, (N + TILE_DIM_Y - 1) / TILE_DIM_Y);

    // Launch the kernel
    softmaxKernel2D_rows<<<blocks, threads>>>(d_input, d_sums, N, M);
    softmaxKernel2D_elementwise<<<blocks, threads>>>(d_input, d_sums, d_output,  N, M);

    // Copy to host
    cudaMemcpy(h_output.data(), d_output, N * M * sizeof(mt), cudaMemcpyDeviceToHost);

    // Check
    if (verifyOutput(h_output.data(), N, M)) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failure!" << std::endl;
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}