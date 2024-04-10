#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define BLOCK_DIM 8
// const int NUM_REPS = 100;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code == cudaSuccess) return;
    fprintf(stderr,"Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
}

double timeStamp() {
    struct timeval tv; 
    gettimeofday(&tv, NULL);
    return tv.tv_usec / 1000.0 + tv.tv_sec;
}

void displayResults(float *A, float *T, int M, int N){
    // display results
	printf("Matrix A: \n");
	printf("----------\n");
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("A: %f ", A[i * N + j]);
		}
		printf("\n");
	}

	printf("----------\n");
	printf("Transpose: \n");
	printf("----------\n");
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < M; ++j) {
			printf("%f ", T[i * M + j]);
		}
		printf("\n");
	}
}

void transposeCPU(float *A, float *T, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T[j * M + i] = A[i * N + j];
        }
    }
}

void validateResults(float *h_A, float *h_T, int M, int N){
    // Allocate memory for the transpose matrix on CPU
    float *h_T_CPU = (float *)malloc(M * N * sizeof(float));
    // Transpose matrix A on CPU
    transposeCPU(h_A, h_T_CPU, M, N);

    // Validate the results
    int incorrectCount = 0;
    for (int i = 0; i < M * N; ++i) {
        if (abs(h_T_CPU[i] - h_T[i]) > 1e-5) {
            incorrectCount++;
            // Uncomment the next line to print each incorrect element
            // printf("Mismatch at index %d, CPU: %f, GPU: %f\n", i, h_T_CPU[i], h_T[i]);
        }
    }
    
    if (incorrectCount == 0) {
        printf("Validation Passed!\n");
    } else {
        printf("Validation Failed: %d elements incorrect.\n", incorrectCount);
    }

    // Clean up CPU transpose matrix
    free(h_T_CPU);
}

__global__ void transposeNaive(float *d_A, float *d_T, int M, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// swap elements via transpose
	if (row < M && col < N) {
		d_T[col * M + row] = d_A[row * N + col];
	}
}

__global__ void transpose(float *d_A, float *d_T, int M, int N)
{
	__shared__ float block[TILE_DIM][TILE_DIM+1];
	
	unsigned int row = blockIdx.y * TILE_DIM + threadIdx.y;
	unsigned int col = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int index_in = row * N + col;
    unsigned int index_out = col * M + row;
	
    if((row < M) && (col < N) && (index_in < M*N)) {
		block[threadIdx.y][threadIdx.x] = d_A[index_in];
	}

	__syncthreads();

	row = blockIdx.y * TILE_DIM + threadIdx.x;
	col = blockIdx.x * TILE_DIM + threadIdx.y;
	if((row < M) && (col < N) && (index_out < M*N)) {
		d_T[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

torch::Tensor forward(torch::Tensor A) {
    // A and B are 4D tensors in row major format: 
    // A = (batchsize, head, M, K)
    const int M = A.size(2);
    const int N = A.size(3);

    // Initialize A, Z to host memory
    torch::Tensor C = torch::zeros({A.size(0), A.size(1), N, M}, A.options().device(torch::kCUDA));
    auto A_data = A.data_ptr<float>();
    auto C_data = C.data_ptr<float>();

	dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
	dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (M + blockDim.y - 1)/blockDim.y);

    double start, end;
    start = timeStamp();
    transpose<<<gridDim, blockDim>>>(A_data, C_data, M, N);
    // transposeNaive<<<gridDim, blockDim>>>(A_data, C_data, M, N);
    cudaDeviceSynchronize();
    end = timeStamp();

    printf("GPU execution time: %.4f milliseconds\n", (end-start));

	return C;
}


// % nvcc -arch sm_89 transpose.cu -o transpose
// % transpose