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
#define BLOCK_DIM 32
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

__global__ void transposeSharedMem(float *d_A, float *d_T, int M, int N) {
	__shared__ float tile[TILE_DIM][TILE_DIM+1];
	
	unsigned int row = blockIdx.y * TILE_DIM + threadIdx.y;
	unsigned int col = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int index_in = row * N + col;
	
    if((row < M) && (col < N) && (index_in < M*N)) {
        tile[threadIdx.y][threadIdx.x] = d_A[index_in];
	}
    
	__syncthreads();
    
	row = blockIdx.y * TILE_DIM + threadIdx.x;
	col = blockIdx.x * TILE_DIM + threadIdx.y;
	if((row < M) && (col < N)) {
        unsigned int index_out = col * M + row;
		d_T[index_out] = tile[threadIdx.x][threadIdx.y];
	}
}

__global__ void copySharedMem(float *idata, float *odata, int M, int N)
{
  __shared__ float tile[TILE_DIM * TILE_DIM];
    // block is 8 x 32 so x is 8 and y is 32
  int x = blockIdx.x * TILE_DIM + threadIdx.x*4; // 0,4,8,12,16,20,24,28
  int y = blockIdx.y * TILE_DIM + threadIdx.y; // 0,1,2,3,...,31
    //   printf("%d",blockIdx.y);
    //   printf("%d",blockIdx.x);
  int width = gridDim.x * TILE_DIM;

    if (x>=N || y>=M){return;}
    // load all your elements into shared memory
    for (int j=0; j<4;j+=1){
        tile[threadIdx.y*TILE_DIM+threadIdx.x*4+j]=idata[y*N+x+j]; //thread 0: 0,1,2,3, thread 1: 4,5,6,7 ... ,28,29,30,31, loading is done with offset
    }
    
    __syncthreads();
  // shared memory now contain an exact copy of the tile. We need to load this back coalesced now

    // calculate the elelements that this thread will load back and to where it will load back
    //idx=(threadIdx.x*BLOCK_ROWS+j)*TILE_DIM
    int idy = threadIdx.y;
    int idx;

    for (int j = 0; j < 4; j += 1){
        idx=(threadIdx.x*4+j); // 
        odata[blockIdx.x*TILE_DIM*N + blockIdx.y*TILE_DIM + idy*M + idx]=tile[idx*TILE_DIM+idy];
    }}



__global__ void copySharedMem_coalesced(float *idata, float *odata, int M, int N)
{
  __shared__ float tile[TILE_DIM * TILE_DIM];
    // block is 8 x 32 so x is 8 and y is 32
    int x = blockIdx.x * TILE_DIM + threadIdx.x; // 0,4,8,12,16,20,24,28
    int y = blockIdx.y * TILE_DIM + threadIdx.y; // 0,1,2,3,...,31
    //   printf("%d",blockIdx.y);
    //   printf("%d",blockIdx.x);
    int width = gridDim.x * TILE_DIM;

    if (x>=N || y>=M){return;}
    // load all your elements into shared memory
    for (int j=0; j<TILE_DIM;j+=BLOCK_ROWS){
        tile[(threadIdx.y+j)*TILE_DIM+threadIdx.x]=idata[(y+j)*N+x]; //thread 0: 0,1,2,3, thread 1: 4,5,6,7 ... ,28,29,30,31, loading is done with offset
    }
    
    __syncthreads();
    // shared memory now contain an exact copy of the tile. We need to load this back coalesced now

    // calculate the elelements that this thread will load back and to where it will load back
    //idx=(threadIdx.x*BLOCK_ROWS+j)*TILE_DIM
    int idy;
    int idx=threadIdx.x;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
        idy=(threadIdx.y+j); // 
        odata[blockIdx.x*TILE_DIM*N + blockIdx.y*TILE_DIM + idy*M + idx]=tile[idx*TILE_DIM+idy];
    }
}

void run_transpose_cublas(torch::Tensor A, torch::Tensor C) {
    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;  // cuBLAS functions status
    cublasHandle_t handle;
    stat = cublasCreate(&handle);

    const float alpha = 1.0;
    const float beta = 0.0;

    // loop over batchsize and head
    for (int i = 0; i < A.size(0); i++) {
        for (int j = 0; j < A.size(1); j++) {
            // get the i-th batch and j-th head
            torch::Tensor Aij = A[i][j];
            torch::Tensor Cij = C[i][j];

            // perform matrix transposition using cublasSgeam
            stat = cublasSgeam(handle,
                               CUBLAS_OP_T,  // transpose A
                               CUBLAS_OP_N,  // do not transpose B (NULL)
                               Aij.size(1),  // number of rows of A^T
                               Aij.size(0),  // number of columns of A^T
                               &alpha,
                               Aij.data_ptr<float>(),  // pointer to A
                               Aij.size(1),  // leading dimension of A
                               &beta,
                               NULL,  // B is NULL
                               Aij.size(1),  // set ldb to a valid value
                               Cij.data_ptr<float>(),  // pointer to C
                               Aij.size(1));  // leading dimension of C
        }
    }

    cublasDestroy(handle);
}

torch::Tensor forward(torch::Tensor A) {
    // A and B are 4D tensors in row major format:
    // A = (batchsize, head, M, K)
    double start, end;
    start = timeStamp();
    const int M = A.size(2);
    const int N = A.size(3);

    // Initialize A, Z to host memory, A is MxN and C is NxM. Thus x should be 

    
    torch::Tensor C = torch::zeros({A.size(0), A.size(1), N, M}, A.options().device(torch::kCUDA));
    auto A_data = A.data_ptr<float>();
    auto C_data = C.data_ptr<float>();

	dim3 blockDim(32, 8); // each thread will process 4 cosnecutive 
	dim3 gridDim((N + 32 - 1)/32, (M + 32 - 1)/32);
    // dim3 blockDim(BLOCK_DIM, BLOCK_DIM); // each thread will process 4 cosnecutive 
	// dim3 gridDim((N + BLOCK_DIM - 1)/BLOCK_DIM, (M + BLOCK_DIM - 1)/BLOCK_DIM);

   
    //transposeCoalesced<<<gridDim, blockDim>>>(A_data, C_data, M, N);
    //transposeSharedMem<<<gridDim, blockDim>>>(A_data, C_data, M, N);
    // copySharedMem_coalesced<<<gridDim, blockDim>>>(A_data, C_data, M, N);
    run_transpose_cublas(A, C);
    cudaDeviceSynchronize();
    end = timeStamp();

    printf("GPU execution time: %.4f milliseconds\n", (end-start));

	return C;
}


// % nvcc -arch sm_89 transpose.cu -o transpose
// % transpose