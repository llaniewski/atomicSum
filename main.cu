#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

typedef unsigned long long int double_i;
#define R2I(x) __double_as_longlong(x)
#define I2R(x) __longlong_as_double(x)

  __device__ inline void atomicAddP(double* address, double val)
  {
	  if (val != 0.0) {
		double_i* address_as_ull = (double_i*) address;
		double_i  old = *address_as_ull;
		double_i  assumed, nw;
		do {
			assumed = old;
			nw = R2I(val + I2R(assumed));
			old = atomicCAS(address_as_ull, assumed, nw);
		} while (assumed != old);
	  }
  }

__shared__ double  sumtab[256];

__device__ inline void atomicSum_f(double * sum) {
	int i = blockDim.x*blockDim.y;
	int k = blockDim.x*blockDim.y;
	int j = blockDim.x*threadIdx.y + threadIdx.x;
	double v;
	while (i> 1) {
		k = i >> 1;
		i = i - k;
		__syncthreads();
		if (j<k) v = sumtab[j] + sumtab[j+i];
		__syncthreads();
		if (j<k) sumtab[j] = v;
	}
	__syncthreads();
	if (j==0) {
		double val = sumtab[0];
		if (val != 0.0) {
			atomicAddP(sum, val);
		}
	}
}

__device__ inline void atomicSum(double * sum, double val)
{
	__syncthreads();
	int j = blockDim.x*threadIdx.y + threadIdx.x;
	sumtab[j] = val;
	__syncthreads();
	atomicSum_f(sum);
}


const int M = 10;

__host__ __device__ int bin(double x) {
	int i = floor(x*M);
	if (i < 0) return 0;
	if (i < M) return i;
	return M-1;
}

__global__ void test0(double* tab, int N, double * hist) {
	int i = threadIdx.x + blockDim.x*( threadIdx.y + blockDim.y*( blockIdx.x ) );
	int j = bin(tab[i]);
	double val = 1.0;
	atomicAddP(&hist[j],val);
}

__global__ void test1(double* tab, int N, double * hist) {
	int i = threadIdx.x + blockDim.x*( threadIdx.y + blockDim.y*( blockIdx.x ) );
	int j = bin(tab[i]);
	double val;
	for (int k=0; k<M; k++) {
		if (j == k) {
			val = 1.0;
		} else {
			val = 0.0;
		}
		atomicAddP(&hist[k],val);
	}
}

__global__ void test2(double* tab, int N, double * hist) {
	int i = threadIdx.x + blockDim.x*( threadIdx.y + blockDim.y*( blockIdx.x ) );
	int j = bin(tab[i]);
	double val;
	for (int k=0; k<M; k++) {
		if (j == k) {
			val = 1.0;
		} else {
			val = 0.0;
		}
		atomicSum(&hist[k],val);
	}
}


__device__ inline void atomicSumWarp(double * sum, double val)
{
	#define FULL_MASK 0xffffffff
	for (int offset = 16; offset > 0; offset /= 2)
	    val += __shfl_down_sync(FULL_MASK, val, offset);
	if (threadIdx.x == 0) atomicAddP(sum,val);
//	if (threadIdx.x == 0) *sum += val;
}

__global__ void test3(double* tab, int N, double * hist) {
	int i = threadIdx.x + blockDim.x*( threadIdx.y + blockDim.y*( blockIdx.x ) );
	int j = bin(tab[i]);
	double val;
	for (int k=0; k<M; k++) {
		if (j == k) {
			val = 1.0;
		} else {
			val = 0.0;
		}
		atomicSumWarp(&hist[k],val);
	}
}

__global__ void test4(double* tab, int N, double * hist) {
	int i = threadIdx.x + blockDim.x*( threadIdx.y + blockDim.y*( blockIdx.x ) );
	int j = bin(tab[i]);
	double val;
	for (int k=0; k<M; k++) {
		if (j == k) {
			val = 1.0;
		} else {
			val = 0.0;
		}
		if (__any_sync(FULL_MASK, val != 0)) atomicSumWarp(&hist[k],val);
	}
}


__device__ inline void atomicSumWarpX(double * sum, double val)
{
	#define FULL_MASK 0xffffffff
	for (int offset = 16; offset > 0; offset /= 2)
	    val += __shfl_xor_sync(FULL_MASK, val, offset);
	if (threadIdx.x == 0) atomicAddP(sum,val);
//	if (threadIdx.x == 0) *sum += val;
}

__global__ void test5(double* tab, int N, double * hist) {
	int i = threadIdx.x + blockDim.x*( threadIdx.y + blockDim.y*( blockIdx.x ) );
	int j = bin(tab[i]);
	double val;
	for (int k=0; k<M; k++) {
		if (j == k) {
			val = 1.0;
		} else {
			val = 0.0;
		}
		if (__any_sync(FULL_MASK, val != 0)) atomicSumWarpX(&hist[k],val);
	}
}


int main () {
//	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	srand(0);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int N = 1024000*2;
	double *tab;
	double *gtab;
	double *hist;
	double *chist;
	double *ghist;
	cudaMallocHost(&tab, N*sizeof(double));
	cudaMalloc(&gtab, N*sizeof(double));
	cudaMallocHost(&hist, M*sizeof(double));
	cudaMallocHost(&chist, M*sizeof(double));
	cudaMalloc(&ghist, M*sizeof(double));
	printf("Start...\n");
	dim3 blx, thx;
	thx.x = 32;
	thx.y = 8;
	thx.z = 1;
	blx.x = N/(thx.x*thx.y);
	blx.y = 1;
	blx.z = 1;
	for (int sharebank = 0; sharebank < 1; sharebank ++) {
		if (sharebank == 1) {
			cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		}
		for (int dist=0; dist<2; dist++) {
			std::string dist_name;
			switch (dist) {
				case 0: dist_name = "rand"; for (int i=0; i<N; i++) tab[i] = 1.0*rand()/RAND_MAX; break;
				case 1: dist_name = "unif"; for (int i=0; i<N; i++) tab[i] = 1.0*i/N; break;
			}
			for (int i=0; i<M; i++) chist[i] = 0;
			for (int i=0; i<N; i++) chist[bin(tab[i])]++;
			for (int test = 0; test < 6; test ++) {
				std::string test_name;
				cudaMemcpy(gtab, tab, N*sizeof(double), cudaMemcpyHostToDevice);
				for (int rep=0; rep<1; rep++) {
					for (int i=0; i<M; i++) hist[i] = 0;
					cudaMemcpy(ghist, hist, M*sizeof(double), cudaMemcpyHostToDevice);
					cudaEventRecord(start);
					switch (test) {
						case 0: test_name = "atomicAdd"; test0<<<blx,thx>>>(gtab, N, ghist); break;
						case 1: test_name = "for atomicAdd"; test1<<<blx,thx>>>(gtab, N, ghist); break;
						case 2: test_name = "for atomicSum"; test2<<<blx,thx>>>(gtab, N, ghist); break;
						case 3: test_name = "for atomicSumWarp"; test3<<<blx,thx>>>(gtab, N, ghist); break;
						case 4: test_name = "for atomicSumWarp (__any)"; test4<<<blx,thx>>>(gtab, N, ghist); break;
						case 5: test_name = "for atomicSumWarpX (__any)"; test5<<<blx,thx>>>(gtab, N, ghist); break;
					}
					cudaEventRecord(stop);
					cudaEventSynchronize(stop);
					float milliseconds = 0;
					cudaEventElapsedTime(&milliseconds, start, stop);
					cudaMemcpy(hist, ghist, M*sizeof(double), cudaMemcpyDeviceToHost);
					bool good = true;
					for (int i=0; i<M; i++) if (fabs(hist[i] - chist[i]) > 1e-10) { good = false; break; }
					if (good) printf("[ GOOD ]"); else printf("[ BAD  ]");
					printf(" --- %10.2f ms --- %s - %s\n", milliseconds, dist_name.c_str(), test_name.c_str());
				}
			}
		}
	}
	cudaFree(gtab);
	cudaFreeHost(tab);
	cudaDeviceReset();
	return 0;
}
