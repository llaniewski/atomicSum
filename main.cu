#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#if __CUDA_ARCH__ < 200
#define CROSS_NEED_ATOMICADD
#endif

#ifdef CROSS_NEED_ATOMICADD
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
#else
	#define atomicAddP atomicAdd
#endif
__shared__ double  sumtab[256];

struct block_sum {
__device__ static inline void atomicSum(double * sum, double val) {
	int i = blockDim.x*blockDim.y;
	int k = blockDim.x*blockDim.y;
	int j = blockDim.x*threadIdx.y + threadIdx.x;
	double v;
	__syncthreads();
	sumtab[j] = val;
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
};

struct direct_sum {
__device__ static inline void atomicSum(double * sum, double val)
{
	atomicAddP(sum,val);
}
};


struct warp_sum {
__device__ static inline void atomicSum(double * sum, double val)
{
	#define FULL_MASK 0xffffffff
	for (int offset = 16; offset > 0; offset /= 2)
	    val += __shfl_down_sync(FULL_MASK, val, offset);
	if (threadIdx.x == 0) atomicAddP(sum,val);
}
};

struct warp_any_sum {
__device__ static inline void atomicSum(double * sum, double val)
{
	#define FULL_MASK 0xffffffff
	if (__all_sync(FULL_MASK, val == 0.0)) return;
	for (int offset = 16; offset > 0; offset /= 2)
	    val += __shfl_down_sync(FULL_MASK, val, offset);
	if (threadIdx.x == 0) atomicAddP(sum,val);
}
};

struct warp_xor_any_sum {
__device__ static inline void atomicSum(double * sum, double val)
{
	#define FULL_MASK 0xffffffff
	if (__all_sync(FULL_MASK, val == 0.0)) return;
	for (int offset = 16; offset > 0; offset /= 2)
	    val += __shfl_xor_sync(FULL_MASK, val, offset);
	if (threadIdx.x == 0) atomicAddP(sum,val);
}
};


struct cg_block_sum {
__device__ static inline void atomicSum(double * sum, double val) {
	cg::coalesced_group active = cg::coalesced_threads();
	//cg::thread_block active = cg::this_thread_block();
	//cg::labeled_partition(active, sum);
	double val2 = cg::reduce(active, val, cg::plus<double>());
	if (active.thread_rank() == 0)  atomicAddP(sum,val2);
}
};


struct cg_block_part_lab_sum {
__device__ static inline void atomicSum(double * sum, double val) {
	cg::coalesced_group active = cg::coalesced_threads();
	cg::coalesced_group com = cg::labeled_partition(active, (unsigned long long)(void*)sum);
	double val2 = cg::reduce(com, val, cg::plus<double>());
	if (com.thread_rank() == 0)  atomicAddP(sum,val2);
}
};


// struct cg_block_part_lab_sum_invoke {
// __device__ static inline void atomicSum(double * sum, double val) {
// 	cg::coalesced_group active = cg::coalesced_threads();
// 	cg::coalesced_group com = cg::labeled_partition(active, (unsigned long long)(void*)sum);
// 	cg::experimental::reduce_update_async(com, cuda::atomic(sum),val, cg::plus<double>());
// 	//cg::invoke_one(com, atomicAddP, sum, val2);
// }
// };



const int M = 10;

__host__ __device__ int bin(double x) {
	int i = floor(x*M);
	if (i < 0) return -1;
	if (i < M) return i;
	return -1;
}

template < class T >
__global__ void DirectCall(double* tab, int N, double * hist) {
	int i = threadIdx.x + blockDim.x*( threadIdx.y + blockDim.y*( blockIdx.x ) );
	int j = bin(tab[i]);
	double val = 1.0;
	if (j != -1) T::atomicSum(&hist[j],val);
}

template < class T >
__global__ void ForCall(double* tab, int N, double * hist) {
	int i = threadIdx.x + blockDim.x*( threadIdx.y + blockDim.y*( blockIdx.x ) );
	int j = bin(tab[i]);
	double val;
	for (int k=0; k<M; k++) {
		if (j == k) {
			val = 1.0;
		} else {
			val = 0.0;
		}
		T::atomicSum(&hist[k],val);
	}
}

int main () {
	srand(0);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int N = 1024*1024*128;
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
				case 0: dist_name = "rand"; for (int i=0; i<N; i++) tab[i] = 1.1*rand()/RAND_MAX; break;
				case 1: dist_name = "unif"; for (int i=0; i<N; i++) tab[i] = 1.1*i/N; break;
			}
			for (int i=0; i<M; i++) chist[i] = 0;
			int out=0;
			for (int i=0; i<N; i++) { int j = bin(tab[i]); if (j != -1) chist[j]++; else out++; }
			printf("out of range: %d\n", out);
			for (int test = 0; test < 9; test ++) {
				std::string test_name;
				cudaMemcpy(gtab, tab, N*sizeof(double), cudaMemcpyHostToDevice);
				for (int rep=0; rep<1; rep++) {
					for (int i=0; i<M; i++) hist[i] = 0;
					cudaMemcpy(ghist, hist, M*sizeof(double), cudaMemcpyHostToDevice);
					cudaEventRecord(start);
					switch (test) {
						case 0: test_name = "    atomicAdd"; DirectCall< direct_sum > <<<blx,thx>>>(gtab, N, ghist); break;
						case 1: test_name = "for atomicAdd"; ForCall< direct_sum > <<<blx,thx>>>(gtab, N, ghist); break;
						case 2: test_name = "for atomicSum"; ForCall< block_sum > <<<blx,thx>>>(gtab, N, ghist); break;
						case 3: test_name = "for atomicSumWarp"; ForCall< warp_sum > <<<blx,thx>>>(gtab, N, ghist); break;
						case 4: test_name = "for atomicSumWarp (__any)"; ForCall< warp_any_sum > <<<blx,thx>>>(gtab, N, ghist); break;
						case 5: test_name = "for atomicSumWarpX (__any)"; ForCall< warp_xor_any_sum > <<<blx,thx>>>(gtab, N, ghist); break;
						case 6: test_name = "for atomicSum (cg)"; ForCall< cg_block_sum > <<<blx,thx>>>(gtab, N, ghist); break;
						case 7: test_name = "for atomicSum (cg)"; ForCall< cg_block_part_lab_sum > <<<blx,thx>>>(gtab, N, ghist); break;
						case 8: test_name = "    atomicSum (cg)"; DirectCall< cg_block_part_lab_sum > <<<blx,thx>>>(gtab, N, ghist); break;
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
