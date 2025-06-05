#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

#include <stdexcept>

#include <mma.h>
#include <cuda_fp16.h>


#define N_ITERS 8



inline __device__ __host__ uint32_t div_round_up(uint32_t a, uint32_t b) {
    return (a+b-1)/b;
}




template <int act_mode, typename fragment_t>
__host__ __device__ void warp_activation(fragment_t& frag) {
    switch (act_mode) {
        case 0:
            #pragma unroll
            for (int t=0; t < frag.num_elements; t++) {
                frag.x[t] = frag.x[t]>(__half)0.0f ? frag.x[t] : (__half)0.0f;
            }
            return;
        case 1:
            #pragma unroll
            for (int t=0; t < frag.num_elements; t++) {
                frag.x[t] = frag.x[t]*(__half)(frag.x[t]>(__half)0.0f ? 1.0f : 0.01f);
            }
            return;
        case 2:
            #pragma unroll
            for (int t=0; t < frag.num_elements; t++) {
                frag.x[t] = frag.x[t]/((__half)1.0f+hexp(-frag.x[t]));
            }
            return;
        default:
            return;
    } 
}


template<int WIDTH, int act_mode>
__device__ void linear_fst(
    int W1, 
    const __half* __restrict__ inputs, 
    const __half* __restrict__ params,
    __half* __restrict__ act_shmem
) {
    constexpr uint32_t N_BLOCKS = WIDTH / 16;
    uint32_t BIAS = WIDTH*N_ITERS*16;
    using namespace nvcuda;
    
    // Indices
    const uint32_t li = threadIdx.x; // index in warp ("lane index")
    const uint32_t wi = threadIdx.y; // index in block ("warp index")

    const uint32_t bias_offset = 8*li;
    const uint32_t lane_offset = bias_offset % 16;
    const uint32_t row = bias_offset / 16;

    const uint32_t weights_col = 16 * wi;
    
    
    
    // load bias to shared memory
    if (wi==0 && li < WIDTH/8) {
        *(int4*)&act_shmem[BIAS+bias_offset] = *(int4*)&params[bias_offset];
    }
    
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> result_frag[N_ITERS];
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> bias_frag;
    
    // load bias to register
    __syncthreads();
    wmma::load_matrix_sync(bias_frag,act_shmem+BIAS+weights_col,0,wmma::mem_row_major);
    
    params += WIDTH;
    

    // for each WIDTH block of W1
    #pragma unroll
    for (int c=0; c<W1; c+=16) {
        // load inputs to shared memory
        #pragma unroll
        for (int idx=wi; idx<N_ITERS; idx+=N_BLOCKS) {
            *(int4*)&act_shmem[lane_offset+(row+16*idx)*16]=*(int4*)&inputs[c+lane_offset+(row+16*idx)*W1];
        }
        
        __syncthreads();
        // load matrix to registers
        wmma::load_matrix_sync(weights_frag,params+c+weights_col*W1,W1);
        
        #pragma unroll
        for (int l=0; l<N_ITERS; l++) {
            // load inputs to shared memory
            wmma::load_matrix_sync(act_frag, act_shmem+(16*l)*16,16);
            // matmul
            if (c==0) {
                wmma::mma_sync(result_frag[l], act_frag, weights_frag, bias_frag);
            } else {
                wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
            }

            // apply activation
            if (c+16>=W1) {
                warp_activation<act_mode>(result_frag[l]);
            }
        }
        __syncthreads();    
    }

    // store outputs into shared memory
    #pragma unroll
    for (int l = 0; l<N_ITERS; l++) {
        wmma::store_matrix_sync(act_shmem+weights_col+l*16*WIDTH, result_frag[l], WIDTH,wmma::mem_row_major);
    }
}



template<int WIDTH, int act_mode>
__device__ void linear( 
    const __half* __restrict__ params,
    __half* __restrict__ act_shmem
) {
    constexpr uint32_t N_BLOCKS = WIDTH / 16;
    constexpr uint32_t BIAS = WIDTH*N_ITERS*16;
    using namespace nvcuda;
    
    // Indices
    const uint32_t li = threadIdx.x; // index in warp ("lane index")
    const uint32_t wi = threadIdx.y; // index in block ("warp index")

    const uint32_t lane_offset = (8 * li) % WIDTH;
    const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

    const uint32_t weights_col = 16 * wi;
    
    
    
    // load bias to shared memory
    if (wi==0 && li < WIDTH/8) {
        *(int4*)&act_shmem[lane_offset+BIAS] = *(int4*)&params[lane_offset];
    }
    
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag[N_BLOCKS];
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> result_frag[N_ITERS];
    //wmma::fragment<wmma::accumulator, 16, 16, 16, __half> bias_frag;
    
    // load bias to register
    __syncthreads();
    wmma::load_matrix_sync(result_frag[N_ITERS-1],act_shmem+BIAS+weights_col,0,wmma::mem_row_major);
    
    params += WIDTH;
    
    
    
    
    // load matrix to registers
    for (int i=0; i<N_BLOCKS; i++) {
        wmma::load_matrix_sync(weights_frag[i],params+(16*i)+weights_col*WIDTH,WIDTH);
    }
        
        
    #pragma unroll
    for (int l=0; l<N_ITERS; l++) {
        // load inputs to shared memory
        wmma::load_matrix_sync(act_frag, act_shmem+(16*l)*WIDTH,WIDTH);
        // matmul
        wmma::mma_sync(result_frag[l], act_frag, weights_frag[0], result_frag[N_ITERS-1]);

        #pragma unroll
        for (int i=1; i<N_BLOCKS; i++) {
            // load inputs to shared memory
            wmma::load_matrix_sync(act_frag, act_shmem+16*i+(16*l)*WIDTH,WIDTH);
            // matmul
            wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
        }

        // apply activation
        warp_activation<act_mode>(result_frag[l]);
        
    }
    __syncthreads();

    // store outputs into shared memory
    #pragma unroll
    for (int l = 0; l<N_ITERS; l++) {
        wmma::store_matrix_sync(act_shmem+weights_col+l*16*WIDTH, result_frag[l], WIDTH,wmma::mem_row_major);
    }
}


template<int WIDTH>
__device__ void linear_lst( 
    int W2,
    __half* __restrict__ outputs,
    const __half* __restrict__ params,
    __half* __restrict__ act_shmem
) {
    constexpr uint32_t N_BLOCKS = WIDTH / 16;
    constexpr uint32_t BIAS = WIDTH*N_ITERS*16;
    using namespace nvcuda;
    
    // Indices
    const uint32_t li = threadIdx.x; // index in warp ("lane index")
    const uint32_t wi = threadIdx.y; // index in block ("warp index")

    const uint32_t lane_offset = li*8;
    const uint32_t weights_row = lane_offset % WIDTH;
	const uint32_t weights_col = (lane_offset + 8 * 32 * wi) / WIDTH;
    

    // load bias into shared memory
    if (wi==0 && li < W2/8) {
        *(int4*)&act_shmem[BIAS+lane_offset] = *(int4*)&params[lane_offset];
    }

    __half* __restrict__ weights_shmem = act_shmem + BIAS + W2;
    params += W2;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag[N_BLOCKS];
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> result_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> bias_frag;
    
    #pragma unroll
    for (int c=0; c<W2; c+=16) {
        // load weight into shared memory
	    *(int4*)&weights_shmem[weights_row+weights_col*WIDTH] 
                = *(int4*)&params[c*WIDTH+weights_row+weights_col*WIDTH];
        __syncthreads();

        // load bias to register
        wmma::load_matrix_sync(bias_frag,act_shmem+BIAS+c,0,wmma::mem_row_major);
        
        // load weights to register
        #pragma unroll
        for (uint32_t i = 0; i < N_BLOCKS; i++) {
            wmma::load_matrix_sync(weights_frag[i], weights_shmem+16*i, WIDTH);
        }

        #pragma unroll
        for (int idx=wi; idx<N_ITERS; idx+=N_BLOCKS) {
            // load activation
            wmma::load_matrix_sync(act_frag, act_shmem + 16*idx*WIDTH, WIDTH);
            wmma::mma_sync(result_frag, act_frag, weights_frag[0], bias_frag);

            #pragma unroll
            for (uint32_t i = 1; i < N_BLOCKS; i++) {
                // load activation
                wmma::load_matrix_sync(act_frag, act_shmem + 16*i + 16*idx*WIDTH, WIDTH);
                wmma::mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
            }

            // store result to global memory
            wmma::store_matrix_sync(outputs+c+idx*16*W2, result_frag, W2, wmma::mem_row_major);
        }   
    }
}


template<int WIDTH, int act_mode>
__global__ void mlp_fused_kernel(
    int W1, int W2, int DEPTH,
    const __half* __restrict__ inputs,
    __half* __restrict__ outputs,
    const __half* __restrict__ params
) {
    extern __shared__ __half shmem[];
    __half* act_shmem = shmem;

    // Indices
    const uint32_t bi = blockIdx.x*16*N_ITERS;
    inputs += bi*W1;
    outputs += bi*W2;
    
    linear_fst<WIDTH,act_mode>(W1,inputs,params,act_shmem);
    params += (W1+1)*WIDTH;

    constexpr int stride=(WIDTH+1)*WIDTH;
    
    for (int i=0; i<DEPTH; i++) {
        linear<WIDTH,act_mode>(params+i*stride,act_shmem);
    }
    params += DEPTH*stride;

    linear_lst<WIDTH>(W2,outputs,params,act_shmem);

}
                



void mlp_fused32(
    int DEPTH, int act_mode,
    at::Tensor inputs, // BxW1
    at::Tensor outputs, // BxW2
    at::Tensor params
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const uint32_t batch_size = inputs.size(0);
    const uint32_t W1 = inputs.size(1);
    const uint32_t W2 = outputs.size(1);
    constexpr uint32_t WIDTH = 32;
    constexpr uint32_t N_BLOCK_ROWS = WIDTH / 16;

    if (batch_size % (16 * N_ITERS) != 0) {
        throw std::runtime_error("Batch size must be a multiple of 128\n");
    }
    
    const dim3 threads = { 32u, N_BLOCK_ROWS, 1 };
    constexpr uint32_t n_elems_per_block = 16 * N_ITERS;
    uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);
    const dim3 blocks = { n_blocks, 1u, 1u };

    size_t shmem_size = sizeof(__half)*((16*N_ITERS)*WIDTH+W2+16*WIDTH);
    
    switch (act_mode) {
        case 0:
            mlp_fused_kernel<32,0><<<blocks,threads,shmem_size,stream>>>(
                W1,W2,DEPTH,
                reinterpret_cast<__half*>(inputs.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(outputs.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(params.data_ptr<at::Half>())
            );
            break;
        case 1:
            mlp_fused_kernel<32,1><<<blocks,threads,shmem_size,stream>>>(
                W1,W2,DEPTH,
                reinterpret_cast<__half*>(inputs.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(outputs.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(params.data_ptr<at::Half>())
            );
            break;
        case 2:
            mlp_fused_kernel<32,2><<<blocks,threads,shmem_size,stream>>>(
                W1,W2,DEPTH,
                reinterpret_cast<__half*>(inputs.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(outputs.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(params.data_ptr<at::Half>())
            );
            break;
        default:
            break;
    }
    cudaDeviceSynchronize();
}


void mlp_fused16(
    int DEPTH, int act_mode,
    at::Tensor inputs, // BxW1
    at::Tensor outputs, // BxW2
    at::Tensor params
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const uint32_t batch_size = inputs.size(0);
    const uint32_t W1 = inputs.size(1);
    const uint32_t W2 = outputs.size(1);
    constexpr uint32_t WIDTH = 16;
    constexpr uint32_t N_BLOCK_ROWS = WIDTH / 16;

    if (batch_size % (16 * N_ITERS) != 0) {
        throw std::runtime_error("Batch size must be a multiple of 128\n");
    }
    
    const dim3 threads = { 32u, N_BLOCK_ROWS, 1 };
    constexpr uint32_t n_elems_per_block = 16 * N_ITERS;
    uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);
    const dim3 blocks = { n_blocks, 1u, 1u };

    size_t shmem_size = sizeof(__half)*((16*N_ITERS)*WIDTH+W2+16*WIDTH);
    
    switch (act_mode) {
        case 0:
            mlp_fused_kernel<16,0><<<blocks,threads,shmem_size,stream>>>(
                W1,W2,DEPTH,
                reinterpret_cast<__half*>(inputs.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(outputs.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(params.data_ptr<at::Half>())
            );
            break;
        case 1:
            mlp_fused_kernel<16,1><<<blocks,threads,shmem_size,stream>>>(
                W1,W2,DEPTH,
                reinterpret_cast<__half*>(inputs.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(outputs.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(params.data_ptr<at::Half>())
            );
            break;
        case 2:
            mlp_fused_kernel<16,2><<<blocks,threads,shmem_size,stream>>>(
                W1,W2,DEPTH,
                reinterpret_cast<__half*>(inputs.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(outputs.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(params.data_ptr<at::Half>())
            );
            break;
        default:
            break;
    }
    cudaDeviceSynchronize();
}
