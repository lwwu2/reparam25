#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

#include <stdexcept>

#include <mma.h>
#include <cuda_fp16.h>


#define N_ITERS 8
#define WIDTH 16



inline __device__ __host__ uint32_t div_round_up(uint32_t a, uint32_t b) {
    return (a+b-1)/b;
}




template <typename fragment_t>
__host__ __device__ void warp_activation(fragment_t& dx,fragment_t& dy,fragment_t& frag) {
            #pragma unroll
            for (int t=0; t < frag.num_elements; t++) {
                __half sig = (__half)1.0f/((__half)1.0f+hexp(-frag.x[t]));
                frag.x[t] = frag.x[t]*sig;
                __half dsig = frag.x[t]*((__half)1.0f-sig) + sig;
                dx.x[t] = dx.x[t]*dsig;
                dy.x[t] = dy.x[t]*dsig;
            }
            return; 
}

template <typename fragment_t>
__host__ __device__ void warp_activation(fragment_t& dx,fragment_t& dy,fragment_t& src_dx,fragment_t& src_dy,fragment_t& frag) {
            #pragma unroll
            for (int t=0; t < frag.num_elements; t++) {
                __half sig = (__half)1.0f/((__half)1.0f+hexp(-frag.x[t]));
                frag.x[t] = frag.x[t]*sig;
                __half dsig = frag.x[t]*((__half)1.0f-sig) + sig;
                dx.x[t] = src_dx.x[t]*dsig;
                dy.x[t] = src_dy.x[t]*dsig;
            }
            return;
}



__device__ void linear16_fst(
    int W1, 
    const __half* __restrict__ inputs, 
    const __half* __restrict__ params,
    __half* __restrict__ act_shmem
) {
    constexpr uint32_t BIAS = WIDTH*N_ITERS*16;
    const uint32_t N_BLOCKS = W1/WIDTH;
    using namespace nvcuda;
    
    // Indices
    const uint32_t li = threadIdx.x; // index in warp ("lane index")
    const uint32_t wi = threadIdx.y; // index in block ("warp index")

    const uint32_t bias_offset = 8*li;
    //const uint32_t lane_offset = bias_offset % 16;
    //const uint32_t row = bias_offset / 16;

    const uint32_t weights_col = 16 * wi;
    
    // load dx, dy, bias to shared memory
    if (wi==0 && li <(WIDTH*3)/8) {
        *(int4*)&act_shmem[BIAS+bias_offset] = *(int4*)&params[bias_offset];
    }
    
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag[3];//know W1<16x3
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> result_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> bias_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> dx_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> dy_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> dx0_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> dy0_frag;
    
    // load dx,dy,bias to register
    __syncthreads();
    wmma::load_matrix_sync(dx0_frag,act_shmem+BIAS+weights_col,0,wmma::mem_row_major);
    wmma::load_matrix_sync(dy0_frag,act_shmem+BIAS+WIDTH+weights_col,0,wmma::mem_row_major);
    wmma::load_matrix_sync(bias_frag,act_shmem+BIAS+WIDTH*2+weights_col,0,wmma::mem_row_major);

    params += WIDTH*3;

    // load weight to registers
    #pragma unroll
    for (int i=0; i<N_BLOCKS; i++) {
        wmma::load_matrix_sync(weights_frag[i],params+i*16,W1);
    }

    
    #pragma unroll
    for (int l=0; l<N_ITERS; l++) {
        // load inputs directly
        wmma::load_matrix_sync(act_frag,inputs+(16*l)*W1,W1);
        wmma::mma_sync(result_frag,act_frag,weights_frag[0],bias_frag);

        #pragma unroll
        for (int i=1; i <N_BLOCKS; i++) {
            wmma::load_matrix_sync(act_frag,inputs+i*16+(16*l)*W1,W1);
            wmma::mma_sync(result_frag,act_frag,weights_frag[i],result_frag);
        }

        warp_activation(dx_frag,dy_frag,dx0_frag,dy0_frag,result_frag);

        wmma::store_matrix_sync(act_shmem+weights_col+l*16*WIDTH, result_frag, WIDTH,wmma::mem_row_major);
        wmma::store_matrix_sync(act_shmem+BIAS+weights_col+l*16*WIDTH, dx_frag, WIDTH,wmma::mem_row_major);
        wmma::store_matrix_sync(act_shmem+BIAS*2+weights_col+l*16*WIDTH, dy_frag, WIDTH,wmma::mem_row_major);
    }
}



__device__ void linear16( 
    const __half* __restrict__ params,
    __half* __restrict__ act_shmem
) {
    //constexpr uint32_t N_BLOCKS = WIDTH / 16;
    constexpr uint32_t BIAS = WIDTH*N_ITERS*16;
    using namespace nvcuda;
    
    // Indices
    const uint32_t li = threadIdx.x; // index in warp ("lane index")
    const uint32_t wi = threadIdx.y; // index in block ("warp index")

    const uint32_t lane_offset = (8 * li) % WIDTH;
    //const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

    const uint32_t weights_col = 16 * wi;
    
    
    
    // load bias to shared memory
    if (wi==0 && li < WIDTH/8) {
        *(int4*)&act_shmem[lane_offset+BIAS*3] = *(int4*)&params[lane_offset];
    }
    
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> result_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> bias_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> dx_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> dy_frag;
    
    // load bias to register
    __syncthreads();
    wmma::load_matrix_sync(bias_frag,act_shmem+BIAS*3+weights_col,0,wmma::mem_row_major);
    
    params += WIDTH;
    
    // load matrix to registers
    wmma::load_matrix_sync(weights_frag,params+weights_col*WIDTH,WIDTH);

        
        
    #pragma unroll
    for (int l=0; l<N_ITERS; l++) {
        // fill dx dy
        wmma::fill_fragment(dx_frag, (__half)0.0f);
        wmma::fill_fragment(dy_frag, (__half)0.0f);

        // load inputs to shared memory
        wmma::load_matrix_sync(act_frag, act_shmem+(16*l)*WIDTH,WIDTH);
        // matmul
        wmma::mma_sync(result_frag, act_frag, weights_frag, bias_frag);

        wmma::load_matrix_sync(act_frag, act_shmem+BIAS+(16*l)*WIDTH,WIDTH);
        wmma::mma_sync(dx_frag, act_frag, weights_frag,dx_frag);
        wmma::load_matrix_sync(act_frag, act_shmem+BIAS*2+(16*l)*WIDTH,WIDTH);
        wmma::mma_sync(dy_frag, act_frag, weights_frag,dy_frag);

        // apply activation
        warp_activation(dx_frag,dy_frag,result_frag);

        wmma::store_matrix_sync(act_shmem+weights_col+l*16*WIDTH, result_frag, WIDTH,wmma::mem_row_major);
        wmma::store_matrix_sync(act_shmem+BIAS+weights_col+l*16*WIDTH, dx_frag, WIDTH,wmma::mem_row_major);
        wmma::store_matrix_sync(act_shmem+BIAS*2+weights_col+l*16*WIDTH, dy_frag, WIDTH,wmma::mem_row_major);
        
    }
}



__device__ void linear16_lst(
    const __half* __restrict__ params,
    __half* __restrict__ act_shmem
) {
    // has 8 dim output
    constexpr uint32_t N_OUTPUTS = N_ITERS/2;
    constexpr uint32_t BIAS = WIDTH*N_ITERS*16;
    using namespace nvcuda;
    
    // Indices
    const uint32_t li = threadIdx.x; // index in warp ("lane index")
    const uint32_t wi = threadIdx.y; // index in block ("warp index")

    const uint32_t lane_offset = li*8;
    //const uint32_t weights_idx = lane_offset + 8 * 32 *wi;
    //const uint32_t weights_row = lane_offset % WIDTH;
	//const uint32_t weights_col = (lane_offset + 8 * 32 * wi) / WIDTH;
    

    // load bias and weight into shared memory
    //if (weights_idx < 8*17) {
    //    *(int4*)&act_shmem[BIAS*3+weights_idx] = *(int4*)&params[weights_idx];
    //}
    if (wi==0 && li<1) {
         *(int4*)&act_shmem[BIAS*3+lane_offset] = *(int4*)&params[lane_offset];
    }
    params += 8;
    
    wmma::fragment<wmma::matrix_a, 32, 8, 16, __half, wmma::row_major> act_frag;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, __half, wmma::col_major> weights_frag;
    // 8*16->4*32
    wmma::fragment<wmma::accumulator, 32, 8, 16, __half> result_frag;
    wmma::fragment<wmma::accumulator, 32, 8, 16, __half> bias_frag;
    wmma::fragment<wmma::accumulator, 32, 8, 16, __half> dx_frag;
    wmma::fragment<wmma::accumulator, 32, 8, 16, __half> dy_frag;


    __syncthreads();
    // load bias into register
    wmma::load_matrix_sync(bias_frag,act_shmem+BIAS*3,0,wmma::mem_row_major);
    // load weight into register 
    //wmma::load_matrix_sync(weights_frag, act_shmem+BIAS*3+8,WIDTH);
    wmma::load_matrix_sync(weights_frag, params, WIDTH);


    #pragma unroll
    for (int l=0; l<N_OUTPUTS; l++) {
        // load act into register
        wmma::fill_fragment(dx_frag,(__half)0.0f);
        wmma::fill_fragment(dy_frag,(__half)0.0f);

        wmma::load_matrix_sync(act_frag, act_shmem + 32*l*WIDTH, WIDTH);
        wmma::mma_sync(result_frag, act_frag, weights_frag, bias_frag);

        wmma::load_matrix_sync(act_frag, act_shmem + BIAS+32*l*WIDTH, WIDTH);
        wmma::mma_sync(dx_frag, act_frag, weights_frag, dx_frag);

        wmma::load_matrix_sync(act_frag, act_shmem + BIAS*2+32*l*WIDTH, WIDTH);
        wmma::mma_sync(dy_frag, act_frag, weights_frag, dy_frag);

        // store result to global memory
        //wmma::store_matrix_sync(outputs+l*32*8, result_frag, 8, wmma::mem_row_major);
        // store intermediate to shared memory
        wmma::store_matrix_sync(act_shmem+l*32*8, result_frag, 8, wmma::mem_row_major);
        wmma::store_matrix_sync(act_shmem+BIAS+l*32*8, dx_frag, 8, wmma::mem_row_major);
        wmma::store_matrix_sync(act_shmem+BIAS*2+l*32*8, dy_frag, 8, wmma::mem_row_major);
    }
}

__global__ void mlp_reparam16_kernel(
    int W1,
    const __half* __restrict__ inputs,
    __half* __restrict__ outputs,
    const __half* __restrict__ params
) {
    extern __shared__ __half shmem[];
    __half* act_shmem = shmem;

    // Indices
    const uint32_t bi = blockIdx.x*16*N_ITERS;
    inputs += bi*W1;
    outputs += bi*4;
    
    linear16_fst(W1,inputs,params,act_shmem);
    params += (W1+3)*WIDTH;

    constexpr int stride=(WIDTH+1)*WIDTH;
    for (int i=0; i<2; i++) {
        linear16(params+i*stride,act_shmem);
    }
    params += 2*stride;

    linear16_lst(params,act_shmem);

    // handle the last activation
    // Bx8 primal, Bx8 dx, Bx8 dy
    __syncthreads();

    // Indices
    constexpr uint32_t BIAS = WIDTH*N_ITERS*16;
    const uint32_t row = threadIdx.x + threadIdx.y*32;
    
    __half x[4];
    __half dx[4];
    __half dy[4];

    #pragma unroll
    for (int l=0; l<(N_ITERS/2); l++) {
        // load result to registers
        *(int2*)x = *(int2*)&act_shmem[(row+l*32)*8];
        *(int2*)dx = *(int2*)&act_shmem[BIAS+(row+l*32)*8];
        *(int2*)dy = *(int2*)&act_shmem[BIAS*2+(row+l*32)*8];
        
        
        // apply normalization
        __half r = hexp(-x[2]);
        x[2] = hlog((__half)1.0f+(__half)1.0f/r);
        r = (__half)1.0f/((__half)1.0f+r);
        dx[2] *= r;
        dy[2] *= r;
        
        __half r2 = __hmax(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],(__half)1e-14f);
        x[3] = hsqrt(r2);
        x[0] /= x[3];
        x[1] /= x[3];
        x[2] /= x[3];

        // compute jacobian determinant
        r = x[2]*(
            x[2]*(dx[0]*dy[1]-dx[1]*dy[0])
          + x[1]*(dx[2]*dy[0]-dx[0]*dy[2])
          + x[0]*(dx[1]*dy[2]-dx[2]*dy[1])
        )/r2;
        x[3] = __habs(r);
        
        // store back to outputs
        *(int2*)&outputs[(row+l*32)*4] = *(int2*)x;
    }
}

void mlp_reparam16(
    at::Tensor inputs, // BxW1
    at::Tensor outputs, // Bx4
    at::Tensor params
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const uint32_t batch_size = inputs.size(0);
    const uint32_t W1 = inputs.size(1);
    constexpr uint32_t N_BLOCK_ROWS = WIDTH / 16;

    if (batch_size % (16 * N_ITERS) != 0) {
        throw std::runtime_error("Batch size must be a multiple of 128\n");
    }
    
    const dim3 threads = { 32u, N_BLOCK_ROWS, 1 };
    constexpr uint32_t n_elems_per_block = 16 * N_ITERS;
    uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);
    const dim3 blocks = { n_blocks, 1u, 1u };

    size_t shmem_size = sizeof(__half)*((16*N_ITERS)*WIDTH*3+8+16*WIDTH);
    
    mlp_reparam16_kernel<<<blocks,threads,shmem_size,stream>>>(
        W1,
        reinterpret_cast<__half*>(inputs.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(outputs.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(params.data_ptr<at::Half>())
    );
    cudaDeviceSynchronize();
}