#include "ggml.h"
#include "common.cuh"
#include "mmvmxfp4.cuh"

typedef union {
    half f16;
    uint16_t  u16;
} f16_t;

// Kernel - 
// <<<
//      block_nums: x * y * z == number of blocks launched
//          x: nrows == src0->ne[1]
//          y: nchannels_dst == dst->ne[1] with ids or dst->ne[2] without ids
//          z: nsamples_dst == dst->ne[3]
//      block_dims: x * y * z == number of threads per block (e.g., 32)
//          x: block_size_best
//          y: 1
//          z: 1
//      smem = (warp_size*sizeof(float)) -- probably doesn't need to change
//      stream
// >>>
template <typename type_acc, int block_size>
static __global__ void mul_mat_vec_mxfp4(
        const block_mxfp4 * __restrict__ x, const float * __restrict__ y, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int64_t ncols2, const int64_t nchannels_y, const int64_t stride_row,
        const int64_t channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int64_t sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst) {
    const int64_t row         = blockIdx.x;
    const int64_t channel_dst = blockIdx.y;
    const int64_t channel_x   = ids ? ids[channel_dst]          : channel_dst / channel_ratio;
    const int64_t channel_y   = ids ? channel_dst % nchannels_y : channel_dst;
    const int64_t sample_dst  = blockIdx.z;
    const int64_t sample_x    = sample_dst / sample_ratio;
    const int64_t sample_y    = sample_dst;
    const int     tid         = threadIdx.x;
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();

    const uint16_t dst_bias = 15;
    const uint16_t dst_0p5 = 0x3800;
    const uint16_t dst_m_bits = 10;

    // TODO something in this math has to be adjusted for block size so x jumps by blocks
    // but y jumps by 32*<thing> to keep things aligned
    //
    // stride_row winds up being src0->ne[0] // Seems like the most likely place to adjust
    // stride_channel = src0->nb[2] / ggml_type_size(src0->type)
    // stride_sample_x = src0->nb[3] / ts_src0
    // Straw-man
    //  - each of the stride values become block based (32 items)
    //  - each of the multiples in front of the block are *32
    //
    //
    // block_size ranges from 32 - 256
    if (tid == 0) {
        printf("%s [%u:%u:%u] [%d] x+=%lld y+=%lld == sample_x:%lld stride_sample_x:%lld channel_x:%lld stride_channel_x:%lld row:%lld stride_row:%lld ncols2:%lld\n", __func__,
            blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
            sample_x  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row,
            sample_y  *stride_sample_y   + channel_y  *stride_channel_y,
            sample_x  ,stride_sample_x  , channel_x  ,stride_channel_x   , row, stride_row,
            ncols2
        );
    }

    x   += sample_x  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    y   += sample_y  *stride_sample_y   + channel_y  *stride_channel_y;
    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
    
    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[]; // allocated in GPU shared memory: warp_size*sizeof(float)
    float * buf_iw = (float *) data_mmv;

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf = 0.0f;

    // Derived from float case
    // TODO how to translate this to the mxfp4 pattern?
    // x2: struct with 8 byte alignment for .x and .y
    //  - doesn't this mean we skip every other float, or was the data loaded aligned already?
    // const float2 * x2 = (const float2 *) x->qs; // WRONG but just to get it compiling...

    // tid ranges from 0 - 31
    // block_size ranges from 32 - 256
    // ncols2 = ncols / 2
    //
    // Each thread grabs 2 items then jumps by the block size
    // e.g.
    // mx  b0      b1   b2  
    // t0: [0] 0,1      [32] 64,65 ...
    // t1: [1] 2,3      [33] 66,67 ...
    // t2: ...
    // t31: [31] 62,63  [63] 128,129 ...
    //
    for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
        int offset0 = col2 / (MXFP4/2);
        int i = col2 % (MXFP4/2);
        const block_mxfp4 *x2 = x+offset0;

        // TODO there's probably a better way to do this via GPU ops...
        float scale = 0.0f;
        uint32_t d = (((uint32_t)x2->d) << 23);
        memcpy(&scale, &d, sizeof(float));
        uint16_t em0 = x2->qs[i] & 0x07;
        uint16_t em1 = x2->qs[i] & 0x70;
        // float16 values
        f16_t x0;
        f16_t x1;
        x0.u16 = (em0 << (dst_m_bits - 1)) | ((x2->qs[i] & 0x08) << 12);
        x1.u16 = (em1 << (dst_m_bits - 5)) | ((x2->qs[i] & 0x80) << 8);

        // Three cases:
        // x is normal and non-zero: Correct bias
        if ((em0 & 0x06) != 0) {
            x0.u16 = x0.u16 + ((dst_bias - 1) << dst_m_bits);
        }
        if ((em1 & 0x60) != 0) {
            x1.u16 = x1.u16 + ((dst_bias - 1) << dst_m_bits);
        }
        // x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in the dst type
        if (em0 == 0x01) {
            x0.u16 = dst_0p5 | (x0.u16 & 0x8000);
        }
        if (em1 == 0x10) {
            x1.u16 = dst_0p5 | (x1.u16 & 0x8000);
        }
        // x is zero, do nothing

        // TODO NaN handling?

        const float2 tmpx = {x0.f16, x1.f16};
        const float2 tmpy = y2[col2];
        sumf += tmpx.x*tmpy.x*scale;
        sumf += tmpx.y*tmpy.y*scale;

        // Spot check debug
        // if (tid == 0 || tid == 31) {
            printf("%s [%u:%u:%u] [%d] x0:%f x1:%f y0:%f y1:%f\n", __func__, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, scale*tmpx.x, scale*tmpx.y, tmpy.x, tmpy.y);
        // }
    }

    // f16 and bf16 cases
//     } else if constexpr (std::is_same<T, half>::value) {
//         const half2 * x2 = (const half2 *) x;

//         if (std::is_same<type_acc, float>::value) {
//             for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
//                 const float2 tmpx = __half22float2(x2[col2]);
//                 const float2 tmpy = y2[col2];
//                 sumf += tmpx.x * tmpy.x;
//                 sumf += tmpx.y * tmpy.y;
//             }
//         } else {
// #ifdef FP16_AVAILABLE
//             half2 sumh2 = make_half2(0.0f, 0.0f);

//             for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
//                 const float2 tmp = y2[col2];
//                 sumh2 += x2[col2] * make_half2(tmp.x, tmp.y);
//             }

//             sumf = __low2float(sumh2) + __high2float(sumh2);
// #else
//             NO_DEVICE_CODE;
// #endif // FP16_AVAILABLE
//         }
//     } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
//         const int * x2 = (const int *) x;
//         for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
//             const int    tmpx = x2[col2];
//             const float2 tmpy = y2[col2];
//             sumf += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]) * tmpy.x;
//             sumf += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]) * tmpy.y;
//         }
//     } else {
//         static_assert(std::is_same<T, void>::value, "unsupported type");
//     }

    sumf = warp_reduce_sum<warp_size>(sumf);

    if (block_size > warp_size) {
        buf_iw[tid/warp_size] = sumf;
        __syncthreads();
        if (tid >= warp_size) {
            return;
        }
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum<warp_size>(sumf);
    }

    if (tid != 0) {
        return;
    }

    dst[row] = sumf;
}

template <typename type_acc>
static void launch_mul_mat_vec_cuda_mxfp4(
        const block_mxfp4 * x, const float * y, const int32_t * ids, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        cudaStream_t stream) {
    GGML_ASSERT(ncols      % 2 == 0);
    // GGML_ASSERT(stride_row % 2 == 0); // TODO 
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(       nsamples_dst  % nsamples_x  == 0);
    const int64_t channel_ratio = nchannels_dst / nchannels_x;
    const int64_t sample_ratio  = nsamples_dst  / nsamples_x;
    int device;
    int warp_size;

    CUDA_CHECK(cudaGetDevice(&device));
    warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    const int smem = warp_size*sizeof(float);
    const dim3 block_nums(nrows, nchannels_dst, nsamples_dst);
    const dim3 block_dims(block_size_best, 1, 1);
    printf("%s launching [%u:%u:%u] [%d] ncols2:%lld nchannels_y:%lld stride_row:%lld channel_ratio:%lld stride_channel_x:%lld stride_channel_y:%lld stride_channel_dst:%lld sample_ratio:%lld stride_sample_x:%lld stride_sample_y:%lld stride_sample_dst:%lld\n", __func__,
        block_nums.x, block_nums.y, block_nums.z, block_dims.x,
        ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
        stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst
    );

    switch (block_size_best) {
        case   32: {
            mul_mat_vec_mxfp4<type_acc,  32><<<block_nums, block_dims, smem, stream>>>
                (x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case   64: {
            mul_mat_vec_mxfp4<type_acc,  64><<<block_nums, block_dims, smem, stream>>>
                (x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case   96: {
            mul_mat_vec_mxfp4<type_acc,  96><<<block_nums, block_dims, smem, stream>>>
                (x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  128: {
            mul_mat_vec_mxfp4<type_acc, 128><<<block_nums, block_dims, smem, stream>>>
                (x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  160: {
            mul_mat_vec_mxfp4<type_acc, 160><<<block_nums, block_dims, smem, stream>>>
                (x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  192: {
            mul_mat_vec_mxfp4<type_acc, 192><<<block_nums, block_dims, smem, stream>>>
                (x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  224: {
            mul_mat_vec_mxfp4<type_acc, 224><<<block_nums, block_dims, smem, stream>>>
                (x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case  256: {
            mul_mat_vec_mxfp4<type_acc, 256><<<block_nums, block_dims, smem, stream>>>
                (x, y, ids, dst, ncols/2, nchannels_y, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
                 stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

template<typename T> // TODO remove this template once it's functional
static void mul_mat_vec_cuda_mxfp4(
        const T * x, const float * y, const int32_t * ids, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        enum ggml_prec prec, cudaStream_t stream) {
    launch_mul_mat_vec_cuda_mxfp4<float>
        (x, y, ids, dst, ncols, nrows, stride_row, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
         stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
}

void ggml_cuda_mul_mat_vec_mxfp4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    // const size_t bs_src0 = ggml_blck_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    // printf("%s XXX called ts_src0:%lld nb00:%lld s01:%lld s02:%lld s03:%lld\n", __func__, ts_src0, nb00, src0->nb[1] / ts_src0, src0->nb[2] / ts_src0, src0->nb[3] / ts_src0);

    GGML_ASSERT(!ids || ne12 == 1); // Implementation is only correct for  batch size 1.
    GGML_ASSERT(ne13 == ne3);

    // GGML_ASSERT(        nb00       == ts_src0); // TODO adjust for block sizing logic
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    printf("%s src0 ne[%lld %lld %lld %lld] nb[%lld %lld %lld %lld] ts=%lld bs=%lld\n", __func__, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], ggml_type_size(src0->type), ggml_blck_size(src0->type));
    const int64_t stride_row = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t stride_channel_x = src0->nb[2] / ts_src0; // TODO adjust for block sizing logic
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    // const int64_t stride_sample_x = src0->nb[3] / ts_src0;
    const int64_t stride_sample_x = src0->nb[3] / ts_src0; // TODO adjust for block sizing logic
    const int64_t stride_sample_y = src1->nb[3] / ts_src1;
    const int64_t stride_sample_dst  =  dst->nb[3] / ts_dst;
    const int64_t nsamples_dst = ne3;
    const int64_t nsamples_x = ne03;
    const int64_t nchannels_x = ne02;
    const int64_t nrows = ne01;
    const int64_t ncols = ne00;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    GGML_ASSERT(ncols_dst == 1);

    const block_mxfp4 * src0_d = (const block_mxfp4 *) src0->data;
    mul_mat_vec_cuda_mxfp4(src0_d, src1_d, ids_d, dst_d, ncols, nrows, stride_row,
        nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
        nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, prec, ctx.stream());
/*static void mul_mat_vec_cuda_mxfp4(
const T * x, const float * y, const int32_t * ids, float * dst,
const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,

enum ggml_prec prec, cudaStream_t stream) {
*/
}

void ggml_cuda_op_mul_mat_vec_mxfp4(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {
    printf("%s src0 ne[%lld %lld %lld %lld] nb[%lld %lld %lld %lld] ts=%lld bs=%lld\n", __func__, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], ggml_type_size(src0->type), ggml_blck_size(src0->type));

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    GGML_ASSERT(src1_ncols == 1);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;


    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row         = ne00 / MXFP4; 
    const int64_t nchannels_x        = 1;
    const int64_t nchannels_y        = 1;
    const int64_t nchannels_dst      = 1;
    const int64_t stride_channel_x   = 0;
    const int64_t stride_channel_y   = 0;
    const int64_t stride_channel_dst = 0;
    const int64_t nsamples_x         = 1;
    const int64_t nsamples_dst       = 1;
    const int64_t stride_sample_x    = 0;
    const int64_t stride_sample_y    = 0;
    const int64_t stride_sample_dst  = 0;

    const block_mxfp4 * src0_d = (const block_mxfp4 *) src0->data;
    mul_mat_vec_cuda_mxfp4(src0_d, src1_ddf_i, nullptr, dst_dd_i, ne00, row_diff, stride_row,
        nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
        nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, prec, stream);

    GGML_UNUSED(ctx);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}
