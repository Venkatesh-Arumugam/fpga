/******************************************************************************
 * VERSION 1: MEMORY-OPTIMIZED WITH LOCAL BUFFERING (FIXED)
 * File: v1_dct_accel_fixed.cpp
 * Description: Add local buffers to improve memory access patterns
 * Expected: 2-3x speedup, efficient memory bandwidth usage
 ******************************************************************************/

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_uint<8>  pixel_t;
typedef ap_int<16>  coeff_t;
typedef ap_fixed<24,12> dct_t;

static const dct_t C[8][8] = {
    {0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553},
    {0.490393, 0.415735, 0.277785, 0.097545,-0.097545,-0.277785,-0.415735,-0.490393},
    {0.461940, 0.191342,-0.191342,-0.461940,-0.461940,-0.191342, 0.191342, 0.461940},
    {0.415735,-0.097545,-0.490393,-0.277785, 0.277785, 0.490393, 0.097545,-0.415735},
    {0.353553,-0.353553,-0.353553, 0.353553, 0.353553,-0.353553,-0.353553, 0.353553},
    {0.277785,-0.490393, 0.097545, 0.415735,-0.415735,-0.097545, 0.490393,-0.277785},
    {0.191342,-0.461940, 0.461940,-0.191342,-0.191342, 0.461940,-0.461940, 0.191342},
    {0.097545,-0.277785, 0.415735,-0.490393, 0.490393,-0.415735, 0.277785,-0.097545}
};

static void dct_2d(pixel_t in[8][8], coeff_t out[8][8])
{
#pragma HLS INLINE off
    dct_t tmp[8][8];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=0

    for (int u = 0; u < 8; u++) {
#pragma HLS UNROLL
        for (int v = 0; v < 8; v++) {
#pragma HLS UNROLL
            dct_t acc = 0;
            for (int x = 0; x < 8; x++) {
#pragma HLS UNROLL
                acc += C[u][x] * (dct_t)((int)in[x][v] - 128);
            }
            tmp[u][v] = acc;
        }
    }

    for (int u = 0; u < 8; u++) {
#pragma HLS UNROLL
        for (int v = 0; v < 8; v++) {
#pragma HLS UNROLL
            dct_t acc = 0;
            for (int y = 0; y < 8; y++) {
#pragma HLS UNROLL
                acc += tmp[u][y] * C[v][y];
            }
            int val = (int)hls::round(acc);
            if (val < -32768) val = -32768;
            if (val >  32767) val =  32767;
            out[u][v] = (coeff_t)val;
        }
    }
}

extern "C" void dct_accel(
    const pixel_t* inR,
    const pixel_t* inG,
    const pixel_t* inB,
    coeff_t* outR,
    coeff_t* outG,
    coeff_t* outB,
    int width,
    int height
) {
#pragma HLS INTERFACE m_axi port=inR offset=slave bundle=gmem0 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=inG offset=slave bundle=gmem1 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=inB offset=slave bundle=gmem2 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=outR offset=slave bundle=gmem3 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=outG offset=slave bundle=gmem4 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=outB offset=slave bundle=gmem5 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=width
#pragma HLS INTERFACE s_axilite port=height
#pragma HLS INTERFACE s_axilite port=return

    // Local buffers for burst reads/writes
    pixel_t local_buf_R[64];
    pixel_t local_buf_G[64];
    pixel_t local_buf_B[64];
    coeff_t local_coef_R[64];
    coeff_t local_coef_G[64];
    coeff_t local_coef_B[64];

#pragma HLS ARRAY_PARTITION variable=local_buf_R cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=local_buf_G cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=local_buf_B cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=local_coef_R cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=local_coef_G cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=local_coef_B cyclic factor=8

    for (int by = 0; by < height; by += 8) {
        for (int bx = 0; bx < width; bx += 8) {

            // Burst read into local buffer
            read_loop: for (int i = 0; i < 64; i++) {
#pragma HLS PIPELINE II=1
                int y = i / 8;
                int x = i % 8;
                int gx = bx + x;
                int gy = by + y;
                if (gx < width && gy < height) {
                    int idx = gy * width + gx;
                    local_buf_R[i] = inR[idx];
                    local_buf_G[i] = inG[idx];
                    local_buf_B[i] = inB[idx];
                } else {
                    local_buf_R[i] = (pixel_t)0;
                    local_buf_G[i] = (pixel_t)0;
                    local_buf_B[i] = (pixel_t)0;
                }
            }

            // Reshape to 8x8 and compute DCT
            pixel_t blk_R[8][8], blk_G[8][8], blk_B[8][8];
            coeff_t coef_R[8][8], coef_G[8][8], coef_B[8][8];

#pragma HLS ARRAY_PARTITION variable=blk_R complete dim=0
#pragma HLS ARRAY_PARTITION variable=blk_G complete dim=0
#pragma HLS ARRAY_PARTITION variable=blk_B complete dim=0
#pragma HLS ARRAY_PARTITION variable=coef_R complete dim=0
#pragma HLS ARRAY_PARTITION variable=coef_G complete dim=0
#pragma HLS ARRAY_PARTITION variable=coef_B complete dim=0

            // Reshape from 1D buffer to 2D block
            reshape_to_2d: for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int idx = y * 8 + x;
                    blk_R[y][x] = local_buf_R[idx];
                    blk_G[y][x] = local_buf_G[idx];
                    blk_B[y][x] = local_buf_B[idx];
                }
            }

            // Compute DCT for all channels
            dct_2d(blk_R, coef_R);
            dct_2d(blk_G, coef_G);
            dct_2d(blk_B, coef_B);

            // Flatten back to 1D for burst write
            reshape_to_1d: for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int idx = y * 8 + x;
                    local_coef_R[idx] = coef_R[y][x];
                    local_coef_G[idx] = coef_G[y][x];
                    local_coef_B[idx] = coef_B[y][x];
                }
            }

            // Burst write from local buffer
            write_loop: for (int i = 0; i < 64; i++) {
#pragma HLS PIPELINE II=1
                int y = i / 8;
                int x = i % 8;
                int gx = bx + x;
                int gy = by + y;
                if (gx < width && gy < height) {
                    int idx = gy * width + gx;
                    outR[idx] = local_coef_R[i];
                    outG[idx] = local_coef_G[i];
                    outB[idx] = local_coef_B[i];
                }
            }
        }
    }
}
