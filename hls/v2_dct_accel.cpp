/******************************************************************************
 * VERSION 2: HARDCODED DIMENSIONS WITH COMPATIBLE INTERFACE
 * File: v2_dct_accel_fixed.cpp
 * Description: Fixed 1920x1080 with width/height params for host compatibility
 * Expected: 3-4x speedup, very high resource usage
 ******************************************************************************/

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_uint<8>  pixel_t;
typedef ap_int<16>  coeff_t;
typedef ap_fixed<24,12> dct_t;

#define WIDTH 1920
#define HEIGHT 1080

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
    int width,   // Accept but ignore - use hardcoded WIDTH
    int height   // Accept but ignore - use hardcoded HEIGHT
) {
// Interface compatible with standard host
#pragma HLS INTERFACE m_axi port=inR offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=inG offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=inB offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=outR offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=outG offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=outB offset=slave bundle=gmem5
#pragma HLS INTERFACE s_axilite port=width
#pragma HLS INTERFACE s_axilite port=height
#pragma HLS INTERFACE s_axilite port=return

    // Use hardcoded dimensions for optimization
    const int w = WIDTH;
    const int h = HEIGHT;

    for (int by = 0; by < h; by += 8) {
        for (int bx = 0; bx < w; bx += 8) {
#pragma HLS PIPELINE II=3

            pixel_t blk_R[8][8], blk_G[8][8], blk_B[8][8];
            coeff_t coef_R[8][8], coef_G[8][8], coef_B[8][8];

#pragma HLS ARRAY_PARTITION variable=blk_R complete dim=0
#pragma HLS ARRAY_PARTITION variable=blk_G complete dim=0
#pragma HLS ARRAY_PARTITION variable=blk_B complete dim=0
#pragma HLS ARRAY_PARTITION variable=coef_R complete dim=0
#pragma HLS ARRAY_PARTITION variable=coef_G complete dim=0
#pragma HLS ARRAY_PARTITION variable=coef_B complete dim=0

            // Load block
            for (int y = 0; y < 8; y++) {
#pragma HLS UNROLL
                for (int x = 0; x < 8; x++) {
#pragma HLS UNROLL
                    int gx = bx + x;
                    int gy = by + y;
                    int idx = gy * w + gx;

                    blk_R[y][x] = inR[idx];
                    blk_G[y][x] = inG[idx];
                    blk_B[y][x] = inB[idx];
                }
            }

            // Compute DCT
            dct_2d(blk_R, coef_R);
            dct_2d(blk_G, coef_G);
            dct_2d(blk_B, coef_B);

            // Store coefficients
            for (int y = 0; y < 8; y++) {
#pragma HLS UNROLL
                for (int x = 0; x < 8; x++) {
#pragma HLS UNROLL
                    int gx = bx + x;
                    int gy = by + y;
                    int idx = gy * w + gx;

                    outR[idx] = coef_R[y][x];
                    outG[idx] = coef_G[y][x];
                    outB[idx] = coef_B[y][x];
                }
            }
        }
    }
}
