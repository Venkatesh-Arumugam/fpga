/******************************************************************************
 * CONFIGURATION 2: PIPELINED
 * File: dct_config2_pipelined.cpp
 * Description: Add PIPELINE II=1 to BLOCK_X loop
 * Expected: 2-3x speedup, moderate resource increase
 ******************************************************************************/

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_uint<8>  pixel_t;
typedef ap_int<16>  coeff_t;
typedef ap_fixed<24,12> dct_t;

static const int N = 8;

static const dct_t C[N][N] = {
    {0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553},
    {0.490393, 0.415735, 0.277785, 0.097545,-0.097545,-0.277785,-0.415735,-0.490393},
    {0.461940, 0.191342,-0.191342,-0.461940,-0.461940,-0.191342, 0.191342, 0.461940},
    {0.415735,-0.097545,-0.490393,-0.277785, 0.277785, 0.490393, 0.097545,-0.415735},
    {0.353553,-0.353553,-0.353553, 0.353553, 0.353553,-0.353553,-0.353553, 0.353553},
    {0.277785,-0.490393, 0.097545, 0.415735,-0.415735,-0.097545, 0.490393,-0.277785},
    {0.191342,-0.461940, 0.461940,-0.191342,-0.191342, 0.461940,-0.461940, 0.191342},
    {0.097545,-0.277785, 0.415735,-0.490393, 0.490393,-0.415735, 0.277785,-0.097545}
};

static void dct_2d(pixel_t in_blk[8][8], coeff_t out_blk[8][8])
{
#pragma HLS INLINE
    dct_t tmp[8][8];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=0

    for (int u = 0; u < 8; u++) {
#pragma HLS UNROLL
        for (int v = 0; v < 8; v++) {
#pragma HLS UNROLL
            dct_t acc = 0;
            for (int x = 0; x < 8; x++) {
#pragma HLS UNROLL
                acc += C[u][x] * (dct_t)((int)in_blk[x][v] - 128);
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
            out_blk[u][v] = (coeff_t)val;
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
#pragma HLS INTERFACE m_axi port=inR offset=slave bundle=gmem0 depth=2073600
#pragma HLS INTERFACE m_axi port=inG offset=slave bundle=gmem1 depth=2073600
#pragma HLS INTERFACE m_axi port=inB offset=slave bundle=gmem2 depth=2073600
#pragma HLS INTERFACE m_axi port=outR offset=slave bundle=gmem3 depth=2073600
#pragma HLS INTERFACE m_axi port=outG offset=slave bundle=gmem4 depth=2073600
#pragma HLS INTERFACE m_axi port=outB offset=slave bundle=gmem5 depth=2073600
#pragma HLS INTERFACE s_axilite port=width
#pragma HLS INTERFACE s_axilite port=height
#pragma HLS INTERFACE s_axilite port=return

    pixel_t R_blk[8][8], G_blk[8][8], B_blk[8][8];
#pragma HLS ARRAY_PARTITION variable=R_blk complete dim=0
#pragma HLS ARRAY_PARTITION variable=G_blk complete dim=0
#pragma HLS ARRAY_PARTITION variable=B_blk complete dim=0

    coeff_t R_coef[8][8], G_coef[8][8], B_coef[8][8];
#pragma HLS ARRAY_PARTITION variable=R_coef complete dim=0
#pragma HLS ARRAY_PARTITION variable=G_coef complete dim=0
#pragma HLS ARRAY_PARTITION variable=B_coef complete dim=0

BLOCK_Y:
    for (int by = 0; by < height; by += 8) {
    BLOCK_X:
        for (int bx = 0; bx < width; bx += 8) {
#pragma HLS PIPELINE II=1

        LOAD_BLOCK:
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < width && gy < height) {
                        int idx = gy * width + gx;
                        R_blk[y][x] = inR[idx];
                        G_blk[y][x] = inG[idx];
                        B_blk[y][x] = inB[idx];
                    } else {
                        R_blk[y][x] = 0;
                        G_blk[y][x] = 0;
                        B_blk[y][x] = 0;
                    }
                }
            }

            dct_2d(R_blk, R_coef);
            dct_2d(G_blk, G_coef);
            dct_2d(B_blk, B_coef);

        STORE_BLOCK:
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < width && gy < height) {
                        int idx = gy * width + gx;
                        outR[idx] = R_coef[x][y];
                        outG[idx] = G_coef[x][y];
                        outB[idx] = B_coef[x][y];
                    }
                }
            }
        }
    }
}