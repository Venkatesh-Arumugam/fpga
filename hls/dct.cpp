//V1 Pipelined Sequential DCT
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdio.h>

typedef ap_uint<8>  pixel_t;
typedef ap_int<16>  coeff_t;

// Smaller fixed-point allows DSP-free arithmetic
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

// =============================
//  Variant-1: Sequential DCT
// =============================
static void dct_2d(pixel_t in_blk[8][8], coeff_t out_blk[8][8])
{
#pragma HLS INLINE off     // Prevent huge inlining → faster HLS compile

    dct_t tmp[8][8];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=2

    // ---------- ROW TRANSFORM ----------
ROW:
    for (int u = 0; u < 8; u++) {
#pragma HLS PIPELINE II=1
        for (int v = 0; v < 8; v++) {

            dct_t acc = 0;
            for (int x = 0; x < 8; x++) {
#pragma HLS DEPENDENCE variable=in_blk inter false
                acc += C[u][x] * (dct_t)((int)in_blk[x][v] - 128);
            }
            tmp[u][v] = acc;
        }
    }

    // ---------- COLUMN TRANSFORM ----------
COL:
    for (int u = 0; u < 8; u++) {
#pragma HLS PIPELINE II=1
        for (int v = 0; v < 8; v++) {

            dct_t acc = 0;
            for (int y = 0; y < 8; y++) {
#pragma HLS DEPENDENCE variable=tmp inter false
                acc += tmp[u][y] * C[v][y];
            }
            int val = (int)hls::round(acc);

            if (val < -32768) val = -32768;
            if (val >  32767) val =  32767;

            out_blk[u][v] = (coeff_t)val;
        }
    }
}

// ============================================================
// Top-level kernel
// ============================================================
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
    printf("DCT ACCEL KERNEL: Variant 1 - Pipelined Sequential Version\n");

    pixel_t R_blk[8][8], G_blk[8][8], B_blk[8][8];
#pragma HLS ARRAY_PARTITION variable=R_blk complete dim=2
#pragma HLS ARRAY_PARTITION variable=G_blk complete dim=2
#pragma HLS ARRAY_PARTITION variable=B_blk complete dim=2

    coeff_t R_coef[8][8], G_coef[8][8], B_coef[8][8];
#pragma HLS ARRAY_PARTITION variable=R_coef complete dim=2
#pragma HLS ARRAY_PARTITION variable=G_coef complete dim=2
#pragma HLS ARRAY_PARTITION variable=B_coef complete dim=2

BLOCK_Y:
    for (int by = 0; by < height; by += 8) {
    BLOCK_X:
        for (int bx = 0; bx < width; bx += 8) {

            // -------- LOAD BLOCK --------
LOAD:
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
#pragma HLS PIPELINE II=1
                    int idx = (by + y) * width + (bx + x);
                    if (bx + x < width && by + y < height) {
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

            // -------- COMPUTE --------
            dct_2d(R_blk, R_coef);
            dct_2d(G_blk, G_coef);
            dct_2d(B_blk, B_coef);

            // -------- STORE BLOCK --------
STORE:
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
#pragma HLS PIPELINE II=1
                    int idx = (by + y) * width + (bx + x);
                    if (bx + x < width && by + y < height) {
                        outR[idx] = R_coef[y][x];
                        outG[idx] = G_coef[y][x];
                        outB[idx] = B_coef[y][x];
                    }
                }
            }
        }
    }
}
