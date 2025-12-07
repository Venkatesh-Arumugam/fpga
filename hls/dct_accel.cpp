#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_uint<8>    pixel_t;
typedef ap_fixed<24,6> dct_t;   
typedef ap_int<16>    coeff_t;

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

static void dct_2d(pixel_t in_blk[8][8], coeff_t out_blk[8][8]) {
#pragma HLS INLINE

    dct_t tmp[8][8];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=0

    // --------------------------
    // Pass 1: Row Transform
    // For each input row y, compute DCT giving frequency components u
    // tmp[y][u] = Σ_x C[u][x] * (in_blk[y][x] - 128)
    // --------------------------
    ROW_LOOP: for (int y = 0; y < 8; y++) {
#pragma HLS UNROLL
        for (int u = 0; u < 8; u++) {
#pragma HLS UNROLL
            dct_t acc = 0;
            for (int x = 0; x < 8; x++) {
#pragma HLS UNROLL
                acc += C[u][x] * (dct_t)((int)in_blk[y][x] - 128);
            }
            tmp[y][u] = acc;
        }
    }

    // --------------------------
    // Pass 2: Column Transform
    // For each frequency component (u,v), sum over spatial rows y
    // out_blk[u][v] = Σ_y C[v][y] * tmp[y][u]
    // --------------------------
    COL_LOOP: for (int u = 0; u < 8; u++) {
#pragma HLS UNROLL
        for (int v = 0; v < 8; v++) {
#pragma HLS UNROLL
            dct_t acc = 0;
            for (int y = 0; y < 8; y++) {
#pragma HLS UNROLL
                acc += C[v][y] * tmp[y][u];
            }
            int val = (int)hls::round(acc);
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
    pixel_t R_blk[8][8], G_blk[8][8], B_blk[8][8];
#pragma HLS ARRAY_PARTITION variable=R_blk complete dim=0
#pragma HLS ARRAY_PARTITION variable=G_blk complete dim=0
#pragma HLS ARRAY_PARTITION variable=B_blk complete dim=0

    coeff_t R_coef[8][8], G_coef[8][8], B_coef[8][8];
#pragma HLS ARRAY_PARTITION variable=R_coef complete dim=0
#pragma HLS ARRAY_PARTITION variable=G_coef complete dim=0
#pragma HLS ARRAY_PARTITION variable=B_coef complete dim=0

    BLOCK_Y: for (int by = 0; by < height; by += 8) {
        BLOCK_X: for (int bx = 0; bx < width; bx += 8) {

            // --------------------------
            // Load 8x8 block from memory
            // --------------------------
            LOAD_Y: for (int y = 0; y < 8; y++) {
                LOAD_X: for (int x = 0; x < 8; x++) {
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

            // --------------------------
            // Compute DCT for each channel
            // --------------------------
            dct_2d(R_blk, R_coef);
            dct_2d(G_blk, G_coef);
            dct_2d(B_blk, B_coef);

            // --------------------------
            // Store DCT coefficients
            // Note: We store R_coef[u][v] at position (by+u, bx+v)
            // This matches the expected layout where coefficient [u][v]
            // goes to spatial position (u, v) within the block
            // --------------------------
            STORE_Y: for (int y = 0; y < 8; y++) {
                STORE_X: for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < width && gy < height) {
                        int idx = gy * width + gx;
                        // Store coefficient [y][x] at position (gy, gx)
                        outR[idx] = R_coef[y][x];
                        outG[idx] = G_coef[y][x];
                        outB[idx] = B_coef[y][x];
                    }
                }
            }
        }
    }
}
