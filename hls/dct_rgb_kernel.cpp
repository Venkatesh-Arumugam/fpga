#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>

// Pixel and internal types
typedef ap_uint<8>    pixel_t;
typedef ap_fixed<18,4> dct_t;

static const int N = 8;

// Precomputed DCT matrix
static const dct_t C[N][N] = {
    {0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f},
    {0.490393f, 0.415735f, 0.277785f, 0.097545f,-0.097545f,-0.277785f,-0.415735f,-0.490393f},
    {0.461940f, 0.191342f,-0.191342f,-0.461940f,-0.461940f,-0.191342f, 0.191342f, 0.461940f},
    {0.415735f,-0.097545f,-0.490393f,-0.277785f, 0.277785f, 0.490393f, 0.097545f,-0.415735f},
    {0.353553f,-0.353553f,-0.353553f, 0.353553f, 0.353553f,-0.353553f,-0.353553f, 0.353553f},
    {0.277785f,-0.490393f, 0.097545f, 0.415735f,-0.415735f,-0.097545f, 0.490393f,-0.277785f},
    {0.191342f,-0.461940f, 0.461940f,-0.191342f,-0.191342f, 0.461940f,-0.461940f, 0.191342f},
    {0.097545f,-0.277785f, 0.415735f,-0.490393f, 0.490393f,-0.415735f, 0.277785f,-0.097545f}
};

static void dct_2d(pixel_t in_blk[8][8], pixel_t out_blk[8][8]) {
#pragma HLS INLINE
    dct_t tmp[8][8];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=0

    // Row transform
    ROWS:
    for (int u = 0; u < N; u++) {
#pragma HLS UNROLL
        for (int v = 0; v < N; v++) {
#pragma HLS UNROLL
            dct_t acc = 0;
            for (int x = 0; x < N; x++) {
#pragma HLS UNROLL
                acc += C[u][x] * (dct_t)((int)in_blk[x][v] - 128);
            }
            tmp[u][v] = acc;
        }
    }

    // Column transform
    COLS:
    for (int u = 0; u < N; u++) {
#pragma HLS UNROLL
        for (int v = 0; v < N; v++) {
#pragma HLS UNROLL
            dct_t acc = 0;
            for (int y = 0; y < N; y++) {
#pragma HLS UNROLL
                acc += tmp[u][y] * C[v][y];
            }
            // clamp and convert back
            int val = (int)hls::round(acc + 128);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            out_blk[u][v] = (pixel_t)val;
        }
    }
}

extern "C" void dct_rgb(
    const pixel_t* inR,
    const pixel_t* inG,
    const pixel_t* inB,
    pixel_t* outR,
    pixel_t* outG,
    pixel_t* outB,
    int width,
    int height
) {
#pragma HLS INTERFACE m_axi port=inR  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=inG  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=inB  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=outR offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=outG offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=outB offset=slave bundle=gmem5

#pragma HLS INTERFACE s_axilite port=inR   bundle=control
#pragma HLS INTERFACE s_axilite port=inG   bundle=control
#pragma HLS INTERFACE s_axilite port=inB   bundle=control
#pragma HLS INTERFACE s_axilite port=outR  bundle=control
#pragma HLS INTERFACE s_axilite port=outG  bundle=control
#pragma HLS INTERFACE s_axilite port=outB  bundle=control
#pragma HLS INTERFACE s_axilite port=width bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    pixel_t R_blk[8][8], G_blk[8][8], B_blk[8][8];
#pragma HLS ARRAY_PARTITION variable=R_blk complete dim=0
#pragma HLS ARRAY_PARTITION variable=G_blk complete dim=0
#pragma HLS ARRAY_PARTITION variable=B_blk complete dim=0

    pixel_t R_out[8][8], G_out[8][8], B_out[8][8];
#pragma HLS ARRAY_PARTITION variable=R_out complete dim=0
#pragma HLS ARRAY_PARTITION variable=G_out complete dim=0
#pragma HLS ARRAY_PARTITION variable=B_out complete dim=0

    // Process image in 8x8 tiles
    for (int by = 0; by < height; by += 8) {
        for (int bx = 0; bx < width; bx += 8) {
#pragma HLS PIPELINE II=1

            // Load tile for all 3 channels
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
#pragma HLS UNROLL
                    int gx = bx + x;
                    int gy = by + y;
                    int idx = gy * width + gx;

                    if (gx < width && gy < height) {
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

            // Run DCT on R, G, B separately
            dct_2d(R_blk, R_out);
            dct_2d(G_blk, G_out);
            dct_2d(B_blk, B_out);

            // Store tile
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
#pragma HLS UNROLL
                    int gx = bx + x;
                    int gy = by + y;
                    int idx = gy * width + gx;

                    if (gx < width && gy < height) {
                        outR[idx] = R_out[y][x];
                        outG[idx] = G_out[y][x];
                        outB[idx] = B_out[y][x];
                    }
                }
            }
        }
    }
}

