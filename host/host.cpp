#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>

// Pixel type (8-bit) and internal fixed-point type
typedef ap_uint<8>    pixel_t;
typedef ap_fixed<18,4> dct_t;

// DCT block size
static const int N = 8;

// Precomputed 8×8 DCT matrix
static const dct_t C[N][N] = {
    { 0.353553f,  0.353553f,  0.353553f,  0.353553f,  0.353553f,  0.353553f,  0.353553f,  0.353553f },
    { 0.490393f,  0.415735f,  0.277785f,  0.097545f, -0.097545f, -0.277785f, -0.415735f,-0.490393f },
    { 0.461940f,  0.191342f, -0.191342f, -0.461940f, -0.461940f, -0.191342f,  0.191342f, 0.461940f },
    { 0.415735f, -0.097545f, -0.490393f, -0.277785f,  0.277785f,  0.490393f,  0.097545f,-0.415735f },
    { 0.353553f, -0.353553f, -0.353553f,  0.353553f,  0.353553f, -0.353553f, -0.353553f, 0.353553f },
    { 0.277785f, -0.490393f,  0.097545f,  0.415735f, -0.415735f, -0.097545f,  0.490393f,-0.277785f },
    { 0.191342f, -0.461940f,  0.461940f, -0.191342f, -0.191342f,  0.461940f, -0.461940f, 0.191342f },
    { 0.097545f, -0.277785f,  0.415735f, -0.490393f,  0.490393f, -0.415735f,  0.277785f,-0.097545f }
};

// -------------------------------------------------------------
// 2D DCT on an 8×8 block (one channel)
// -------------------------------------------------------------
static void dct_block(pixel_t in_blk[N][N], pixel_t out_blk[N][N]) {
#pragma HLS INLINE

    dct_t tmp[N][N];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=0

    // Row transform
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
    for (int u = 0; u < N; u++) {
#pragma HLS UNROLL
        for (int v = 0; v < N; v++) {
#pragma HLS UNROLL
            dct_t acc = 0;
            for (int y = 0; y < N; y++) {
#pragma HLS UNROLL
                acc += tmp[u][y] * C[v][y];
            }

            int val = (int)hls::round(acc + 128);
            if (val < 0) val = 0;
            if (val > 255) val = 255;

            out_blk[u][v] = (pixel_t)val;
        }
    }
}

// -------------------------------------------------------------
// Apply DCT to an entire image channel (R / G / B)
// -------------------------------------------------------------
static void dct_channel(
    const pixel_t *in,
    pixel_t       *out,
    int            width,
    int            height
) {
#pragma HLS INLINE off

    pixel_t blk_in[N][N];
    pixel_t blk_out[N][N];
#pragma HLS ARRAY_PARTITION variable=blk_in complete dim=0
#pragma HLS ARRAY_PARTITION variable=blk_out complete dim=0

    for (int by = 0; by < height; by += N) {
        for (int bx = 0; bx < width; bx += N) {
#pragma HLS PIPELINE II=1

            // Load block
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
#pragma HLS UNROLL
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < width && gy < height)
                        blk_in[y][x] = in[gy * width + gx];
                    else
                        blk_in[y][x] = 0;
                }
            }

            // Perform DCT
            dct_block(blk_in, blk_out);

            // Store block
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
#pragma HLS UNROLL
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < width && gy < height)
                        out[gy * width + gx] = blk_out[y][x];
                }
            }
        }
    }
}

// -------------------------------------------------------------
// Top-level kernel for **RGB** images
// -------------------------------------------------------------
extern "C" void dct_rgb(
    const pixel_t *inR,
    const pixel_t *inG,
    const pixel_t *inB,
    pixel_t       *outR,
    pixel_t       *outG,
    pixel_t       *outB,
    int            width,
    int            height
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

#pragma HLS DATAFLOW

    dct_channel(inR, outR, width, height);
    dct_channel(inG, outG, width, height);
    dct_channel(inB, outB, width, height);
}
