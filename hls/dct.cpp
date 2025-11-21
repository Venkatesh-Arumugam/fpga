#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>

// Use 8-bit pixels and a fixed-point internal type
typedef ap_uint<8>    pixel_t;
typedef ap_fixed<18,4> dct_t;

// 1D DCT size
static const int N = 8;

// Precomputed DCT matrix (float-like). You can tune later.
static const dct_t C[N][N] = {
    { 0.353553f,  0.353553f,  0.353553f,  0.353553f,  0.353553f,  0.353553f,  0.353553f,  0.353553f },
    { 0.490393f,  0.415735f,  0.277785f,  0.097545f, -0.097545f, -0.277785f, -0.415735f, -0.490393f },
    { 0.461940f,  0.191342f, -0.191342f, -0.461940f, -0.461940f, -0.191342f,  0.191342f,  0.461940f },
    { 0.415735f, -0.097545f, -0.490393f, -0.277785f,  0.277785f,  0.490393f,  0.097545f, -0.415735f },
    { 0.353553f, -0.353553f, -0.353553f,  0.353553f,  0.353553f, -0.353553f, -0.353553f,  0.353553f },
    { 0.277785f, -0.490393f,  0.097545f,  0.415735f, -0.415735f, -0.097545f,  0.490393f, -0.277785f },
    { 0.191342f, -0.461940f,  0.461940f, -0.191342f, -0.191342f,  0.461940f, -0.461940f,  0.191342f },
    { 0.097545f, -0.277785f,  0.415735f, -0.490393f,  0.490393f, -0.415735f,  0.277785f, -0.097545f }
};

// 1-channel 8x8 block DCT (in-place-ish on local arrays)
static void dct_block_1ch(pixel_t in_blk[N][N], pixel_t out_blk[N][N]) {
#pragma HLS INLINE

    dct_t tmp[N][N];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=0

    // Row-wise transform: tmp = C * in_blk
    RowLoop:
    for (int u = 0; u < N; u++) {
#pragma HLS UNROLL
        ColLoop:
        for (int v = 0; v < N; v++) {
#pragma HLS UNROLL
            dct_t acc = 0;
            InnerRow:
            for (int x = 0; x < N; x++) {
#pragma HLS UNROLL
                acc += C[u][x] * (dct_t)((int)in_blk[x][v] - 128); // center around 0
            }
            tmp[u][v] = acc;
        }
    }

    // Column-wise: out_blk = tmp * C^T
    ColLoop2:
    for (int u = 0; u < N; u++) {
#pragma HLS UNROLL
        ColLoop3:
        for (int v = 0; v < N; v++) {
#pragma HLS UNROLL
            dct_t acc = 0;
            InnerCol:
            for (int y = 0; y < N; y++) {
#pragma HLS UNROLL
                acc += tmp[u][y] * C[v][y];
            }
            // Simple clamp+round back to 8-bit for now
            int val = (int)hls::round(acc + 128); // shift back
            if (val < 0)   val = 0;
            if (val > 255) val = 255;
            out_blk[u][v] = (pixel_t)val;
        }
    }
}

// Process one full channel, image in row-major layout
static void dct_channel(
    const pixel_t *in,
    pixel_t       *out,
    int           width,
    int           height
) {
#pragma HLS INLINE off

    pixel_t blk_in[N][N];
    pixel_t blk_out[N][N];
#pragma HLS ARRAY_PARTITION variable=blk_in  complete dim=0
#pragma HLS ARRAY_PARTITION variable=blk_out complete dim=0

    // Process tiles of 8x8
    BlockRow:
    for (int by = 0; by < height; by += N) {
        BlockCol:
        for (int bx = 0; bx < width; bx += N) {
#pragma HLS PIPELINE II=1

            // Load 8x8 block from global memory
            LoadY:
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
#pragma HLS UNROLL
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < width && gy < height) {
                        blk_in[y][x] = in[gy * width + gx];
                    } else {
                        blk_in[y][x] = 0; // padding
                    }
                }
            }

            // 2D DCT on this block
            dct_block_1ch(blk_in, blk_out);

            // Store back
            StoreY:
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
#pragma HLS UNROLL
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < width && gy < height) {
                        out[gy * width + gx] = blk_out[y][x];
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Top-level kernel: 3 channels: Y, Cb, Cr
// ------------------------------------------------------------------
extern "C" void dct8x8_color(
    const pixel_t *inY,
    const pixel_t *inCb,
    const pixel_t *inCr,
    pixel_t       *outY,
    pixel_t       *outCb,
    pixel_t       *outCr,
    int           width,
    int           height
) {
#pragma HLS INTERFACE m_axi     port=inY   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=inCb  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=inCr  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi     port=outY  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi     port=outCb offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi     port=outCr offset=slave bundle=gmem5

#pragma HLS INTERFACE s_axilite port=inY     bundle=control
#pragma HLS INTERFACE s_axilite port=inCb    bundle=control
#pragma HLS INTERFACE s_axilite port=inCr    bundle=control
#pragma HLS INTERFACE s_axilite port=outY    bundle=control
#pragma HLS INTERFACE s_axilite port=outCb   bundle=control
#pragma HLS INTERFACE s_axilite port=outCr   bundle=control
#pragma HLS INTERFACE s_axilite port=width   bundle=control
#pragma HLS INTERFACE s_axilite port=height  bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

#pragma HLS DATAFLOW

    dct_channel(inY,  outY,  width, height);
    dct_channel(inCb, outCb, width, height);
    dct_channel(inCr, outCr, width, height);
}
