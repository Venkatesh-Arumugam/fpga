#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>

// 8-bit pixel, fixed-point internal type
typedef ap_uint<8>    pixel_t;
typedef ap_fixed<18,4> dct_t;

static const int N = 8;

// Precomputed 8×8 DCT basis matrix
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


// Clamp helper
static inline pixel_t clamp_to_u8(int val) {
#pragma HLS INLINE
    if (val < 0)   return 0;
    if (val > 255) return 255;
    return (pixel_t)val;
}

// ---------------------------------------------------------------------
// 2D DCT on a single 8×8 block (one channel)
// ---------------------------------------------------------------------
static void dct_block(pixel_t in_blk[N][N], pixel_t out_blk[N][N]) {
#pragma HLS INLINE

    dct_t tmp[N][N];

    // Row transform: tmp[u][x] = sum_y C[u][y] * (in_blk[y][x] - 128)
    Row_Transform_U:
    for (int u = 0; u < N; u++) {
        Row_Transform_X:
        for (int x = 0; x < N; x++) {
#pragma HLS PIPELINE II=1
            dct_t acc = 0;
        Row_Transform_Y:
            for (int y = 0; y < N; y++) {
                int shifted = (int)in_blk[y][x] - 128;
                acc += C[u][y] * (dct_t)shifted;
            }
            tmp[u][x] = acc;
        }
    }

    // Column transform: out_blk[u][v] = sum_x tmp[u][x] * C[v][x]
    Col_Transform_U:
    for (int u = 0; u < N; u++) {
        Col_Transform_V:
        for (int v = 0; v < N; v++) {
#pragma HLS PIPELINE II=1
            dct_t acc = 0;
        Col_Transform_X:
            for (int x = 0; x < N; x++) {
                acc += tmp[u][x] * C[v][x];
            }

            int val = (int)hls::round(acc + 128); // simple shift back
            out_blk[u][v] = clamp_to_u8(val);
        }
    }
}

// ---------------------------------------------------------------------
// Apply DCT to an entire image channel
// ---------------------------------------------------------------------
static void dct_channel(
    const pixel_t *in,
    pixel_t       *out,
    int            width,
    int            height
) {
#pragma HLS INLINE off

    pixel_t blk_in[N][N];
    pixel_t blk_out[N][N];

    // Process image in 8×8 blocks
    for (int by = 0; by < height; by += N) {
        for (int bx = 0; bx < width; bx += N) {
            // Load 8×8 block from global memory
        Load_Block_Y:
            for (int y = 0; y < N; y++) {
            Load_Block_X:
                for (int x = 0; x < N; x++) {
#pragma HLS PIPELINE II=1
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < width && gy < height)
                        blk_in[y][x] = in[gy * width + gx];
                    else
                        blk_in[y][x] = 0;
                }
            }

            // Compute DCT on this block
            dct_block(blk_in, blk_out);

            // Store 8×8 block back to global memory
        Store_Block_Y:
            for (int y = 0; y < N; y++) {
            Store_Block_X:
                for (int x = 0; x < N; x++) {
#pragma HLS PIPELINE II=1
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < width && gy < height)
                        out[gy * width + gx] = blk_out[y][x];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Top-level kernel for RGB image processing
// ---------------------------------------------------------------------
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

#pragma HLS INTERFACE s_axilite port=inR    bundle=control
#pragma HLS INTERFACE s_axilite port=inG    bundle=control
#pragma HLS INTERFACE s_axilite port=inB    bundle=control
#pragma HLS INTERFACE s_axilite port=outR   bundle=control
#pragma HLS INTERFACE s_axilite port=outG   bundle=control
#pragma HLS INTERFACE s_axilite port=outB   bundle=control
#pragma HLS INTERFACE s_axilite port=width  bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Note: Channels are processed sequentially to control resource usage.
    dct_channel(inR, outR, width, height);
    dct_channel(inG, outG, width, height);
    dct_channel(inB, outB, width, height);
}
