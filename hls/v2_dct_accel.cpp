/******************************************************************************
 * CONFIGURATION 3: DATAFLOW ARCHITECTURE
 * File: dct_config3_dataflow.cpp
 * Description: Separate load/compute/store with dataflow optimization
 * Expected: 4-5x speedup, higher resource usage
 ******************************************************************************/

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>
#include <hls_stream.h>

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

struct block_data {
    pixel_t R[8][8];
    pixel_t G[8][8];
    pixel_t B[8][8];
};

struct coeff_data {
    coeff_t R[8][8];
    coeff_t G[8][8];
    coeff_t B[8][8];
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

static void load_blocks_df(
    const pixel_t* inR,
    const pixel_t* inG,
    const pixel_t* inB,
    hls::stream<block_data>& block_stream,
    int width,
    int height
) {
    for (int by = 0; by < height; by += 8) {
        for (int bx = 0; bx < width; bx += 8) {
#pragma HLS PIPELINE II=1
            block_data blk;
            
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < width && gy < height) {
                        int idx = gy * width + gx;
                        blk.R[y][x] = inR[idx];
                        blk.G[y][x] = inG[idx];
                        blk.B[y][x] = inB[idx];
                    } else {
                        blk.R[y][x] = 0;
                        blk.G[y][x] = 0;
                        blk.B[y][x] = 0;
                    }
                }
            }
            block_stream.write(blk);
        }
    }
}

static void compute_dct_df(
    hls::stream<block_data>& in_stream,
    hls::stream<coeff_data>& out_stream,
    int width,
    int height
) {
    int num_blocks = ((height + 7) / 8) * ((width + 7) / 8);
    
    for (int i = 0; i < num_blocks; i++) {
#pragma HLS PIPELINE II=1
        block_data blk = in_stream.read();
        coeff_data coef;
        
        dct_2d(blk.R, coef.R);
        dct_2d(blk.G, coef.G);
        dct_2d(blk.B, coef.B);
        
        out_stream.write(coef);
    }
}

static void store_blocks_df(
    hls::stream<coeff_data>& coeff_stream,
    coeff_t* outR,
    coeff_t* outG,
    coeff_t* outB,
    int width,
    int height
) {
    for (int by = 0; by < height; by += 8) {
        for (int bx = 0; bx < width; bx += 8) {
#pragma HLS PIPELINE II=1
            coeff_data coef = coeff_stream.read();
            
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < width && gy < height) {
                        int idx = gy * width + gx;
                        outR[idx] = coef.R[x][y];
                        outG[idx] = coef.G[x][y];
                        outB[idx] = coef.B[x][y];
                    }
                }
            }
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
#pragma HLS INTERFACE m_axi port=inR offset=slave bundle=gmem0 depth=2073600 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=inG offset=slave bundle=gmem1 depth=2073600 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=inB offset=slave bundle=gmem2 depth=2073600 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=outR offset=slave bundle=gmem3 depth=2073600 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=outG offset=slave bundle=gmem4 depth=2073600 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=outB offset=slave bundle=gmem5 depth=2073600 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=width
#pragma HLS INTERFACE s_axilite port=height
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS DATAFLOW
    
    hls::stream<block_data> block_stream("block_stream");
#pragma HLS STREAM variable=block_stream depth=4
    
    hls::stream<coeff_data> coeff_stream("coeff_stream");
#pragma HLS STREAM variable=coeff_stream depth=4
    
    load_blocks_df(inR, inG, inB, block_stream, width, height);
    compute_dct_df(block_stream, coeff_stream, width, height);
    store_blocks_df(coeff_stream, outR, outG, outB, width, height);
}
