#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

using pixel_t = uint8_t;
using coeff_t = int16_t;

static const int N = 8;

// Same DCT matrix as in hardware (double precision)
static const double C_d[N][N] = {
    {0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553},
    {0.490393, 0.415735, 0.277785, 0.097545,-0.097545,-0.277785,-0.415735,-0.490393},
    {0.461940, 0.191342,-0.191342,-0.461940,-0.461940,-0.191342, 0.191342, 0.461940},
    {0.415735,-0.097545,-0.490393,-0.277785, 0.277785, 0.490393, 0.097545,-0.415735},
    {0.353553,-0.353553,-0.353553, 0.353553, 0.353553,-0.353553,-0.353553, 0.353553},
    {0.277785,-0.490393, 0.097545, 0.415735,-0.415735,-0.097545, 0.490393,-0.277785},
    {0.191342,-0.461940, 0.461940,-0.191342,-0.191342, 0.461940,-0.461940, 0.191342},
    {0.097545,-0.277785, 0.415735,-0.490393, 0.490393,-0.415735, 0.277785,-0.097545}
};

// Standard JPEG luminance quant matrix (example)
static const int Q_luma[64] = {
     16, 11, 10, 16, 24, 40, 51, 61,
     12, 12, 14, 19, 26, 58, 60, 55,
     14, 13, 16, 24, 40, 57, 69, 56,
     14, 17, 22, 29, 51, 87, 80, 62,
     18, 22, 37, 56, 68,109,103, 77,
     24, 35, 55, 64, 81,104,113, 92,
     49, 64, 78, 87,103,121,120,101,
     72, 92, 95, 98,112,100,103, 99
};

// Zigzag order for 8x8
static const int zigzag[64] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

inline void dct_block_cpu(const pixel_t in[8][8], coeff_t out[8][8]) {
    double tmp[8][8];

    // Pass 1: Row transform
    // tmp[y][u] = Σ_x C[u][x] * (in[y][x] - 128)
    for (int y = 0; y < N; y++) {
        for (int u = 0; u < N; u++) {
            double acc = 0.0;
            for (int x = 0; x < N; x++) {
                acc += C_d[u][x] * (double(in[y][x]) - 128.0);
            }
            tmp[y][u] = acc;
        }
    }

    // Pass 2: Column transform
    // out[u][v] = Σ_y C[v][y] * tmp[y][u]
    for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
            double acc = 0.0;
            for (int y = 0; y < N; y++) {
                acc += C_d[v][y] * tmp[y][u];
            }
            int val = (int)std::lround(acc);
            if (val < -32768) val = -32768;
            if (val >  32767) val =  32767;
            out[u][v] = (coeff_t)val;
        }
    }
}

// ============================================
// INVERSE DCT (CPU)
// ============================================
inline void idct_block_cpu(const coeff_t in[8][8], pixel_t out[8][8]) {
    double tmp[8][8];

    // Pass 1: Inverse column transform
    // tmp[y][u] = Σ_v C[v][y] * in[u][v]
    for (int y = 0; y < N; y++) {
        for (int u = 0; u < N; u++) {
            double acc = 0.0;
            for (int v = 0; v < N; v++) {
                acc += C_d[v][y] * (double)in[u][v];
            }
            tmp[y][u] = acc;
        }
    }

    // Pass 2: Inverse row transform
    // out[y][x] = Σ_u C[u][x] * tmp[y][u] + 128
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            double acc = 0.0;
            for (int u = 0; u < N; u++) {
                acc += C_d[u][x] * tmp[y][u];
            }
            int val = (int)std::lround(acc + 128.0);
            if (val < 0)   val = 0;
            if (val > 255) val = 255;
            out[y][x] = (pixel_t)val;
        }
    }
}
// Quantize 8x8 block using Q_luma (you can adapt for chroma if needed)
inline void quant_block(const coeff_t in[8][8], coeff_t out[8][8]) {
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int idx = y*8 + x;
            int q = Q_luma[idx];
            int v = in[y][x];
            int qv = (int)std::round((double)v / (double)q);
            if (qv < -32768) qv = -32768;
            if (qv >  32767) qv =  32767;
            out[y][x] = (coeff_t)qv;
        }
    }
}

inline void dequant_block(const coeff_t in[8][8], coeff_t out[8][8]) {
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int idx = y*8 + x;
            int q = Q_luma[idx];
            int v = in[y][x];
            int dq = v * q;
            if (dq < -32768) dq = -32768;
            if (dq >  32767) dq =  32767;
            out[y][x] = (coeff_t)dq;
        }
    }
}

// Convert 8x8 block to zigzag 64-vector
inline void zigzag_block(const coeff_t blk[8][8], std::vector<coeff_t> &out) {
    out.resize(64);
    for (int i = 0; i < 64; i++) {
        int idx = zigzag[i];
        int y = idx / 8;
        int x = idx % 8;
        out[i] = blk[y][x];
    }
}

inline void inv_zigzag_block(const std::vector<coeff_t> &in, coeff_t blk[8][8]) {
    for (int i = 0; i < 64; i++) {
        int idx = zigzag[i];
        int y = idx / 8;
        int x = idx % 8;
        blk[y][x] = in[i];
    }
}

// Simple RLE: (value, run_length) for zero-runs on AC coefficients.
// Here we just RLE the full 64 entries for demo.
inline void rle_encode(const std::vector<coeff_t> &in,
                       std::vector<std::pair<coeff_t,int>> &out)
{
    out.clear();
    int n = (int)in.size();
    int i = 0;
    while (i < n) {
        coeff_t v = in[i];
        int run = 1;
        while (i+run < n && in[i+run] == v) {
            run++;
        }
        out.emplace_back(v, run);
        i += run;
    }
}

inline void rle_decode(const std::vector<std::pair<coeff_t,int>> &in,
                       std::vector<coeff_t> &out)
{
    out.clear();
    for (auto &p : in) {
        for (int k = 0; k < p.second; k++) {
            out.push_back(p.first);
        }
    }
}

// PSNR
inline double compute_psnr_channel(const std::vector<pixel_t> &orig,
                                   const std::vector<pixel_t> &recon)
{
    const int Np = (int)orig.size();
    double mse = 0.0;
    for (int i = 0; i < Np; i++) {
        double d = double(orig[i]) - double(recon[i]);
        mse += d*d;
    }
    mse /= double(Np);
    if (mse == 0.0) return 99.0;
    double maxI = 255.0;
    return 10.0 * std::log10((maxI*maxI)/mse);
}
