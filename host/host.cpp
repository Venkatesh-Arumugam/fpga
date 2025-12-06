#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "experimental/xrt_xclbin.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using pixel_t = uint8_t;

// =============================================================
// IDCT CONSTANT MATRIX (for 8x8)
// =============================================================
static const float Cmat[8][8] = {
    {0.353553,0.353553,0.353553,0.353553,0.353553,0.353553,0.353553,0.353553},
    {0.490393,0.415735,0.277785,0.097545,-0.097545,-0.277785,-0.415735,-0.490393},
    {0.461940,0.191342,-0.191342,-0.461940,-0.461940,-0.191342,0.191342,0.461940},
    {0.415735,-0.097545,-0.490393,-0.277785,0.277785,0.490393,0.097545,-0.415735},
    {0.353553,-0.353553,-0.353553,0.353553,0.353553,-0.353553,-0.353553,0.353553},
    {0.277785,-0.490393,0.097545,0.415735,-0.415735,-0.097545,0.490393,-0.277785},
    {0.191342,-0.461940,0.461940,-0.191342,-0.191342,0.461940,-0.461940,0.191342},
    {0.097545,-0.277785,0.415735,-0.490393,0.490393,-0.415735,0.277785,-0.097545}
};

// =============================================================
// IDCT BLOCK
// =============================================================
static void idct_block(float in[8][8], float out[8][8]) {
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            float sum = 0.0f;
            for (int u = 0; u < 8; u++)
                for (int v = 0; v < 8; v++)
                    sum += Cmat[u][x] * Cmat[v][y] * in[u][v];
            out[x][y] = sum;
        }
    }
}

// =============================================================
// Compute PSNR
// =============================================================
double compute_psnr(const std::vector<pixel_t>& a,
                    const std::vector<pixel_t>& b)
{
    double mse = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = (double)a[i] - (double)b[i];
        mse += diff * diff;
    }
    mse /= a.size();
    if (mse == 0) return 100.0;
    return 10.0 * log10((255.0 * 255.0) / mse);
}

// =============================================================
// MAIN
// =============================================================
int main(int argc, char *argv[]) {

    if (argc < 4) {
        std::cout << "Usage: host.exe <xclbin> <input.png> <output.png> [iters]\n";
        return 1;
    }

    std::string xclbin_path = argv[1];
    std::string input_path  = argv[2];
    std::string output_path = argv[3];
    int iters = (argc > 4) ? std::stoi(argv[4]) : 1;

    // -----------------------------------------------------------
    // Load image
    // -----------------------------------------------------------
    int W, H, C;
    unsigned char* img = stbi_load(input_path.c_str(), &W, &H, &C, 3);
    if (!img) throw std::runtime_error("Cannot load image!");

    size_t N = (size_t)W * H;

    std::vector<pixel_t> R(N), G(N), B(N);
    std::vector<pixel_t> outR(N), outG(N), outB(N);

    for (size_t i = 0; i < N; i++) {
        R[i] = img[3*i+0];
        G[i] = img[3*i+1];
        B[i] = img[3*i+2];
    }
    stbi_image_free(img);

    // -----------------------------------------------------------
    // FPGA Setup
    // -----------------------------------------------------------
    xrt::device device(0);
    auto xclbin = xrt::xclbin(xclbin_path);
    auto uuid = device.load_xclbin(xclbin);
    xrt::kernel krnl(device, uuid, "dct_rgb");

    size_t BYTES = N * sizeof(pixel_t);

    xrt::bo bo_inR (device, BYTES, krnl.group_id(0));
    xrt::bo bo_inG (device, BYTES, krnl.group_id(1));
    xrt::bo bo_inB (device, BYTES, krnl.group_id(2));
    xrt::bo bo_outR(device, BYTES, krnl.group_id(3));
    xrt::bo bo_outG(device, BYTES, krnl.group_id(4));
    xrt::bo bo_outB(device, BYTES, krnl.group_id(5));

    bo_inR.write(R.data());
    bo_inG.write(G.data());
    bo_inB.write(B.data());

    bo_inR.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inG.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // -----------------------------------------------------------
    // Run kernel
    // -----------------------------------------------------------
    auto run = krnl(bo_inR, bo_inG, bo_inB,
                    bo_outR, bo_outG, bo_outB,
                    W, H);
    run.wait();

    bo_outR.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outG.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    bo_outR.read(outR.data());
    bo_outG.read(outG.data());
    bo_outB.read(outB.data());



    // DEBUG: Print first 32 output pixels from FPGA
std::cout << "\n===== FPGA RAW OUTPUT (first 32 pixels) =====\n";
for (int i = 0; i < 32; i++) {
    std::cout << "R[" << i << "]=" << (int)R_out[i]
              << " G[" << i << "]=" << (int)G_out[i]
              << "  B[" << i << "]=" << (int)B_out[i] << "\n";
}
std::cout << "============================================\n\n";

    // -----------------------------------------------------------
    // Apply CPU IDCT
    // -----------------------------------------------------------
    std::vector<pixel_t> recR(N), recG(N), recB(N);

    float blk[8][8], rec[8][8];

    auto idct_channel = [&](const std::vector<pixel_t>& F, std::vector<pixel_t>& Out) {
        for (int by = 0; by < H; by += 8) {
            for (int bx = 0; bx < W; bx += 8) {

                for (int u = 0; u < 8; u++)
                    for (int v = 0; v < 8; v++) {
                        int gx = bx + v, gy = by + u;
                        float coeff = 0;
                        if (gx < W && gy < H) coeff = (float)F[gy*W + gx] - 128.0f;
                        blk[u][v] = coeff;
                    }

                idct_block(blk, rec);

                for (int u = 0; u < 8; u++)
                    for (int v = 0; v < 8; v++) {
                        int gx = bx + v, gy = by + u;
                        if (gx < W && gy < H) {
                            float val = rec[u][v];
                            if (val < 0) val = 0;
                            if (val > 255) val = 255;
                            Out[gy*W + gx] = (uint8_t)val;
                        }
                    }
            }
        }
    };

    idct_channel(outR, recR);
    idct_channel(outG, recG);
    idct_channel(outB, recB);

    // -----------------------------------------------------------
    // Compute PSNR
    // -----------------------------------------------------------
    double psnr_R = compute_psnr(R, recR);
    double psnr_G = compute_psnr(G, recG);
    double psnr_B = compute_psnr(B, recB);

    std::cout << "PSNR(R) = " << psnr_R << " dB\n";
    std::cout << "PSNR(G) = " << psnr_G << " dB\n";
    std::cout << "PSNR(B) = " << psnr_B << " dB\n";

    // -----------------------------------------------------------
    // Save reconstructed RGB
    // -----------------------------------------------------------
    std::vector<pixel_t> finalRGB(N * 3);
    for (size_t i = 0; i < N; i++) {
        finalRGB[3*i+0] = recR[i];
        finalRGB[3*i+1] = recG[i];
        finalRGB[3*i+2] = recB[i];
    }

    stbi_write_png(output_path.c_str(), W, H, 3, finalRGB.data(), W*3);

    std::cout << "Reconstructed image written to " << output_path << "\n";
    return 0;
}

