// host.cpp – FPGA DCT (dct_rgb) + CPU IDCT + PSNR (spatial-domain only)

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <cstdint>
#include <string>

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "experimental/xrt_xclbin.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using pixel_t = uint8_t;

// ===========================================================
// 8×8 DCT matrix (same as in your HLS code) & IDCT
// ===========================================================
static const float Cmat[8][8] = {
    {0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f},
    {0.490393f, 0.415735f, 0.277785f, 0.097545f,-0.097545f,-0.277785f,-0.415735f,-0.490393f},
    {0.461940f, 0.191342f,-0.191342f,-0.461940f,-0.461940f,-0.191342f, 0.191342f, 0.461940f},
    {0.415735f,-0.097545f,-0.490393f,-0.277785f, 0.277785f, 0.490393f, 0.097545f,-0.415735f},
    {0.353553f,-0.353553f,-0.353553f, 0.353553f, 0.353553f,-0.353553f,-0.353553f, 0.353553f},
    {0.277785f,-0.490393f, 0.097545f, 0.415735f,-0.415735f,-0.097545f, 0.490393f,-0.277785f},
    {0.191342f,-0.461940f, 0.461940f,-0.191342f,-0.191342f, 0.461940f,-0.461940f, 0.191342f},
    {0.097545f,-0.277785f, 0.415735f,-0.490393f, 0.490393f,-0.415735f, 0.277785f,-0.097545f}
};

void idct_block(float in[8][8], float out[8][8]) {
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            float sum = 0.0f;
            for (int u = 0; u < 8; u++) {
                for (int v = 0; v < 8; v++) {
                    sum += Cmat[u][x] * Cmat[v][y] * in[u][v];
                }
            }
            out[x][y] = sum;
        }
    }
}

// ===========================================================
// PSNR between two RGB images (both as interleaved uint8_t)
// ===========================================================
double compute_psnr(const std::vector<pixel_t>& a,
                    const std::vector<pixel_t>& b)
{
    if (a.size() != b.size()) {
        throw std::runtime_error("PSNR: image size mismatch");
    }
    double mse = 0.0;
    size_t N = a.size();
    for (size_t i = 0; i < N; i++) {
        double d = double(a[i]) - double(b[i]);
        mse += d * d;
    }
    mse /= double(N);
    if (mse == 0.0) return 100.0;  // identical

    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

// ===========================================================
// MAIN HOST PIPELINE
// ===========================================================
int main(int argc, char* argv[]) {
    try {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0]
                      << " <xclbin> <input.png> <output.png> [iterations]\n";
            return 1;
        }

        std::string xclbin_path = argv[1];
        std::string input_path  = argv[2];
        std::string output_path = argv[3];
        int iterations = (argc > 4) ? std::stoi(argv[4]) : 1;

        // ----------------------------------------------------
        // 1) Load input image
        // ----------------------------------------------------
        int W = 0, H = 0, C = 0;
        uint8_t* img_data = stbi_load(input_path.c_str(), &W, &H, &C, 3);
        if (!img_data) {
            std::cerr << "ERROR: Failed to load " << input_path << "\n";
            return 1;
        }
        size_t num_pixels = size_t(W) * H;

        std::vector<pixel_t> rgb_orig(num_pixels * 3);
        std::vector<pixel_t> R(num_pixels), G(num_pixels), B(num_pixels);

        for (size_t i = 0; i < num_pixels; i++) {
            uint8_t r = img_data[3*i+0];
            uint8_t g = img_data[3*i+1];
            uint8_t b = img_data[3*i+2];

            rgb_orig[3*i+0] = r;
            rgb_orig[3*i+1] = g;
            rgb_orig[3*i+2] = b;

            R[i] = r;
            G[i] = g;
            B[i] = b;
        }
        stbi_image_free(img_data);

        std::cout << "Loaded " << input_path
                  << " (" << W << "x" << H << ", 3ch)\n";

        // ----------------------------------------------------
        // 2) Open device and load xclbin
        // ----------------------------------------------------
        std::cout << "Opening device 0...\n";
        xrt::device device{0};

        std::cout << "Loading xclbin: " << xclbin_path << "\n";
        xrt::xclbin xclbin{xclbin_path};
        auto uuid = device.load_xclbin(xclbin);

        std::cout << "Creating kernel handle dct_rgb...\n";
        xrt::kernel krnl{device, uuid, "dct_rgb"};

        // ----------------------------------------------------
        // 3) Create BOs and transfer input
        // ----------------------------------------------------
        size_t bytes = num_pixels * sizeof(pixel_t);

        xrt::bo bo_inR  {device, bytes, krnl.group_id(0)};
        xrt::bo bo_inG  {device, bytes, krnl.group_id(1)};
        xrt::bo bo_inB  {device, bytes, krnl.group_id(2)};
        xrt::bo bo_outR {device, bytes, krnl.group_id(3)};
        xrt::bo bo_outG {device, bytes, krnl.group_id(4)};
        xrt::bo bo_outB {device, bytes, krnl.group_id(5)};

        bo_inR.write(R.data());
        bo_inG.write(G.data());
        bo_inB.write(B.data());

        bo_inR.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_inG.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ----------------------------------------------------
        // 4) Run kernel (DCT) with timing
        // ----------------------------------------------------
        std::cout << "Running kernel for " << iterations << " iterations...\n";

        // Warm-up
        {
            auto run = krnl(bo_inR, bo_inG, bo_inB,
                            bo_outR, bo_outG, bo_outB,
                            W, H);
            run.wait();
        }

        double total_ms = 0.0;
        for (int it = 0; it < iterations; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();

            auto run = krnl(bo_inR, bo_inG, bo_inB,
                            bo_outR, bo_outG, bo_outB,
                            W, H);
            run.wait();

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms += ms;

            std::cout << "Iteration " << it << ": " << ms << " ms\n";
        }

        double avg_ms = total_ms / iterations;
        double pixels_per_s = (num_pixels * iterations) / (total_ms / 1000.0);
        std::cout << "Average kernel time: " << avg_ms << " ms\n";
        std::cout << "Effective throughput: " << pixels_per_s / 1e6 << " MPixels/s\n";

        // ----------------------------------------------------
        // 5) Read back FPGA DCT output
        // ----------------------------------------------------
        std::vector<pixel_t> Rdct(num_pixels), Gdct(num_pixels), Bdct(num_pixels);

        bo_outR.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_outG.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_outB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        bo_outR.read(Rdct.data());
        bo_outG.read(Gdct.data());
        bo_outB.read(Bdct.data());

        // ----------------------------------------------------
        // 6) CPU IDCT (per channel, 8×8 blocks)
        //    NOTE: We assume output pixels = coeff + 128
        // ----------------------------------------------------
        std::vector<pixel_t> Rout(num_pixels), Gout(num_pixels), Bout(num_pixels);
        float blk[8][8], rec[8][8];

        auto cpu_idct = [&](const std::vector<pixel_t>& coeffs,
                            std::vector<pixel_t>& out)
        {
            for (int by = 0; by < H; by += 8) {
                for (int bx = 0; bx < W; bx += 8) {

                    // Load 8×8 block of "coefficients"
                    for (int u = 0; u < 8; u++) {
                        for (int v = 0; v < 8; v++) {
                            int gx = bx + v;
                            int gy = by + u;
                            float cval = 0.0f;
                            if (gx < W && gy < H) {
                                // assume stored as coeff + 128
                                cval = float(coeffs[gy * W + gx]) - 128.0f;
                            }
                            blk[u][v] = cval;
                        }
                    }

                    // IDCT
                    idct_block(blk, rec);

                    // Store reconstructed pixels
                    for (int u = 0; u < 8; u++) {
                        for (int v = 0; v < 8; v++) {
                            int gx = bx + v;
                            int gy = by + u;
                            if (gx < W && gy < H) {
                                float val = rec[u][v];
                                if (val < 0.0f)   val = 0.0f;
                                if (val > 255.0f) val = 255.0f;
                                out[gy * W + gx] = static_cast<pixel_t>(std::lround(val));
                            }
                        }
                    }
                }
            }
        };

        cpu_idct(Rdct, Rout);
        cpu_idct(Gdct, Gout);
        cpu_idct(Bdct, Bout);

        // ----------------------------------------------------
        // 7) Merge channels → reconstructed RGB image
        // ----------------------------------------------------
        std::vector<pixel_t> rgb_rec(num_pixels * 3);
        for (size_t i = 0; i < num_pixels; ++i) {
            rgb_rec[3*i+0] = Rout[i];
            rgb_rec[3*i+1] = Gout[i];
            rgb_rec[3*i+2] = Bout[i];
        }

        // ----------------------------------------------------
        // 8) PSNR (ONLY in spatial domain, after IDCT)
        // ----------------------------------------------------
        double psnr = compute_psnr(rgb_orig, rgb_rec);
        std::cout << "\n=== Spatial-domain PSNR (after IDCT) ===\n";
        std::cout << "PSNR = " << psnr << " dB\n";

        // ----------------------------------------------------
        // 9) Save reconstructed image
        // ----------------------------------------------------
        if (!stbi_write_png(output_path.c_str(), W, H, 3,
                            rgb_rec.data(), W * 3))
        {
            std::cerr << "ERROR: Failed to write " << output_path << "\n";
        } else {
            std::cout << "Wrote reconstructed image: " << output_path << "\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}
