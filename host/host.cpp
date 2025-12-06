// host.cpp – CPU side for dct_rgb on U280 (pure RGB + PSNR)
#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <cmath>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "experimental/xrt_xclbin.h"

// ---- stb_image / stb_image_write ---------------------------
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using pixel_t = uint8_t;

// Simple check helper
static void check(bool cond, const std::string &msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

// -------------------------------------------------------
// PSNR utilities
// -------------------------------------------------------
double compute_mse(const std::vector<uint8_t>& a,
                   const std::vector<uint8_t>& b)
{
    double mse = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; i++) {
        double d = double(a[i]) - double(b[i]);
        mse += d * d;
    }
    return mse / double(n);
}

double compute_psnr(double mse)
{
    if (mse == 0.0) return 100.0; // identical images
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

int main(int argc, char *argv[]) {
    try {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0]
                      << " <xclbin> <input.png> <output.png> [iterations]\n";
            return 1;
        }

        std::string xclbin_path = argv[1];
        std::string input_path  = argv[2];
        std::string output_path = argv[3];
        int iterations = (argc > 4) ? std::stoi(argv[4]) : 10;

        // --------------------------------------------------------------------
        // 1) Load image (RGB)
        // --------------------------------------------------------------------
        int width = 0, height = 0, channels = 0;
        unsigned char *img_data = stbi_load(input_path.c_str(),
                                            &width, &height,
                                            &channels, 3); // force 3 channels
        check(img_data != nullptr, "Failed to load input image");

        size_t num_pixels = static_cast<size_t>(width) * height;
        std::vector<pixel_t> rgb_in(num_pixels * 3);
        for (size_t i = 0; i < num_pixels * 3; ++i) {
            rgb_in[i] = img_data[i];
        }
        stbi_image_free(img_data);

        std::cout << "Loaded " << input_path << " (" << width
                  << "x" << height << ", 3ch)\n";

        // Split into planar R/G/B
        std::vector<pixel_t> R(num_pixels), G(num_pixels), B(num_pixels);
        for (size_t i = 0; i < num_pixels; ++i) {
            R[i] = rgb_in[3 * i + 0];
            G[i] = rgb_in[3 * i + 1];
            B[i] = rgb_in[3 * i + 2];
        }

        // Output planar buffers
        std::vector<pixel_t> R_out(num_pixels), G_out(num_pixels), B_out(num_pixels);

        // --------------------------------------------------------------------
        // 2) Open device and load xclbin
        // --------------------------------------------------------------------
        std::cout << "Opening device 0...\n";
        xrt::device device{0};

        std::cout << "Loading xclbin: " << xclbin_path << "\n";
        auto xclbin = xrt::xclbin(xclbin_path);
        auto uuid   = device.load_xclbin(xclbin);

        std::cout << "Creating kernel handle dct_rgb...\n";
        xrt::kernel krnl{device, uuid, "dct_rgb"};

        // --------------------------------------------------------------------
        // 3) Create buffers and copy host data
        // --------------------------------------------------------------------
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

        // --------------------------------------------------------------------
        // 4) Run kernel (benchmark base version)
        // --------------------------------------------------------------------
        std::cout << "Running kernel for " << iterations << " iterations...\n";

        // Warm-up
        {
            auto run = krnl(bo_inR, bo_inG, bo_inB,
                            bo_outR, bo_outG, bo_outB,
                            width, height);
            run.wait();
        }

        double total_ms = 0.0;
        for (int it = 0; it < iterations; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();

            auto run = krnl(bo_inR, bo_inG, bo_inB,
                            bo_outR, bo_outG, bo_outB,
                            width, height);
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

        // --------------------------------------------------------------------
        // 5) Copy results back to host
        // --------------------------------------------------------------------
        bo_outR.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_outG.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_outB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        bo_outR.read(R_out.data());
        bo_outG.read(G_out.data());
        bo_outB.read(B_out.data());

        // --------------------------------------------------------------------
        // 6) Compute PSNR (per-channel + overall RGB)
        // --------------------------------------------------------------------
        std::cout << "\n=== PSNR Analysis (RGB) ===\n";

        // Original planar channels (R,G,B) are already in R, G, B
        double mseR = compute_mse(R, R_out);
        double mseG = compute_mse(G, G_out);
        double mseB = compute_mse(B, B_out);

        double psnrR = compute_psnr(mseR);
        double psnrG = compute_psnr(mseG);
        double psnrB = compute_psnr(mseB);

        std::cout << "R   : MSE = " << mseR << ", PSNR = " << psnrR << " dB\n";
        std::cout << "G   : MSE = " << mseG << ", PSNR = " << psnrG << " dB\n";
        std::cout << "B   : MSE = " << mseB << ", PSNR = " << psnrB << " dB\n";

        double mse_rgb_total = (mseR + mseG + mseB) / 3.0;
        double psnr_rgb_total = compute_psnr(mse_rgb_total);

        std::cout << "Overall RGB PSNR = " << psnr_rgb_total << " dB\n\n";

        // --------------------------------------------------------------------
        // 7) Recombine RGB and save image
        // --------------------------------------------------------------------
        std::vector<pixel_t> rgb_out(num_pixels * 3);
        for (size_t i = 0; i < num_pixels; ++i) {
            rgb_out[3 * i + 0] = R_out[i];
            rgb_out[3 * i + 1] = G_out[i];
            rgb_out[3 * i + 2] = B_out[i];
        }

        int stride = width * 3;
        if (!stbi_write_png(output_path.c_str(), width, height,
                            3, rgb_out.data(), stride)) {
            std::cerr << "Failed to write output image\n";
        } else {
            std::cout << "Wrote result to " << output_path << "\n";
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}
