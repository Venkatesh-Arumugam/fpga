// host.cpp – CPU side for DCT8x8_color on U280
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
// Download these headers and put them in your project.
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

// RGB (0–255) -> YCbCr (0–255) – BT.601-like
static void rgb_to_ycbcr(
    const std::vector<pixel_t> &rgb,
    std::vector<pixel_t>       &Y,
    std::vector<pixel_t>       &Cb,
    std::vector<pixel_t>       &Cr)
{
    size_t n = Y.size();
    for (size_t i = 0; i < n; ++i) {
        float r = rgb[3 * i + 0];
        float g = rgb[3 * i + 1];
        float b = rgb[3 * i + 2];

        float y  =  0.2990f * r + 0.5870f * g + 0.1140f * b;
        float cb = -0.1687f * r - 0.3313f * g + 0.5000f * b + 128.0f;
        float cr =  0.5000f * r - 0.4187f * g - 0.0813f * b + 128.0f;

        int iy  = std::lround(y);
        int icb = std::lround(cb);
        int icr = std::lround(cr);

        if (iy  <   0) iy  = 0;   if (iy  > 255) iy  = 255;
        if (icb <   0) icb = 0;   if (icb > 255) icb = 255;
        if (icr <   0) icr = 0;   if (icr > 255) icr = 255;

        Y[i]  = static_cast<pixel_t>(iy);
        Cb[i] = static_cast<pixel_t>(icb);
        Cr[i] = static_cast<pixel_t>(icr);
    }
}

// YCbCr -> RGB (clamped 0–255)
static void ycbcr_to_rgb(
    const std::vector<pixel_t> &Y,
    const std::vector<pixel_t> &Cb,
    const std::vector<pixel_t> &Cr,
    std::vector<pixel_t>       &rgb)
{
    size_t n = Y.size();
    for (size_t i = 0; i < n; ++i) {
        float y  = Y[i];
        float cb = Cb[i] - 128.0f;
        float cr = Cr[i] - 128.0f;

        float r = y + 1.402f    * cr;
        float g = y - 0.344136f * cb - 0.714136f * cr;
        float b = y + 1.772f    * cb;

        int ir = std::lround(r);
        int ig = std::lround(g);
        int ib = std::lround(b);

        if (ir < 0) ir = 0; if (ir > 255) ir = 255;
        if (ig < 0) ig = 0; if (ig > 255) ig = 255;
        if (ib < 0) ib = 0; if (ib > 255) ib = 255;

        rgb[3 * i + 0] = static_cast<pixel_t>(ir);
        rgb[3 * i + 1] = static_cast<pixel_t>(ig);
        rgb[3 * i + 2] = static_cast<pixel_t>(ib);
    }
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
        // 1) Load image
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

        // Planar Y, Cb, Cr
        std::vector<pixel_t> Y(num_pixels), Cb(num_pixels), Cr(num_pixels);
        rgb_to_ycbcr(rgb_in, Y, Cb, Cr);

        // Output planar buffers
        std::vector<pixel_t> Y_out(num_pixels), Cb_out(num_pixels), Cr_out(num_pixels);

        // --------------------------------------------------------------------
        // 2) Open device and load xclbin
        // --------------------------------------------------------------------
        std::cout << "Opening device 0...\n";
        xrt::device device{0};

        std::cout << "Loading xclbin: " << xclbin_path << "\n";
        auto xclbin = xrt::xclbin(xclbin_path);
        auto uuid   = device.load_xclbin(xclbin);

        std::cout << "Creating kernel handle dct8x8_color...\n";
        xrt::kernel krnl{device, uuid, "dct8x8_color"};

        // --------------------------------------------------------------------
        // 3) Create buffers and copy host data
        // --------------------------------------------------------------------
        size_t bytes = num_pixels * sizeof(pixel_t);

        xrt::bo bo_inY  {device, bytes, krnl.group_id(0)};
        xrt::bo bo_inCb {device, bytes, krnl.group_id(1)};
        xrt::bo bo_inCr {device, bytes, krnl.group_id(2)};
        xrt::bo bo_outY {device, bytes, krnl.group_id(3)};
        xrt::bo bo_outCb{device, bytes, krnl.group_id(4)};
        xrt::bo bo_outCr{device, bytes, krnl.group_id(5)};

        bo_inY.write(Y.data());
        bo_inCb.write(Cb.data());
        bo_inCr.write(Cr.data());

        bo_inY.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_inCb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_inCr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // --------------------------------------------------------------------
        // 4) Run kernel (benchmark base version)
        // --------------------------------------------------------------------
        std::cout << "Running kernel for " << iterations << " iterations...\n";

        // Warm-up
        {
            auto run = krnl(bo_inY, bo_inCb, bo_inCr,
                            bo_outY, bo_outCb, bo_outCr,
                            width, height);
            run.wait();
        }

        double total_ms = 0.0;
        for (int it = 0; it < iterations; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();

            auto run = krnl(bo_inY, bo_inCb, bo_inCr,
                            bo_outY, bo_outCb, bo_outCr,
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
        bo_outY.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_outCb.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_outCr.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        bo_outY.read(Y_out.data());
        bo_outCb.read(Cb_out.data());
        bo_outCr.read(Cr_out.data());

        // --------------------------------------------------------------------
        // 6) Convert back to RGB and save image
        // --------------------------------------------------------------------
        std::vector<pixel_t> rgb_out(num_pixels * 3);
        ycbcr_to_rgb(Y_out, Cb_out, Cr_out, rgb_out);

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
