// host.cpp
// Build: g++ host.cpp -o ../build/host.exe -O2 -std=c++17 \
//          -I. -I/opt/xilinx/xrt/include \
//          -L/opt/xilinx/xrt/lib -lxrt_coreutil

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>

using pixel_t = uint8_t;

// Helper: pad one channel to newW x newH with zeros (same as golden_encoder)
static void pad_channel(std::vector<uint8_t>& chan, int w, int h,
                        int newW, int newH)
{
    if (newW == w && newH == h) return;
    std::vector<uint8_t> out(newW * newH, 0);
    for (int r = 0; r < h; ++r) {
        std::memcpy(&out[r * newW], &chan[r * w], w);
    }
    chan.swap(out);
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <xclbin> <input.png> <output.png>\n";
        return 1;
    }

    std::string xclbin_file = argv[1];
    std::string input_png   = argv[2];
    std::string output_png  = argv[3];

    // ------------------------------------------------------------------
    // Load input image with STB as RGB
    // ------------------------------------------------------------------
    int w, h, ch;
    unsigned char* img = stbi_load(input_png.c_str(), &w, &h, &ch, 3);
    if (!img) {
        std::cerr << "ERROR: Failed to load image: " << input_png << "\n";
        return 1;
    }
    std::cout << "Loaded " << w << "x" << h << " (3 channels)\n";

    // Separate planes R,G,B
    std::vector<uint8_t> R(w * h), G(w * h), B(w * h);
    for (int i = 0; i < w * h; ++i) {
        R[i] = img[3 * i + 0];
        G[i] = img[3 * i + 1];
        B[i] = img[3 * i + 2];
    }
    stbi_image_free(img);

    // Pad to multiples of 8, same logic as golden_encoder.cpp
    int newW = (w + 7) & ~7;
    int newH = (h + 7) & ~7;
    pad_channel(R, w, h, newW, newH);
    pad_channel(G, w, h, newW, newH);
    pad_channel(B, w, h, newW, newH);

    std::cout << "Padded to " << newW << "x" << newH << "\n";

    size_t plane_size_bytes = static_cast<size_t>(newW) * newH * sizeof(pixel_t);

    std::cout << "Opening device 0...\n";
    auto device = xrt::device(0);
    
    std::cout << "Loading xclbin: " << xclbin_file << "\n";
    
    // Load xclbin (returns uuid)
    auto uuid = device.load_xclbin(xclbin_file);
    
    // Create kernel
    std::cout << "Opening kernel 'dct'...\n";
    auto kernel = xrt::kernel(device, uuid, "dct");
    // Kernel name must match the top function in HLS: "dct"
    std::cout << "Opening kernel 'dct'...\n";
    

    // ------------------------------------------------------------------
    // Allocate device buffers (M_AXI bindings depend on your HLS pragmas)
    // We assume arguments: (inR, inG, inB, outR, outG, outB, width, height)
    // ------------------------------------------------------------------
    auto bo_inR  = xrt::bo(device, plane_size_bytes,
                           xrt::bo::flags::normal, kernel.group_id(0));
    auto bo_inG  = xrt::bo(device, plane_size_bytes,
                           xrt::bo::flags::normal, kernel.group_id(1));
    auto bo_inB  = xrt::bo(device, plane_size_bytes,
                           xrt::bo::flags::normal, kernel.group_id(2));

    auto bo_outR = xrt::bo(device, plane_size_bytes,
                           xrt::bo::flags::normal, kernel.group_id(3));
    auto bo_outG = xrt::bo(device, plane_size_bytes,
                           xrt::bo::flags::normal, kernel.group_id(4));
    auto bo_outB = xrt::bo(device, plane_size_bytes,
                           xrt::bo::flags::normal, kernel.group_id(5));

    // ------------------------------------------------------------------
    // Copy host → device
    // ------------------------------------------------------------------
    bo_inR.write(R.data());
    bo_inG.write(G.data());
    bo_inB.write(B.data());

    bo_inR.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inG.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ------------------------------------------------------------------
    // Launch kernel
    // Signature assumed:
    //   void dct(const pixel_t* inR, const pixel_t* inG, const pixel_t* inB,
    //            pixel_t* outR, pixel_t* outG, pixel_t* outB,
    //            int width, int height)
    // ------------------------------------------------------------------
    std::cout << "Running kernel dct(" << newW << "x" << newH << ")...\n";

    auto run = kernel(bo_inR, bo_inG, bo_inB,
                      bo_outR, bo_outG, bo_outB,
                      newW, newH);

    run.wait();
    std::cout << "Kernel execution complete.\n";

    // ------------------------------------------------------------------
    // Copy device → host
    // ------------------------------------------------------------------
    std::vector<uint8_t> outR(newW * newH);
    std::vector<uint8_t> outG(newW * newH);
    std::vector<uint8_t> outB(newW * newH);

    bo_outR.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outG.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    bo_outR.read(outR.data());
    bo_outG.read(outG.data());
    bo_outB.read(outB.data());

    // ------------------------------------------------------------------
    // Merge back to RGB and crop to original w,h for writing
    // ------------------------------------------------------------------
    std::vector<uint8_t> out_img(static_cast<size_t>(w) * h * 3, 0);

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            int idx_padded = row * newW + col;
            int idx_out    = (row * w + col) * 3;

            out_img[idx_out + 0] = outR[idx_padded];
            out_img[idx_out + 1] = outG[idx_padded];
            out_img[idx_out + 2] = outB[idx_padded];
        }
    }

    // Write PNG with original width/height
    if (!stbi_write_png(output_png.c_str(), w, h, 3,
                        out_img.data(), w * 3)) {
        std::cerr << "ERROR: Failed to write output PNG: " << output_png << "\n";
        return 1;
    }

    std::cout << "Wrote output image: " << output_png << "\n";
    return 0;
}


