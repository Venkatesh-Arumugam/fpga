#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cassert>

#include "jpeg_cpu.hpp"   // from above

using std::vector;
using std::cout;
using std::cerr;
using std::endl;

// Helper: process all 8x8 blocks on CPU to get DCT coefficients
void cpu_dct_image(const vector<pixel_t> &chan,
                   int width, int height,
                   vector<coeff_t> &coeff_out)
{
    coeff_out.resize(width * height);
    pixel_t blk_in[8][8];
    coeff_t blk_out[8][8];

    for (int by = 0; by < height; by += 8) {
        for (int bx = 0; bx < width; bx += 8) {

            // load block
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    int idx = gy * width + gx;
                    if (gx < width && gy < height)
                        blk_in[y][x] = chan[idx];
                    else
                        blk_in[y][x] = 0;
                }
            }

            dct_block_cpu(blk_in, blk_out);

            // store coefficients in same layout as FPGA
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < width && gy < height) {
                        int idx = gy * width + gx;
                        coeff_out[idx] = blk_out[y][x];
                    }
                }
            }
        }
    }
}

// Helper: full JPEG-style block pipeline using coeffs
// (Quant -> Zigzag -> RLE -> de-RLE -> invZigzag -> Dequant -> IDCT)
void jpeg_block_pipeline(
    const coeff_t blk_coeff_in[8][8],
    pixel_t blk_recon[8][8])
{
    coeff_t q_blk[8][8], dq_blk[8][8];

    // Quant
    quant_block(blk_coeff_in, q_blk);

    // Zigzag
    vector<coeff_t> zz;
    zigzag_block(q_blk, zz);

    // RLE
    vector<std::pair<coeff_t,int>> rle;
    rle_encode(zz, rle);

    // Decode RLE
    vector<coeff_t> zz2;
    rle_decode(rle, zz2);
    zz2.resize(64); // safety

    // inverse zigzag
    coeff_t q_blk2[8][8];
    inv_zigzag_block(zz2, q_blk2);

    // Dequant
    dequant_block(q_blk2, dq_blk);

    // IDCT
    idct_block_cpu(dq_blk, blk_recon);
}

// Compare two RLE streams bit-for-bit
bool rle_equal(const vector<std::pair<coeff_t,int>> &a,
               const vector<std::pair<coeff_t,int>> &b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].first != b[i].first || a[i].second != b[i].second) return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        cerr << "Usage: " << argv[0]
             << " <xclbin> <input.png> <output.png>\n";
        return 1;
    }

    std::string xclbin_file = argv[1];
    std::string input_png   = argv[2];
    std::string output_png  = argv[3];

    // ------------------ Load image ------------------
    int w, h, ch;
    unsigned char* img = stbi_load(input_png.c_str(), &w, &h, &ch, 3);
    if (!img) {
        cerr << "ERROR: Cannot load input image\n";
        return 1;
    }
    cout << "Loaded " << w << "x" << h << " (3 channels)\n";

    vector<pixel_t> R(w*h), G(w*h), B(w*h);
    for (int i = 0; i < w*h; i++) {
        R[i] = img[3*i + 0];
        G[i] = img[3*i + 1];
        B[i] = img[3*i + 2];
    }
    stbi_image_free(img);

    // ------------------ FPGA setup ------------------
    cout << "Opening device 0...\n";
    xrt::device device(0);

    cout << "Loading xclbin: " << xclbin_file << "\n";
    auto uuid = device.load_xclbin(xclbin_file);

    cout << "Opening kernel 'dct_accel'...\n";
    xrt::kernel kernel(device, uuid, "dct_accel");

    size_t coeff_bytes = size_t(w) * h * sizeof(coeff_t);
    size_t pixel_bytes = size_t(w) * h * sizeof(pixel_t);

    auto bo_inR  = xrt::bo(device, pixel_bytes, xrt::bo::flags::normal, kernel.group_id(0));
    auto bo_inG  = xrt::bo(device, pixel_bytes, xrt::bo::flags::normal, kernel.group_id(1));
    auto bo_inB  = xrt::bo(device, pixel_bytes, xrt::bo::flags::normal, kernel.group_id(2));
    auto bo_outR = xrt::bo(device, coeff_bytes, xrt::bo::flags::normal, kernel.group_id(3));
    auto bo_outG = xrt::bo(device, coeff_bytes, xrt::bo::flags::normal, kernel.group_id(4));
    auto bo_outB = xrt::bo(device, coeff_bytes, xrt::bo::flags::normal, kernel.group_id(5));

    bo_inR.write(R.data());
    bo_inG.write(G.data());
    bo_inB.write(B.data());
    bo_inR.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inG.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    cout << "Running FPGA DCT...\n";
    auto run = kernel(bo_inR, bo_inG, bo_inB,
                      bo_outR, bo_outG, bo_outB,
                      w, h);
    run.wait();
    cout << "Kernel finished.\n";

    vector<coeff_t> Rcoef_fpga(w*h), Gcoef_fpga(w*h), Bcoef_fpga(w*h);
    bo_outR.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outG.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outR.read(Rcoef_fpga.data());
    bo_outG.read(Gcoef_fpga.data());
    bo_outB.read(Bcoef_fpga.data());

    // ------------------ CPU golden DCT ------------------
    vector<coeff_t> Rcoef_cpu, Gcoef_cpu, Bcoef_cpu;
    cpu_dct_image(R, w, h, Rcoef_cpu);
    cpu_dct_image(G, w, h, Gcoef_cpu);
    cpu_dct_image(B, w, h, Bcoef_cpu);

    // Compare raw coefficients (for debugging)
    long diff_count = 0;
    for (int i = 0; i < w*h; i++) {
        if (Rcoef_fpga[i] != Rcoef_cpu[i]) diff_count++;
        if (Gcoef_fpga[i] != Gcoef_cpu[i]) diff_count++;
        if (Bcoef_fpga[i] != Bcoef_cpu[i]) diff_count++;
    }
    cout << "Coefficient mismatches (R+G+B total entries): " << diff_count << "\n";

    // ------------------ JPEG-style pipeline per block ------------------
    vector<pixel_t> R_recon(w*h), G_recon(w*h), B_recon(w*h);

    pixel_t blk_recon[8][8];
    coeff_t blk_fpga[8][8];

    long rle_mismatch_blocks = 0;

    for (int by = 0; by < h; by += 8) {
        for (int bx = 0; bx < w; bx += 8) {

            // Build 8x8 block of coeffs from FPGA (e.g., R channel)
            // We'll do the JPEG pipeline for R,G,B same way
            // Also do RLE comparison between CPU and FPGA path for each block.

            // -------- R channel block --------
            coeff_t blkR_fpga[8][8], blkR_cpu[8][8];

            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < w && gy < h) {
                        int idx = gy * w + gx;
                        blkR_fpga[y][x] = Rcoef_fpga[idx];
                        blkR_cpu[y][x]  = Rcoef_cpu[idx];
                    } else {
                        blkR_fpga[y][x] = 0;
                        blkR_cpu[y][x]  = 0;
                    }
                }
            }

            // Quant + zigzag + RLE on FPGA coeffs
            coeff_t q_fpga[8][8];
            quant_block(blkR_fpga, q_fpga);
            vector<coeff_t> zz_fpga;
            zigzag_block(q_fpga, zz_fpga);
            vector<std::pair<coeff_t,int>> rle_fpga;
            rle_encode(zz_fpga, rle_fpga);

            // Same on CPU coeffs
            coeff_t q_cpu[8][8];
            quant_block(blkR_cpu, q_cpu);
            vector<coeff_t> zz_cpu;
            zigzag_block(q_cpu, zz_cpu);
            vector<std::pair<coeff_t,int>> rle_cpu;
            rle_encode(zz_cpu, rle_cpu);

            if (!rle_equal(rle_fpga, rle_cpu)) {
                rle_mismatch_blocks++;
            }

            // Now do complete JPEG pipeline for R using FPGA coeffs
            jpeg_block_pipeline(blkR_fpga, blk_recon);

            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < w && gy < h) {
                        int idx = gy * w + gx;
                        R_recon[idx] = blk_recon[y][x];
                    }
                }
            }

            // -------- Repeat for G --------
            coeff_t blkG_fpga[8][8];
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < w && gy < h) {
                        int idx = gy * w + gx;
                        blkG_fpga[y][x] = Gcoef_fpga[idx];
                    } else {
                        blkG_fpga[y][x] = 0;
                    }
                }
            }
            jpeg_block_pipeline(blkG_fpga, blk_recon);
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < w && gy < h) {
                        int idx = gy * w + gx;
                        G_recon[idx] = blk_recon[y][x];
                    }
                }
            }

            // -------- Repeat for B --------
            coeff_t blkB_fpga[8][8];
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < w && gy < h) {
                        int idx = gy * w + gx;
                        blkB_fpga[y][x] = Bcoef_fpga[idx];
                    } else {
                        blkB_fpga[y][x] = 0;
                    }
                }
            }
            jpeg_block_pipeline(blkB_fpga, blk_recon);
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < w && gy < h) {
                        int idx = gy * w + gx;
                        B_recon[idx] = blk_recon[y][x];
                    }
                }
            }
        }
    }

    cout << "RLE mismatching blocks (R channel): " << rle_mismatch_blocks << "\n";

    // ------------------ PSNR ------------------
    double psnr_R = compute_psnr_channel(R, R_recon);
    double psnr_G = compute_psnr_channel(G, G_recon);
    double psnr_B = compute_psnr_channel(B, B_recon);
    double psnr_avg = (psnr_R + psnr_G + psnr_B) / 3.0;

    cout << "\n=== PSNR after JPEG-style pipeline (FPGA DCT) ===\n";
    cout << "R: " << psnr_R << " dB\n";
    cout << "G: " << psnr_G << " dB\n";
    cout << "B: " << psnr_B << " dB\n";
    cout << "Avg: " << psnr_avg << " dB\n";

    // ------------------ Write reconstructed image ------------------
    vector<unsigned char> out_img(w*h*3);
    for (int i = 0; i < w*h; i++) {
        out_img[3*i + 0] = R_recon[i];
        out_img[3*i + 1] = G_recon[i];
        out_img[3*i + 2] = B_recon[i];
    }

    if (!stbi_write_png(output_png.c_str(), w, h, 3,
                        out_img.data(), w*3)) {
        cerr << "ERROR: Failed to write output PNG\n";
        return 1;
    }
    cout << "Wrote reconstructed image: " << output_png << "\n";

    return 0;
}
