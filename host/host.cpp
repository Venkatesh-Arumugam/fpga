#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <iomanip>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cassert>
#include <chrono>

#include "jpeg_cpu.hpp"

using std::vector;
using std::cout;
using std::cerr;
using std::endl;

// Performance metrics structure
struct PerfMetrics {
    double load_time_ms;
    double kernel_time_ms;
    double readback_time_ms;
    double total_fpga_time_ms;
    double cpu_dct_time_ms;
    double throughput_mpixels_per_sec;
    double throughput_blocks_per_sec;
    double speedup;
};

// Compression metrics structure
struct CompressionMetrics {
    size_t input_size_bytes;
    size_t output_size_bytes;
    size_t rle_size_bytes;
    double compression_ratio;
    double bits_per_pixel;
    int zero_coeffs;
    int nonzero_coeffs;
    double sparsity_percent;
};

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

// Full JPEG-style block pipeline
void jpeg_block_pipeline(const coeff_t blk_coeff_in[8][8], pixel_t blk_recon[8][8])
{
    coeff_t q_blk[8][8], dq_blk[8][8];

    quant_block(blk_coeff_in, q_blk);

    vector<coeff_t> zz;
    zigzag_block(q_blk, zz);

    vector<std::pair<coeff_t,int>> rle;
    rle_encode(zz, rle);

    vector<coeff_t> zz2;
    rle_decode(rle, zz2);
    zz2.resize(64);

    coeff_t q_blk2[8][8];
    inv_zigzag_block(zz2, q_blk2);

    dequant_block(q_blk2, dq_blk);

    idct_block_cpu(dq_blk, blk_recon);
}

// Calculate compression metrics
CompressionMetrics calculate_compression(
    const vector<coeff_t>& coeffs_R,
    const vector<coeff_t>& coeffs_G,
    const vector<coeff_t>& coeffs_B,
    int width, int height)
{
    CompressionMetrics metrics;

    // Input size (original pixels)
    metrics.input_size_bytes = width * height * 3; // RGB pixels

    // Count zero/nonzero coefficients
    metrics.zero_coeffs = 0;
    metrics.nonzero_coeffs = 0;

    size_t total_rle_pairs = 0;

    for (int by = 0; by < height; by += 8) {
        for (int bx = 0; bx < width; bx += 8) {
            // Process each channel
            for (int ch = 0; ch < 3; ch++) {
                const vector<coeff_t>* coeff_vec = (ch == 0) ? &coeffs_R :
                                                   (ch == 1) ? &coeffs_G : &coeffs_B;

                coeff_t blk[8][8];
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) {
                        int gx = bx + x;
                        int gy = by + y;
                        if (gx < width && gy < height) {
                            int idx = gy * width + gx;
                            blk[y][x] = (*coeff_vec)[idx];
                        } else {
                            blk[y][x] = 0;
                        }
                    }
                }

                // Quantize and count
                coeff_t q_blk[8][8];
                quant_block(blk, q_blk);

                vector<coeff_t> zz;
                zigzag_block(q_blk, zz);

                for (auto val : zz) {
                    if (val == 0) metrics.zero_coeffs++;
                    else metrics.nonzero_coeffs++;
                }

                // Calculate RLE size
                vector<std::pair<coeff_t,int>> rle;
                rle_encode(zz, rle);
                total_rle_pairs += rle.size();
            }
        }
    }

    // RLE encoding: each pair is (value, run_length)
    // Estimate: 2 bytes for value + 1 byte for run length = 3 bytes per pair
    metrics.rle_size_bytes = total_rle_pairs * 3;

    metrics.output_size_bytes = metrics.rle_size_bytes;
    metrics.compression_ratio = (double)metrics.input_size_bytes / metrics.output_size_bytes;
    metrics.bits_per_pixel = (double)(metrics.output_size_bytes * 8) / (width * height * 3);

    int total_coeffs = metrics.zero_coeffs + metrics.nonzero_coeffs;
    metrics.sparsity_percent = (double)metrics.zero_coeffs / total_coeffs * 100.0;

    return metrics;
}

// Print performance report
void print_performance_report(const PerfMetrics& perf, int width, int height)
{
    cout << "\n========================================\n";
    cout << "       PERFORMANCE METRICS\n";
    cout << "========================================\n";
    cout << "Image size: " << width << " x " << height
         << " (" << (width*height/1e6) << " MP)\n";
    cout << "Total blocks: " << ((width+7)/8) * ((height+7)/8) << "\n\n";

    cout << "FPGA Timing:\n";
    cout << "  Data load:      " << std::fixed << std::setprecision(3)
         << perf.load_time_ms << " ms\n";
    cout << "  Kernel exec:    " << perf.kernel_time_ms << " ms\n";
    cout << "  Data readback:  " << perf.readback_time_ms << " ms\n";
    cout << "  Total FPGA:     " << perf.total_fpga_time_ms << " ms\n\n";



    cout << "Throughput:\n";
    cout << "  FPGA:           " << std::setprecision(2)
         << perf.throughput_mpixels_per_sec << " MP/s\n";
    cout << "  FPGA:           " << std::setprecision(0)
         << perf.throughput_blocks_per_sec << " blocks/s\n\n";

   
    cout << "========================================\n";
}

// Print compression report
void print_compression_report(const CompressionMetrics& comp)
{
    cout << "\n========================================\n";
    cout << "       COMPRESSION METRICS\n";
    cout << "========================================\n";
    cout << "Input size (raw):     " << (comp.input_size_bytes/1024.0/1024.0)
         << " MB (" << comp.input_size_bytes << " bytes)\n";
    cout << "Output size (RLE):    " << (comp.output_size_bytes/1024.0/1024.0)
         << " MB (" << comp.output_size_bytes << " bytes)\n\n";

    cout << "Compression ratio:    " << std::fixed << std::setprecision(2)
         << comp.compression_ratio << ":1\n";
    cout << "Bits per pixel:       " << std::setprecision(3)
         << comp.bits_per_pixel << " bpp\n\n";

    cout << "Coefficient sparsity:\n";
    cout << "  Zero coeffs:        " << comp.zero_coeffs
         << " (" << std::setprecision(1) << comp.sparsity_percent << "%)\n";
    cout << "  Non-zero coeffs:    " << comp.nonzero_coeffs
         << " (" << std::setprecision(1) << (100.0 - comp.sparsity_percent) << "%)\n";
    cout << "========================================\n";
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

    PerfMetrics perf;

    // Time data transfer to FPGA
    auto t_start = std::chrono::high_resolution_clock::now();
    bo_inR.write(R.data());
    bo_inG.write(G.data());
    bo_inB.write(B.data());
    bo_inR.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inG.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t_load = std::chrono::high_resolution_clock::now();
    perf.load_time_ms = std::chrono::duration<double, std::milli>(t_load - t_start).count();

    cout << "Running FPGA DCT...\n";

    // Time kernel execution
    auto t_kernel_start = std::chrono::high_resolution_clock::now();
    auto run = kernel(bo_inR, bo_inG, bo_inB,
                      bo_outR, bo_outG, bo_outB,
                      w, h);
    run.wait();
    auto t_kernel_end = std::chrono::high_resolution_clock::now();
    perf.kernel_time_ms = std::chrono::duration<double, std::milli>(t_kernel_end - t_kernel_start).count();

    cout << "Kernel finished.\n";

    // Time data transfer from FPGA
    auto t_read_start = std::chrono::high_resolution_clock::now();
    vector<coeff_t> Rcoef_fpga(w*h), Gcoef_fpga(w*h), Bcoef_fpga(w*h);
    bo_outR.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outG.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outR.read(Rcoef_fpga.data());
    bo_outG.read(Gcoef_fpga.data());
    bo_outB.read(Bcoef_fpga.data());
    auto t_read_end = std::chrono::high_resolution_clock::now();
    perf.readback_time_ms = std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();

    perf.total_fpga_time_ms = perf.load_time_ms + perf.kernel_time_ms + perf.readback_time_ms;

    // ------------------ CPU golden DCT (for comparison) ------------------
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    vector<coeff_t> Rcoef_cpu, Gcoef_cpu, Bcoef_cpu;
    cpu_dct_image(R, w, h, Rcoef_cpu);
    cpu_dct_image(G, w, h, Gcoef_cpu);
    cpu_dct_image(B, w, h, Bcoef_cpu);
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    perf.cpu_dct_time_ms = std::chrono::duration<double, std::milli>(t_cpu_end - t_cpu_start).count();

    // Calculate performance metrics
    double mpixels = (w * h) / 1e6;
    int num_blocks = ((w+7)/8) * ((h+7)/8);

    perf.throughput_mpixels_per_sec = mpixels / (perf.kernel_time_ms / 1000.0);
    perf.throughput_blocks_per_sec = num_blocks / (perf.kernel_time_ms / 1000.0);
    perf.speedup = perf.cpu_dct_time_ms / perf.kernel_time_ms;

    // Compare raw coefficients
    long diff_count = 0;
    for (int i = 0; i < w*h; i++) {
        if (Rcoef_fpga[i] != Rcoef_cpu[i]) diff_count++;
        if (Gcoef_fpga[i] != Gcoef_cpu[i]) diff_count++;
        if (Bcoef_fpga[i] != Bcoef_cpu[i]) diff_count++;
    }
    cout << "\nCoefficient mismatches: " << diff_count << " / " << (w*h*3) << "\n";

    // ------------------ Calculate compression metrics ------------------
    CompressionMetrics comp = calculate_compression(Rcoef_fpga, Gcoef_fpga, Bcoef_fpga, w, h);

    // ------------------ JPEG-style pipeline per block ------------------
    vector<pixel_t> R_recon(w*h), G_recon(w*h), B_recon(w*h);
    pixel_t blk_recon[8][8];

    for (int by = 0; by < h; by += 8) {
        for (int bx = 0; bx < w; bx += 8) {
            // R channel
            coeff_t blkR_fpga[8][8];
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int gx = bx + x;
                    int gy = by + y;
                    if (gx < w && gy < h) {
                        int idx = gy * w + gx;
                        blkR_fpga[y][x] = Rcoef_fpga[idx];
                    } else {
                        blkR_fpga[y][x] = 0;
                    }
                }
            }
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

            // G channel
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

            // B channel
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

    // ------------------ PSNR ------------------
    double psnr_R = compute_psnr_channel(R, R_recon);
    double psnr_G = compute_psnr_channel(G, G_recon);
    double psnr_B = compute_psnr_channel(B, B_recon);
    double psnr_avg = (psnr_R + psnr_G + psnr_B) / 3.0;

    cout << "\n=== PSNR after JPEG-style pipeline ===\n";
    cout << "R: " << std::fixed << std::setprecision(2) << psnr_R << " dB\n";
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

    // ------------------ Print reports ------------------
    print_performance_report(perf, w, h);
    print_compression_report(comp);

    // Summary CSV line for easy comparison
    cout << "\n=== CSV Summary ===\n";
    cout << "Config,Width,Height,LoadMS,KernelMS,ReadMS,TotalMS,CPUMS,Speedup,MP/s,Blocks/s,";
    cout << "InputMB,OutputMB,CompRatio,BPP,Sparsity%,PSNR\n";
    cout << xclbin_file << ","
         << w << "," << h << ","
         << perf.load_time_ms << ","
         << perf.kernel_time_ms << ","
         << perf.readback_time_ms << ","
         << perf.total_fpga_time_ms << ","
         << perf.cpu_dct_time_ms << ","
         << perf.speedup << ","
         << perf.throughput_mpixels_per_sec << ","
         << perf.throughput_blocks_per_sec << ","
         << (comp.input_size_bytes/1024.0/1024.0) << ","
         << (comp.output_size_bytes/1024.0/1024.0) << ","
         << comp.compression_ratio << ","
         << comp.bits_per_pixel << ","
         << comp.sparsity_percent << ","
         << psnr_avg << "\n";

    return 0;
}
