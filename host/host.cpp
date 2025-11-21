#include <iostream>
#include <vector>
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

using pixel_t = unsigned char;

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <xclbin> <input.png> <output.png>\n";
        return 1;
    }

    std::string xclbin_file = argv[1];
    std::string input_file  = argv[2];
    std::string output_file = argv[3];

    // -------------------------------------------------------------
    // Load input PNG using OpenCV (in BGR format)
    // -------------------------------------------------------------
    cv::Mat img = cv::imread(input_file, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "ERROR: Could not load input image " << input_file << "\n";
        return 1;
    }

    int width  = img.cols;
    int height = img.rows;

    // Convert BGR → RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // -------------------------------------------------------------
    // Split channels
    // -------------------------------------------------------------
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    // Flatten channels
    std::vector<pixel_t> R(channels[0].begin<pixel_t>(), channels[0].end<pixel_t>());
    std::vector<pixel_t> G(channels[1].begin<pixel_t>(), channels[1].end<pixel_t>());
    std::vector<pixel_t> B(channels[2].begin<pixel_t>(), channels[2].end<pixel_t>());

    size_t img_size = width * height * sizeof(pixel_t);

    // -------------------------------------------------------------
    // Open FPGA device
    // -------------------------------------------------------------
    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(xclbin_file);
    device.load_xclbin(xclbin);

    auto kernel = xrt::kernel(device, xclbin, "dct");

    // -------------------------------------------------------------
    // Allocate device memory buffers
    // -------------------------------------------------------------
    auto bo_inR  = xrt::bo(device, img_size, xrt::bo::flags::normal, kernel.group_id(0));
    auto bo_inG  = xrt::bo(device, img_size, xrt::bo::flags::normal, kernel.group_id(1));
    auto bo_inB  = xrt::bo(device, img_size, xrt::bo::flags::normal, kernel.group_id(2));

    auto bo_outR = xrt::bo(device, img_size, xrt::bo::flags::normal, kernel.group_id(3));
    auto bo_outG = xrt::bo(device, img_size, xrt::bo::flags::normal, kernel.group_id(4));
    auto bo_outB = xrt::bo(device, img_size, xrt::bo::flags::normal, kernel.group_id(5));

    // -------------------------------------------------------------
    // Copy inputs to device
    // -------------------------------------------------------------
    bo_inR.write(R.data());
    bo_inG.write(G.data());
    bo_inB.write(B.data());

    bo_inR.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inG.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // -------------------------------------------------------------
    // Launch kernel
    // -------------------------------------------------------------
    std::cout << "Launching kernel...\n";

    auto run = kernel(bo_inR, bo_inG, bo_inB,
                      bo_outR, bo_outG, bo_outB,
                      width, height);

    run.wait();

    std::cout << "Kernel execution complete.\n";

    // -------------------------------------------------------------
    // Read back output
    // -------------------------------------------------------------
    std::vector<pixel_t> outR(width * height);
    std::vector<pixel_t> outG(width * height);
    std::vector<pixel_t> outB(width * height);

    bo_outR.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outG.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    bo_outR.read(outR.data());
    bo_outG.read(outG.data());
    bo_outB.read(outB.data());

    // -------------------------------------------------------------
    // Merge channels and save PNG
    // -------------------------------------------------------------
    cv::Mat outRmat(height, width, CV_8UC1, outR.data());
    cv::Mat outGmat(height, width, CV_8UC1, outG.data());
    cv::Mat outBmat(height, width, CV_8UC1, outB.data());

    cv::Mat merged;
    std::vector<cv::Mat> out_channels = {outRmat, outGmat, outBmat};
    cv::merge(out_channels, merged);

    // Convert back to BGR for PNG saving
    cv::cvtColor(merged, merged, cv::COLOR_RGB2BGR);

    cv::imwrite(output_file, merged);

    std::cout << "Saved output image to: " << output_file << "\n";
    return 0;
}
