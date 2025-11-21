#include <iostream>
#include <vector>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

static const float Cmat[8][8] = {
    {0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553},
    {0.490393, 0.415735, 0.277785, 0.097545,-0.097545,-0.277785,-0.415735,-0.490393},
    {0.461940, 0.191342,-0.191342,-0.461940,-0.461940,-0.191342, 0.191342, 0.461940},
    {0.415735,-0.097545,-0.490393,-0.277785, 0.277785, 0.490393, 0.097545,-0.415735},
    {0.353553,-0.353553,-0.353553, 0.353553, 0.353553,-0.353553,-0.353553, 0.353553},
    {0.277785,-0.490393, 0.097545, 0.415735,-0.415735,-0.097545, 0.490393,-0.277785},
    {0.191342,-0.461940, 0.461940,-0.191342,-0.191342, 0.461940,-0.461940, 0.191342},
    {0.097545,-0.277785, 0.415735,-0.490393, 0.490393,-0.415735, 0.277785,-0.097545}
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

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./cpu_idct input_dct.png output.png\n";
        return 1;
    }

    string inPath = argv[1];
    string outPath = argv[2];

    int W, H, C;
    uint8_t* img = stbi_load(inPath.c_str(), &W, &H, &C, 3);
    if (!img) {
        cout << "[ERROR] Failed to load " << inPath << endl;
        return 1;
    }

    cout << "[INFO] Loaded DCT image: " << W << "x" << H << "\n";

    vector<uint8_t> R(W*H), G(W*H), B(W*H);
    vector<uint8_t> outRGB(W*H*3);

    for (int i = 0; i < W*H; i++) {
        R[i] = img[3*i+0];
        G[i] = img[3*i+1];
        B[i] = img[3*i+2];
    }
    stbi_image_free(img);

    float blk[8][8], rec[8][8];

    auto process_channel = [&](vector<uint8_t>& Cin, int chID) {
        vector<uint8_t> Cout(W*H);

        for (int by = 0; by < H; by += 8) {
            for (int bx = 0; bx < W; bx += 8) {

                for (int u = 0; u < 8; u++)
                    for (int v = 0; v < 8; v++) {
                        int gx = bx + v, gy = by + u;
                        float coeff = 0;
                        if (gx < W && gy < H)
                            coeff = (float)Cin[gy*W + gx] - 128.0f;
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
                            Cout[gy*W + gx] = (uint8_t)val;
                        }
                    }
            }
        }

        for (int i = 0; i < W*H; i++) {
            outRGB[3*i + chID] = Cout[i];
        }
    };

    process_channel(R, 0);
    process_channel(G, 1);
    process_channel(B, 2);

    stbi_write_png(outPath.c_str(), W, H, 3, outRGB.data(), W*3);

    cout << "[INFO] Wrote reconstructed image: " << outPath << endl;
    return 0;
}
