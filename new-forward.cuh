//tile_width is width of tile one kernel processes
#define TILE_WIDTH 32
//parameters used to determine large or small layer
#define LARGE_INPUT 6
#define SMALL_INPUT 1
#define LARGE_OUTPUT 16
#define SMALL_OUTPUT 6
#define MASK_WIDTH 5
#define SMALL_FLAG 1
#define LARGE_FLAG 2

#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
//#include <stdio.h>

namespace mxnet
{
namespace op
{

//constant memory for the mask(kernel)
__constant__ float mask_small[SMALL_OUTPUT*SMALL_INPUT*MASK_WIDTH*MASK_WIDTH];
__constant__ float mask_large[LARGE_OUTPUT*LARGE_INPUT*MASK_WIDTH*MASK_WIDTH];

//kernel for the smaller input layer call
__global__ void forward_kernel_small(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_GRID = (W_out-1) / TILE_WIDTH + 1;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    #define mask_small(i3, i2, i1, i0) mask_small[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int n, m, h, w;
    n = blockIdx.x;     //minibatch index
    m = blockIdx.y;     //output feature index

    //these two parameters need boundary check when referencing
    h = (blockIdx.z / W_GRID) * TILE_WIDTH + threadIdx.y;  //height index of output element in a particular output feature map
    w = (blockIdx.z % W_GRID) * TILE_WIDTH + threadIdx.x;  //width index of output element in a particular output feature map

    //loop unrolling
    float result = 0;
    if (h < H_out && w < W_out) {   //check if threadidx is in range of the output feature map
        result += x4d(n, 0, h+0, w+0) * mask_small(m, 0, 0, 0);
        result += x4d(n, 0, h+0, w+1) * mask_small(m, 0, 0, 1);
        result += x4d(n, 0, h+0, w+2) * mask_small(m, 0, 0, 2);
        result += x4d(n, 0, h+0, w+3) * mask_small(m, 0, 0, 3);
        result += x4d(n, 0, h+0, w+4) * mask_small(m, 0, 0, 4);

        result += x4d(n, 0, h+1, w+0) * mask_small(m, 0, 1, 0);
        result += x4d(n, 0, h+1, w+1) * mask_small(m, 0, 1, 1);
        result += x4d(n, 0, h+1, w+2) * mask_small(m, 0, 1, 2);
        result += x4d(n, 0, h+1, w+3) * mask_small(m, 0, 1, 3);
        result += x4d(n, 0, h+1, w+4) * mask_small(m, 0, 1, 4);

        result += x4d(n, 0, h+2, w+0) * mask_small(m, 0, 2, 0);
        result += x4d(n, 0, h+2, w+1) * mask_small(m, 0, 2, 1);
        result += x4d(n, 0, h+2, w+2) * mask_small(m, 0, 2, 2);
        result += x4d(n, 0, h+2, w+3) * mask_small(m, 0, 2, 3);
        result += x4d(n, 0, h+2, w+4) * mask_small(m, 0, 2, 4);

        result += x4d(n, 0, h+3, w+0) * mask_small(m, 0, 3, 0);
        result += x4d(n, 0, h+3, w+1) * mask_small(m, 0, 3, 1);
        result += x4d(n, 0, h+3, w+2) * mask_small(m, 0, 3, 2);
        result += x4d(n, 0, h+3, w+3) * mask_small(m, 0, 3, 3);
        result += x4d(n, 0, h+3, w+4) * mask_small(m, 0, 3, 4);

        result += x4d(n, 0, h+4, w+0) * mask_small(m, 0, 4, 0);
        result += x4d(n, 0, h+4, w+1) * mask_small(m, 0, 4, 1);
        result += x4d(n, 0, h+4, w+2) * mask_small(m, 0, 4, 2);
        result += x4d(n, 0, h+4, w+3) * mask_small(m, 0, 4, 3);
        result += x4d(n, 0, h+4, w+4) * mask_small(m, 0, 4, 4);

        y4d(n, m, h, w) = result;   //update the output element
    }

    #undef y4d
    #undef x4d
    #undef k4d
}

//kernel for the larger input layer call
__global__ void forward_kernel_large(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    //shared memory for the input feature maps
    __shared__ float inputFeature_large[LARGE_INPUT][TILE_WIDTH+MASK_WIDTH-1][TILE_WIDTH+MASK_WIDTH-1];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_GRID = (W_out-1) / TILE_WIDTH + 1;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    #define mask_large(i3, i2, i1, i0) mask_large[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int n, m, h, w;
    n = blockIdx.x;     //minibatch index
    m = blockIdx.y;     //output feature index

    //these two parameters need boundary check when referencing
    h = (blockIdx.z / W_GRID) * TILE_WIDTH + threadIdx.y;  //height index of output element in a particular output feature map
    w = (blockIdx.z % W_GRID) * TILE_WIDTH + threadIdx.x;  //width index of output element in a particular output feature map

    // place input feature data in shared memory
    // only place data required for processing in each block

    // constant used for calculation
    int numRow;
    numRow = (blockIdx.z / W_GRID) * TILE_WIDTH;

    if (threadIdx.y == 0) {
        inputFeature_large[0][ 0][threadIdx.x] = x4d(n, 0, numRow +  0, w);
        inputFeature_large[0][ 1][threadIdx.x] = x4d(n, 0, numRow +  1, w);
        inputFeature_large[0][ 2][threadIdx.x] = x4d(n, 0, numRow +  2, w);
        inputFeature_large[0][ 3][threadIdx.x] = x4d(n, 0, numRow +  3, w);
        inputFeature_large[0][ 4][threadIdx.x] = x4d(n, 0, numRow +  4, w);
        inputFeature_large[0][ 5][threadIdx.x] = x4d(n, 0, numRow +  5, w);
        inputFeature_large[0][ 6][threadIdx.x] = x4d(n, 0, numRow +  6, w);
        inputFeature_large[0][ 7][threadIdx.x] = x4d(n, 0, numRow +  7, w);
        inputFeature_large[0][ 8][threadIdx.x] = x4d(n, 0, numRow +  8, w);
        inputFeature_large[0][ 9][threadIdx.x] = x4d(n, 0, numRow +  9, w);
        inputFeature_large[0][10][threadIdx.x] = x4d(n, 0, numRow + 10, w);
        inputFeature_large[0][11][threadIdx.x] = x4d(n, 0, numRow + 11, w);
        inputFeature_large[0][12][threadIdx.x] = x4d(n, 0, numRow + 12, w);
        inputFeature_large[0][13][threadIdx.x] = x4d(n, 0, numRow + 13, w);
        inputFeature_large[0][14][threadIdx.x] = x4d(n, 0, numRow + 14, w);
        inputFeature_large[0][15][threadIdx.x] = x4d(n, 0, numRow + 15, w);
        inputFeature_large[0][16][threadIdx.x] = x4d(n, 0, numRow + 16, w);
        inputFeature_large[0][17][threadIdx.x] = x4d(n, 0, numRow + 17, w);
        inputFeature_large[0][18][threadIdx.x] = x4d(n, 0, numRow + 18, w);
        inputFeature_large[0][19][threadIdx.x] = x4d(n, 0, numRow + 19, w);
        inputFeature_large[0][20][threadIdx.x] = x4d(n, 0, numRow + 20, w);
        inputFeature_large[0][21][threadIdx.x] = x4d(n, 0, numRow + 21, w);
        inputFeature_large[0][22][threadIdx.x] = x4d(n, 0, numRow + 22, w);
        inputFeature_large[0][23][threadIdx.x] = x4d(n, 0, numRow + 23, w);
        inputFeature_large[0][24][threadIdx.x] = x4d(n, 0, numRow + 24, w);
        inputFeature_large[0][25][threadIdx.x] = x4d(n, 0, numRow + 25, w);
        inputFeature_large[0][26][threadIdx.x] = x4d(n, 0, numRow + 26, w);
        inputFeature_large[0][27][threadIdx.x] = x4d(n, 0, numRow + 27, w);
        inputFeature_large[0][28][threadIdx.x] = x4d(n, 0, numRow + 28, w);
        inputFeature_large[0][29][threadIdx.x] = x4d(n, 0, numRow + 29, w);
        inputFeature_large[0][30][threadIdx.x] = x4d(n, 0, numRow + 30, w);
        inputFeature_large[0][31][threadIdx.x] = x4d(n, 0, numRow + 31, w);
        inputFeature_large[0][32][threadIdx.x] = x4d(n, 0, numRow + 32, w);
        inputFeature_large[0][33][threadIdx.x] = x4d(n, 0, numRow + 33, w);
        inputFeature_large[0][34][threadIdx.x] = x4d(n, 0, numRow + 34, w);
        inputFeature_large[0][35][threadIdx.x] = x4d(n, 0, numRow + 35, w);

    }
    else if (threadIdx.y == 1 && threadIdx.x < K-1) {
        inputFeature_large[0][ 0][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow +  0, w + TILE_WIDTH);
        inputFeature_large[0][ 1][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow +  1, w + TILE_WIDTH);
        inputFeature_large[0][ 2][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow +  2, w + TILE_WIDTH);
        inputFeature_large[0][ 3][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow +  3, w + TILE_WIDTH);
        inputFeature_large[0][ 4][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow +  4, w + TILE_WIDTH);
        inputFeature_large[0][ 5][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow +  5, w + TILE_WIDTH);
        inputFeature_large[0][ 6][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow +  6, w + TILE_WIDTH);
        inputFeature_large[0][ 7][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow +  7, w + TILE_WIDTH);
        inputFeature_large[0][ 8][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow +  8, w + TILE_WIDTH);
        inputFeature_large[0][ 9][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow +  9, w + TILE_WIDTH);
        inputFeature_large[0][10][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 10, w + TILE_WIDTH);
        inputFeature_large[0][11][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 11, w + TILE_WIDTH);
        inputFeature_large[0][12][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 12, w + TILE_WIDTH);
        inputFeature_large[0][13][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 13, w + TILE_WIDTH);
        inputFeature_large[0][14][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 14, w + TILE_WIDTH);
        inputFeature_large[0][15][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 15, w + TILE_WIDTH);
        inputFeature_large[0][16][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 16, w + TILE_WIDTH);
        inputFeature_large[0][17][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 17, w + TILE_WIDTH);
        inputFeature_large[0][18][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 18, w + TILE_WIDTH);
        inputFeature_large[0][19][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 19, w + TILE_WIDTH);
        inputFeature_large[0][20][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 20, w + TILE_WIDTH);
        inputFeature_large[0][21][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 21, w + TILE_WIDTH);
        inputFeature_large[0][22][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 22, w + TILE_WIDTH);
        inputFeature_large[0][23][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 23, w + TILE_WIDTH);
        inputFeature_large[0][24][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 24, w + TILE_WIDTH);
        inputFeature_large[0][25][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 25, w + TILE_WIDTH);
        inputFeature_large[0][26][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 26, w + TILE_WIDTH);
        inputFeature_large[0][27][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 27, w + TILE_WIDTH);
        inputFeature_large[0][28][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 28, w + TILE_WIDTH);
        inputFeature_large[0][29][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 29, w + TILE_WIDTH);
        inputFeature_large[0][30][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 30, w + TILE_WIDTH);
        inputFeature_large[0][31][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 31, w + TILE_WIDTH);
        inputFeature_large[0][32][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 32, w + TILE_WIDTH);
        inputFeature_large[0][33][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 33, w + TILE_WIDTH);
        inputFeature_large[0][34][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 34, w + TILE_WIDTH);
        inputFeature_large[0][35][threadIdx.x + TILE_WIDTH] = x4d(n, 0, numRow + 35, w + TILE_WIDTH);
    }

    else if (threadIdx.y == 2) {
        inputFeature_large[1][ 0][threadIdx.x] = x4d(n, 1, numRow +  0, w);
        inputFeature_large[1][ 1][threadIdx.x] = x4d(n, 1, numRow +  1, w);
        inputFeature_large[1][ 2][threadIdx.x] = x4d(n, 1, numRow +  2, w);
        inputFeature_large[1][ 3][threadIdx.x] = x4d(n, 1, numRow +  3, w);
        inputFeature_large[1][ 4][threadIdx.x] = x4d(n, 1, numRow +  4, w);
        inputFeature_large[1][ 5][threadIdx.x] = x4d(n, 1, numRow +  5, w);
        inputFeature_large[1][ 6][threadIdx.x] = x4d(n, 1, numRow +  6, w);
        inputFeature_large[1][ 7][threadIdx.x] = x4d(n, 1, numRow +  7, w);
        inputFeature_large[1][ 8][threadIdx.x] = x4d(n, 1, numRow +  8, w);
        inputFeature_large[1][ 9][threadIdx.x] = x4d(n, 1, numRow +  9, w);
        inputFeature_large[1][10][threadIdx.x] = x4d(n, 1, numRow + 10, w);
        inputFeature_large[1][11][threadIdx.x] = x4d(n, 1, numRow + 11, w);
        inputFeature_large[1][12][threadIdx.x] = x4d(n, 1, numRow + 12, w);
        inputFeature_large[1][13][threadIdx.x] = x4d(n, 1, numRow + 13, w);
        inputFeature_large[1][14][threadIdx.x] = x4d(n, 1, numRow + 14, w);
        inputFeature_large[1][15][threadIdx.x] = x4d(n, 1, numRow + 15, w);
        inputFeature_large[1][16][threadIdx.x] = x4d(n, 1, numRow + 16, w);
        inputFeature_large[1][17][threadIdx.x] = x4d(n, 1, numRow + 17, w);
        inputFeature_large[1][18][threadIdx.x] = x4d(n, 1, numRow + 18, w);
        inputFeature_large[1][19][threadIdx.x] = x4d(n, 1, numRow + 19, w);
        inputFeature_large[1][20][threadIdx.x] = x4d(n, 1, numRow + 20, w);
        inputFeature_large[1][21][threadIdx.x] = x4d(n, 1, numRow + 21, w);
        inputFeature_large[1][22][threadIdx.x] = x4d(n, 1, numRow + 22, w);
        inputFeature_large[1][23][threadIdx.x] = x4d(n, 1, numRow + 23, w);
        inputFeature_large[1][24][threadIdx.x] = x4d(n, 1, numRow + 24, w);
        inputFeature_large[1][25][threadIdx.x] = x4d(n, 1, numRow + 25, w);
        inputFeature_large[1][26][threadIdx.x] = x4d(n, 1, numRow + 26, w);
        inputFeature_large[1][27][threadIdx.x] = x4d(n, 1, numRow + 27, w);
        inputFeature_large[1][28][threadIdx.x] = x4d(n, 1, numRow + 28, w);
        inputFeature_large[1][29][threadIdx.x] = x4d(n, 1, numRow + 29, w);
        inputFeature_large[1][30][threadIdx.x] = x4d(n, 1, numRow + 30, w);
        inputFeature_large[1][31][threadIdx.x] = x4d(n, 1, numRow + 31, w);
        inputFeature_large[1][32][threadIdx.x] = x4d(n, 1, numRow + 32, w);
        inputFeature_large[1][33][threadIdx.x] = x4d(n, 1, numRow + 33, w);
        inputFeature_large[1][34][threadIdx.x] = x4d(n, 1, numRow + 34, w);
        inputFeature_large[1][35][threadIdx.x] = x4d(n, 1, numRow + 35, w);
    }
    else if (threadIdx.y == 3 && threadIdx.x < K-1) {
        inputFeature_large[1][ 0][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow +  0, w + TILE_WIDTH);
        inputFeature_large[1][ 1][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow +  1, w + TILE_WIDTH);
        inputFeature_large[1][ 2][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow +  2, w + TILE_WIDTH);
        inputFeature_large[1][ 3][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow +  3, w + TILE_WIDTH);
        inputFeature_large[1][ 4][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow +  4, w + TILE_WIDTH);
        inputFeature_large[1][ 5][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow +  5, w + TILE_WIDTH);
        inputFeature_large[1][ 6][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow +  6, w + TILE_WIDTH);
        inputFeature_large[1][ 7][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow +  7, w + TILE_WIDTH);
        inputFeature_large[1][ 8][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow +  8, w + TILE_WIDTH);
        inputFeature_large[1][ 9][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow +  9, w + TILE_WIDTH);
        inputFeature_large[1][10][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 10, w + TILE_WIDTH);
        inputFeature_large[1][11][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 11, w + TILE_WIDTH);
        inputFeature_large[1][12][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 12, w + TILE_WIDTH);
        inputFeature_large[1][13][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 13, w + TILE_WIDTH);
        inputFeature_large[1][14][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 14, w + TILE_WIDTH);
        inputFeature_large[1][15][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 15, w + TILE_WIDTH);
        inputFeature_large[1][16][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 16, w + TILE_WIDTH);
        inputFeature_large[1][17][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 17, w + TILE_WIDTH);
        inputFeature_large[1][18][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 18, w + TILE_WIDTH);
        inputFeature_large[1][19][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 19, w + TILE_WIDTH);
        inputFeature_large[1][20][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 20, w + TILE_WIDTH);
        inputFeature_large[1][21][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 21, w + TILE_WIDTH);
        inputFeature_large[1][22][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 22, w + TILE_WIDTH);
        inputFeature_large[1][23][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 23, w + TILE_WIDTH);
        inputFeature_large[1][24][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 24, w + TILE_WIDTH);
        inputFeature_large[1][25][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 25, w + TILE_WIDTH);
        inputFeature_large[1][26][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 26, w + TILE_WIDTH);
        inputFeature_large[1][27][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 27, w + TILE_WIDTH);
        inputFeature_large[1][28][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 28, w + TILE_WIDTH);
        inputFeature_large[1][29][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 29, w + TILE_WIDTH);
        inputFeature_large[1][30][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 30, w + TILE_WIDTH);
        inputFeature_large[1][31][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 31, w + TILE_WIDTH);
        inputFeature_large[1][32][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 32, w + TILE_WIDTH);
        inputFeature_large[1][33][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 33, w + TILE_WIDTH);
        inputFeature_large[1][34][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 34, w + TILE_WIDTH);
        inputFeature_large[1][35][threadIdx.x + TILE_WIDTH] = x4d(n, 1, numRow + 35, w + TILE_WIDTH);
    }

    else if (threadIdx.y == 4) {
        inputFeature_large[2][ 0][threadIdx.x] = x4d(n, 2, numRow +  0, w);
        inputFeature_large[2][ 1][threadIdx.x] = x4d(n, 2, numRow +  1, w);
        inputFeature_large[2][ 2][threadIdx.x] = x4d(n, 2, numRow +  2, w);
        inputFeature_large[2][ 3][threadIdx.x] = x4d(n, 2, numRow +  3, w);
        inputFeature_large[2][ 4][threadIdx.x] = x4d(n, 2, numRow +  4, w);
        inputFeature_large[2][ 5][threadIdx.x] = x4d(n, 2, numRow +  5, w);
        inputFeature_large[2][ 6][threadIdx.x] = x4d(n, 2, numRow +  6, w);
        inputFeature_large[2][ 7][threadIdx.x] = x4d(n, 2, numRow +  7, w);
        inputFeature_large[2][ 8][threadIdx.x] = x4d(n, 2, numRow +  8, w);
        inputFeature_large[2][ 9][threadIdx.x] = x4d(n, 2, numRow +  9, w);
        inputFeature_large[2][10][threadIdx.x] = x4d(n, 2, numRow + 10, w);
        inputFeature_large[2][11][threadIdx.x] = x4d(n, 2, numRow + 11, w);
        inputFeature_large[2][12][threadIdx.x] = x4d(n, 2, numRow + 12, w);
        inputFeature_large[2][13][threadIdx.x] = x4d(n, 2, numRow + 13, w);
        inputFeature_large[2][14][threadIdx.x] = x4d(n, 2, numRow + 14, w);
        inputFeature_large[2][15][threadIdx.x] = x4d(n, 2, numRow + 15, w);
        inputFeature_large[2][16][threadIdx.x] = x4d(n, 2, numRow + 16, w);
        inputFeature_large[2][17][threadIdx.x] = x4d(n, 2, numRow + 17, w);
        inputFeature_large[2][18][threadIdx.x] = x4d(n, 2, numRow + 18, w);
        inputFeature_large[2][19][threadIdx.x] = x4d(n, 2, numRow + 19, w);
        inputFeature_large[2][20][threadIdx.x] = x4d(n, 2, numRow + 20, w);
        inputFeature_large[2][21][threadIdx.x] = x4d(n, 2, numRow + 21, w);
        inputFeature_large[2][22][threadIdx.x] = x4d(n, 2, numRow + 22, w);
        inputFeature_large[2][23][threadIdx.x] = x4d(n, 2, numRow + 23, w);
        inputFeature_large[2][24][threadIdx.x] = x4d(n, 2, numRow + 24, w);
        inputFeature_large[2][25][threadIdx.x] = x4d(n, 2, numRow + 25, w);
        inputFeature_large[2][26][threadIdx.x] = x4d(n, 2, numRow + 26, w);
        inputFeature_large[2][27][threadIdx.x] = x4d(n, 2, numRow + 27, w);
        inputFeature_large[2][28][threadIdx.x] = x4d(n, 2, numRow + 28, w);
        inputFeature_large[2][29][threadIdx.x] = x4d(n, 2, numRow + 29, w);
        inputFeature_large[2][30][threadIdx.x] = x4d(n, 2, numRow + 30, w);
        inputFeature_large[2][31][threadIdx.x] = x4d(n, 2, numRow + 31, w);
        inputFeature_large[2][32][threadIdx.x] = x4d(n, 2, numRow + 32, w);
        inputFeature_large[2][33][threadIdx.x] = x4d(n, 2, numRow + 33, w);
        inputFeature_large[2][34][threadIdx.x] = x4d(n, 2, numRow + 34, w);
        inputFeature_large[2][35][threadIdx.x] = x4d(n, 2, numRow + 35, w);
    }
    else if (threadIdx.y == 5 && threadIdx.x < K-1) {
        inputFeature_large[2][ 0][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow +  0, w + TILE_WIDTH);
        inputFeature_large[2][ 1][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow +  1, w + TILE_WIDTH);
        inputFeature_large[2][ 2][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow +  2, w + TILE_WIDTH);
        inputFeature_large[2][ 3][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow +  3, w + TILE_WIDTH);
        inputFeature_large[2][ 4][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow +  4, w + TILE_WIDTH);
        inputFeature_large[2][ 5][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow +  5, w + TILE_WIDTH);
        inputFeature_large[2][ 6][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow +  6, w + TILE_WIDTH);
        inputFeature_large[2][ 7][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow +  7, w + TILE_WIDTH);
        inputFeature_large[2][ 8][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow +  8, w + TILE_WIDTH);
        inputFeature_large[2][ 9][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow +  9, w + TILE_WIDTH);
        inputFeature_large[2][10][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 10, w + TILE_WIDTH);
        inputFeature_large[2][11][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 11, w + TILE_WIDTH);
        inputFeature_large[2][12][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 12, w + TILE_WIDTH);
        inputFeature_large[2][13][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 13, w + TILE_WIDTH);
        inputFeature_large[2][14][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 14, w + TILE_WIDTH);
        inputFeature_large[2][15][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 15, w + TILE_WIDTH);
        inputFeature_large[2][16][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 16, w + TILE_WIDTH);
        inputFeature_large[2][17][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 17, w + TILE_WIDTH);
        inputFeature_large[2][18][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 18, w + TILE_WIDTH);
        inputFeature_large[2][19][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 19, w + TILE_WIDTH);
        inputFeature_large[2][20][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 20, w + TILE_WIDTH);
        inputFeature_large[2][21][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 21, w + TILE_WIDTH);
        inputFeature_large[2][22][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 22, w + TILE_WIDTH);
        inputFeature_large[2][23][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 23, w + TILE_WIDTH);
        inputFeature_large[2][24][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 24, w + TILE_WIDTH);
        inputFeature_large[2][25][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 25, w + TILE_WIDTH);
        inputFeature_large[2][26][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 26, w + TILE_WIDTH);
        inputFeature_large[2][27][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 27, w + TILE_WIDTH);
        inputFeature_large[2][28][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 28, w + TILE_WIDTH);
        inputFeature_large[2][29][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 29, w + TILE_WIDTH);
        inputFeature_large[2][30][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 30, w + TILE_WIDTH);
        inputFeature_large[2][31][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 31, w + TILE_WIDTH);
        inputFeature_large[2][32][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 32, w + TILE_WIDTH);
        inputFeature_large[2][33][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 33, w + TILE_WIDTH);
        inputFeature_large[2][34][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 34, w + TILE_WIDTH);
        inputFeature_large[2][35][threadIdx.x + TILE_WIDTH] = x4d(n, 2, numRow + 35, w + TILE_WIDTH);
    }

    else if (threadIdx.y == 6) {
        inputFeature_large[3][ 0][threadIdx.x] = x4d(n, 3, numRow +  0, w);
        inputFeature_large[3][ 1][threadIdx.x] = x4d(n, 3, numRow +  1, w);
        inputFeature_large[3][ 2][threadIdx.x] = x4d(n, 3, numRow +  2, w);
        inputFeature_large[3][ 3][threadIdx.x] = x4d(n, 3, numRow +  3, w);
        inputFeature_large[3][ 4][threadIdx.x] = x4d(n, 3, numRow +  4, w);
        inputFeature_large[3][ 5][threadIdx.x] = x4d(n, 3, numRow +  5, w);
        inputFeature_large[3][ 6][threadIdx.x] = x4d(n, 3, numRow +  6, w);
        inputFeature_large[3][ 7][threadIdx.x] = x4d(n, 3, numRow +  7, w);
        inputFeature_large[3][ 8][threadIdx.x] = x4d(n, 3, numRow +  8, w);
        inputFeature_large[3][ 9][threadIdx.x] = x4d(n, 3, numRow +  9, w);
        inputFeature_large[3][10][threadIdx.x] = x4d(n, 3, numRow + 10, w);
        inputFeature_large[3][11][threadIdx.x] = x4d(n, 3, numRow + 11, w);
        inputFeature_large[3][12][threadIdx.x] = x4d(n, 3, numRow + 12, w);
        inputFeature_large[3][13][threadIdx.x] = x4d(n, 3, numRow + 13, w);
        inputFeature_large[3][14][threadIdx.x] = x4d(n, 3, numRow + 14, w);
        inputFeature_large[3][15][threadIdx.x] = x4d(n, 3, numRow + 15, w);
        inputFeature_large[3][16][threadIdx.x] = x4d(n, 3, numRow + 16, w);
        inputFeature_large[3][17][threadIdx.x] = x4d(n, 3, numRow + 17, w);
        inputFeature_large[3][18][threadIdx.x] = x4d(n, 3, numRow + 18, w);
        inputFeature_large[3][19][threadIdx.x] = x4d(n, 3, numRow + 19, w);
        inputFeature_large[3][20][threadIdx.x] = x4d(n, 3, numRow + 20, w);
        inputFeature_large[3][21][threadIdx.x] = x4d(n, 3, numRow + 21, w);
        inputFeature_large[3][22][threadIdx.x] = x4d(n, 3, numRow + 22, w);
        inputFeature_large[3][23][threadIdx.x] = x4d(n, 3, numRow + 23, w);
        inputFeature_large[3][24][threadIdx.x] = x4d(n, 3, numRow + 24, w);
        inputFeature_large[3][25][threadIdx.x] = x4d(n, 3, numRow + 25, w);
        inputFeature_large[3][26][threadIdx.x] = x4d(n, 3, numRow + 26, w);
        inputFeature_large[3][27][threadIdx.x] = x4d(n, 3, numRow + 27, w);
        inputFeature_large[3][28][threadIdx.x] = x4d(n, 3, numRow + 28, w);
        inputFeature_large[3][29][threadIdx.x] = x4d(n, 3, numRow + 29, w);
        inputFeature_large[3][30][threadIdx.x] = x4d(n, 3, numRow + 30, w);
        inputFeature_large[3][31][threadIdx.x] = x4d(n, 3, numRow + 31, w);
        inputFeature_large[3][32][threadIdx.x] = x4d(n, 3, numRow + 32, w);
        inputFeature_large[3][33][threadIdx.x] = x4d(n, 3, numRow + 33, w);
        inputFeature_large[3][34][threadIdx.x] = x4d(n, 3, numRow + 34, w);
        inputFeature_large[3][35][threadIdx.x] = x4d(n, 3, numRow + 35, w);
    }
    else if (threadIdx.y == 7 && threadIdx.x < K-1) {
        inputFeature_large[3][ 0][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow +  0, w + TILE_WIDTH);
        inputFeature_large[3][ 1][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow +  1, w + TILE_WIDTH);
        inputFeature_large[3][ 2][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow +  2, w + TILE_WIDTH);
        inputFeature_large[3][ 3][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow +  3, w + TILE_WIDTH);
        inputFeature_large[3][ 4][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow +  4, w + TILE_WIDTH);
        inputFeature_large[3][ 5][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow +  5, w + TILE_WIDTH);
        inputFeature_large[3][ 6][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow +  6, w + TILE_WIDTH);
        inputFeature_large[3][ 7][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow +  7, w + TILE_WIDTH);
        inputFeature_large[3][ 8][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow +  8, w + TILE_WIDTH);
        inputFeature_large[3][ 9][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow +  9, w + TILE_WIDTH);
        inputFeature_large[3][10][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 10, w + TILE_WIDTH);
        inputFeature_large[3][11][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 11, w + TILE_WIDTH);
        inputFeature_large[3][12][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 12, w + TILE_WIDTH);
        inputFeature_large[3][13][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 13, w + TILE_WIDTH);
        inputFeature_large[3][14][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 14, w + TILE_WIDTH);
        inputFeature_large[3][15][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 15, w + TILE_WIDTH);
        inputFeature_large[3][16][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 16, w + TILE_WIDTH);
        inputFeature_large[3][17][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 17, w + TILE_WIDTH);
        inputFeature_large[3][18][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 18, w + TILE_WIDTH);
        inputFeature_large[3][19][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 19, w + TILE_WIDTH);
        inputFeature_large[3][20][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 20, w + TILE_WIDTH);
        inputFeature_large[3][21][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 21, w + TILE_WIDTH);
        inputFeature_large[3][22][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 22, w + TILE_WIDTH);
        inputFeature_large[3][23][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 23, w + TILE_WIDTH);
        inputFeature_large[3][24][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 24, w + TILE_WIDTH);
        inputFeature_large[3][25][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 25, w + TILE_WIDTH);
        inputFeature_large[3][26][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 26, w + TILE_WIDTH);
        inputFeature_large[3][27][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 27, w + TILE_WIDTH);
        inputFeature_large[3][28][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 28, w + TILE_WIDTH);
        inputFeature_large[3][29][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 29, w + TILE_WIDTH);
        inputFeature_large[3][30][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 30, w + TILE_WIDTH);
        inputFeature_large[3][31][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 31, w + TILE_WIDTH);
        inputFeature_large[3][32][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 32, w + TILE_WIDTH);
        inputFeature_large[3][33][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 33, w + TILE_WIDTH);
        inputFeature_large[3][34][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 34, w + TILE_WIDTH);
        inputFeature_large[3][35][threadIdx.x + TILE_WIDTH] = x4d(n, 3, numRow + 35, w + TILE_WIDTH);
    }

    else if (threadIdx.y == 8) {
        inputFeature_large[4][ 0][threadIdx.x] = x4d(n, 4, numRow +  0, w);
        inputFeature_large[4][ 1][threadIdx.x] = x4d(n, 4, numRow +  1, w);
        inputFeature_large[4][ 2][threadIdx.x] = x4d(n, 4, numRow +  2, w);
        inputFeature_large[4][ 3][threadIdx.x] = x4d(n, 4, numRow +  3, w);
        inputFeature_large[4][ 4][threadIdx.x] = x4d(n, 4, numRow +  4, w);
        inputFeature_large[4][ 5][threadIdx.x] = x4d(n, 4, numRow +  5, w);
        inputFeature_large[4][ 6][threadIdx.x] = x4d(n, 4, numRow +  6, w);
        inputFeature_large[4][ 7][threadIdx.x] = x4d(n, 4, numRow +  7, w);
        inputFeature_large[4][ 8][threadIdx.x] = x4d(n, 4, numRow +  8, w);
        inputFeature_large[4][ 9][threadIdx.x] = x4d(n, 4, numRow +  9, w);
        inputFeature_large[4][10][threadIdx.x] = x4d(n, 4, numRow + 10, w);
        inputFeature_large[4][11][threadIdx.x] = x4d(n, 4, numRow + 11, w);
        inputFeature_large[4][12][threadIdx.x] = x4d(n, 4, numRow + 12, w);
        inputFeature_large[4][13][threadIdx.x] = x4d(n, 4, numRow + 13, w);
        inputFeature_large[4][14][threadIdx.x] = x4d(n, 4, numRow + 14, w);
        inputFeature_large[4][15][threadIdx.x] = x4d(n, 4, numRow + 15, w);
        inputFeature_large[4][16][threadIdx.x] = x4d(n, 4, numRow + 16, w);
        inputFeature_large[4][17][threadIdx.x] = x4d(n, 4, numRow + 17, w);
        inputFeature_large[4][18][threadIdx.x] = x4d(n, 4, numRow + 18, w);
        inputFeature_large[4][19][threadIdx.x] = x4d(n, 4, numRow + 19, w);
        inputFeature_large[4][20][threadIdx.x] = x4d(n, 4, numRow + 20, w);
        inputFeature_large[4][21][threadIdx.x] = x4d(n, 4, numRow + 21, w);
        inputFeature_large[4][22][threadIdx.x] = x4d(n, 4, numRow + 22, w);
        inputFeature_large[4][23][threadIdx.x] = x4d(n, 4, numRow + 23, w);
        inputFeature_large[4][24][threadIdx.x] = x4d(n, 4, numRow + 24, w);
        inputFeature_large[4][25][threadIdx.x] = x4d(n, 4, numRow + 25, w);
        inputFeature_large[4][26][threadIdx.x] = x4d(n, 4, numRow + 26, w);
        inputFeature_large[4][27][threadIdx.x] = x4d(n, 4, numRow + 27, w);
        inputFeature_large[4][28][threadIdx.x] = x4d(n, 4, numRow + 28, w);
        inputFeature_large[4][29][threadIdx.x] = x4d(n, 4, numRow + 29, w);
        inputFeature_large[4][30][threadIdx.x] = x4d(n, 4, numRow + 30, w);
        inputFeature_large[4][31][threadIdx.x] = x4d(n, 4, numRow + 31, w);
        inputFeature_large[4][32][threadIdx.x] = x4d(n, 4, numRow + 32, w);
        inputFeature_large[4][33][threadIdx.x] = x4d(n, 4, numRow + 33, w);
        inputFeature_large[4][34][threadIdx.x] = x4d(n, 4, numRow + 34, w);
        inputFeature_large[4][35][threadIdx.x] = x4d(n, 4, numRow + 35, w);
    }
    else if (threadIdx.y == 9 && threadIdx.x < K-1) {
        inputFeature_large[4][ 0][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow +  0, w + TILE_WIDTH);
        inputFeature_large[4][ 1][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow +  1, w + TILE_WIDTH);
        inputFeature_large[4][ 2][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow +  2, w + TILE_WIDTH);
        inputFeature_large[4][ 3][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow +  3, w + TILE_WIDTH);
        inputFeature_large[4][ 4][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow +  4, w + TILE_WIDTH);
        inputFeature_large[4][ 5][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow +  5, w + TILE_WIDTH);
        inputFeature_large[4][ 6][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow +  6, w + TILE_WIDTH);
        inputFeature_large[4][ 7][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow +  7, w + TILE_WIDTH);
        inputFeature_large[4][ 8][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow +  8, w + TILE_WIDTH);
        inputFeature_large[4][ 9][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow +  9, w + TILE_WIDTH);
        inputFeature_large[4][10][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 10, w + TILE_WIDTH);
        inputFeature_large[4][11][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 11, w + TILE_WIDTH);
        inputFeature_large[4][12][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 12, w + TILE_WIDTH);
        inputFeature_large[4][13][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 13, w + TILE_WIDTH);
        inputFeature_large[4][14][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 14, w + TILE_WIDTH);
        inputFeature_large[4][15][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 15, w + TILE_WIDTH);
        inputFeature_large[4][16][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 16, w + TILE_WIDTH);
        inputFeature_large[4][17][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 17, w + TILE_WIDTH);
        inputFeature_large[4][18][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 18, w + TILE_WIDTH);
        inputFeature_large[4][19][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 19, w + TILE_WIDTH);
        inputFeature_large[4][20][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 20, w + TILE_WIDTH);
        inputFeature_large[4][21][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 21, w + TILE_WIDTH);
        inputFeature_large[4][22][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 22, w + TILE_WIDTH);
        inputFeature_large[4][23][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 23, w + TILE_WIDTH);
        inputFeature_large[4][24][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 24, w + TILE_WIDTH);
        inputFeature_large[4][25][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 25, w + TILE_WIDTH);
        inputFeature_large[4][26][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 26, w + TILE_WIDTH);
        inputFeature_large[4][27][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 27, w + TILE_WIDTH);
        inputFeature_large[4][28][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 28, w + TILE_WIDTH);
        inputFeature_large[4][29][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 29, w + TILE_WIDTH);
        inputFeature_large[4][30][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 30, w + TILE_WIDTH);
        inputFeature_large[4][31][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 31, w + TILE_WIDTH);
        inputFeature_large[4][32][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 32, w + TILE_WIDTH);
        inputFeature_large[4][33][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 33, w + TILE_WIDTH);
        inputFeature_large[4][34][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 34, w + TILE_WIDTH);
        inputFeature_large[4][35][threadIdx.x + TILE_WIDTH] = x4d(n, 4, numRow + 35, w + TILE_WIDTH);
    }

    else if (threadIdx.y == 10) {
        inputFeature_large[5][ 0][threadIdx.x] = x4d(n, 5, numRow +  0, w);
        inputFeature_large[5][ 1][threadIdx.x] = x4d(n, 5, numRow +  1, w);
        inputFeature_large[5][ 2][threadIdx.x] = x4d(n, 5, numRow +  2, w);
        inputFeature_large[5][ 3][threadIdx.x] = x4d(n, 5, numRow +  3, w);
        inputFeature_large[5][ 4][threadIdx.x] = x4d(n, 5, numRow +  4, w);
        inputFeature_large[5][ 5][threadIdx.x] = x4d(n, 5, numRow +  5, w);
        inputFeature_large[5][ 6][threadIdx.x] = x4d(n, 5, numRow +  6, w);
        inputFeature_large[5][ 7][threadIdx.x] = x4d(n, 5, numRow +  7, w);
        inputFeature_large[5][ 8][threadIdx.x] = x4d(n, 5, numRow +  8, w);
        inputFeature_large[5][ 9][threadIdx.x] = x4d(n, 5, numRow +  9, w);
        inputFeature_large[5][10][threadIdx.x] = x4d(n, 5, numRow + 10, w);
        inputFeature_large[5][11][threadIdx.x] = x4d(n, 5, numRow + 11, w);
        inputFeature_large[5][12][threadIdx.x] = x4d(n, 5, numRow + 12, w);
        inputFeature_large[5][13][threadIdx.x] = x4d(n, 5, numRow + 13, w);
        inputFeature_large[5][14][threadIdx.x] = x4d(n, 5, numRow + 14, w);
        inputFeature_large[5][15][threadIdx.x] = x4d(n, 5, numRow + 15, w);
        inputFeature_large[5][16][threadIdx.x] = x4d(n, 5, numRow + 16, w);
        inputFeature_large[5][17][threadIdx.x] = x4d(n, 5, numRow + 17, w);
        inputFeature_large[5][18][threadIdx.x] = x4d(n, 5, numRow + 18, w);
        inputFeature_large[5][19][threadIdx.x] = x4d(n, 5, numRow + 19, w);
        inputFeature_large[5][20][threadIdx.x] = x4d(n, 5, numRow + 20, w);
        inputFeature_large[5][21][threadIdx.x] = x4d(n, 5, numRow + 21, w);
        inputFeature_large[5][22][threadIdx.x] = x4d(n, 5, numRow + 22, w);
        inputFeature_large[5][23][threadIdx.x] = x4d(n, 5, numRow + 23, w);
        inputFeature_large[5][24][threadIdx.x] = x4d(n, 5, numRow + 24, w);
        inputFeature_large[5][25][threadIdx.x] = x4d(n, 5, numRow + 25, w);
        inputFeature_large[5][26][threadIdx.x] = x4d(n, 5, numRow + 26, w);
        inputFeature_large[5][27][threadIdx.x] = x4d(n, 5, numRow + 27, w);
        inputFeature_large[5][28][threadIdx.x] = x4d(n, 5, numRow + 28, w);
        inputFeature_large[5][29][threadIdx.x] = x4d(n, 5, numRow + 29, w);
        inputFeature_large[5][30][threadIdx.x] = x4d(n, 5, numRow + 30, w);
        inputFeature_large[5][31][threadIdx.x] = x4d(n, 5, numRow + 31, w);
        inputFeature_large[5][32][threadIdx.x] = x4d(n, 5, numRow + 32, w);
        inputFeature_large[5][33][threadIdx.x] = x4d(n, 5, numRow + 33, w);
        inputFeature_large[5][34][threadIdx.x] = x4d(n, 5, numRow + 34, w);
        inputFeature_large[5][35][threadIdx.x] = x4d(n, 5, numRow + 35, w);
    }
    else if (threadIdx.y == 11 && threadIdx.x < K-1) {
        inputFeature_large[5][ 0][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow +  0, w + TILE_WIDTH);
        inputFeature_large[5][ 1][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow +  1, w + TILE_WIDTH);
        inputFeature_large[5][ 2][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow +  2, w + TILE_WIDTH);
        inputFeature_large[5][ 3][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow +  3, w + TILE_WIDTH);
        inputFeature_large[5][ 4][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow +  4, w + TILE_WIDTH);
        inputFeature_large[5][ 5][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow +  5, w + TILE_WIDTH);
        inputFeature_large[5][ 6][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow +  6, w + TILE_WIDTH);
        inputFeature_large[5][ 7][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow +  7, w + TILE_WIDTH);
        inputFeature_large[5][ 8][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow +  8, w + TILE_WIDTH);
        inputFeature_large[5][ 9][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow +  9, w + TILE_WIDTH);
        inputFeature_large[5][10][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 10, w + TILE_WIDTH);
        inputFeature_large[5][11][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 11, w + TILE_WIDTH);
        inputFeature_large[5][12][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 12, w + TILE_WIDTH);
        inputFeature_large[5][13][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 13, w + TILE_WIDTH);
        inputFeature_large[5][14][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 14, w + TILE_WIDTH);
        inputFeature_large[5][15][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 15, w + TILE_WIDTH);
        inputFeature_large[5][16][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 16, w + TILE_WIDTH);
        inputFeature_large[5][17][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 17, w + TILE_WIDTH);
        inputFeature_large[5][18][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 18, w + TILE_WIDTH);
        inputFeature_large[5][19][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 19, w + TILE_WIDTH);
        inputFeature_large[5][20][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 20, w + TILE_WIDTH);
        inputFeature_large[5][21][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 21, w + TILE_WIDTH);
        inputFeature_large[5][22][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 22, w + TILE_WIDTH);
        inputFeature_large[5][23][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 23, w + TILE_WIDTH);
        inputFeature_large[5][24][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 24, w + TILE_WIDTH);
        inputFeature_large[5][25][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 25, w + TILE_WIDTH);
        inputFeature_large[5][26][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 26, w + TILE_WIDTH);
        inputFeature_large[5][27][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 27, w + TILE_WIDTH);
        inputFeature_large[5][28][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 28, w + TILE_WIDTH);
        inputFeature_large[5][29][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 29, w + TILE_WIDTH);
        inputFeature_large[5][30][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 30, w + TILE_WIDTH);
        inputFeature_large[5][31][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 31, w + TILE_WIDTH);
        inputFeature_large[5][32][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 32, w + TILE_WIDTH);
        inputFeature_large[5][33][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 33, w + TILE_WIDTH);
        inputFeature_large[5][34][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 34, w + TILE_WIDTH);
        inputFeature_large[5][35][threadIdx.x + TILE_WIDTH] = x4d(n, 5, numRow + 35, w + TILE_WIDTH);
    }
    __syncthreads();

    //loop unrolling
    float result = 0;
    if (h < H_out && w < W_out) {   //check if threadidx is in range of the output feature map
        //c == 0
        result += inputFeature_large[0][threadIdx.y + 0][threadIdx.x + 0] * mask_large(m, 0, 0, 0);
        result += inputFeature_large[0][threadIdx.y + 0][threadIdx.x + 1] * mask_large(m, 0, 0, 1);
        result += inputFeature_large[0][threadIdx.y + 0][threadIdx.x + 2] * mask_large(m, 0, 0, 2);
        result += inputFeature_large[0][threadIdx.y + 0][threadIdx.x + 3] * mask_large(m, 0, 0, 3);
        result += inputFeature_large[0][threadIdx.y + 0][threadIdx.x + 4] * mask_large(m, 0, 0, 4);

        result += inputFeature_large[0][threadIdx.y + 1][threadIdx.x + 0] * mask_large(m, 0, 1, 0);
        result += inputFeature_large[0][threadIdx.y + 1][threadIdx.x + 1] * mask_large(m, 0, 1, 1);
        result += inputFeature_large[0][threadIdx.y + 1][threadIdx.x + 2] * mask_large(m, 0, 1, 2);
        result += inputFeature_large[0][threadIdx.y + 1][threadIdx.x + 3] * mask_large(m, 0, 1, 3);
        result += inputFeature_large[0][threadIdx.y + 1][threadIdx.x + 4] * mask_large(m, 0, 1, 4);

        result += inputFeature_large[0][threadIdx.y + 2][threadIdx.x + 0] * mask_large(m, 0, 2, 0);
        result += inputFeature_large[0][threadIdx.y + 2][threadIdx.x + 1] * mask_large(m, 0, 2, 1);
        result += inputFeature_large[0][threadIdx.y + 2][threadIdx.x + 2] * mask_large(m, 0, 2, 2);
        result += inputFeature_large[0][threadIdx.y + 2][threadIdx.x + 3] * mask_large(m, 0, 2, 3);
        result += inputFeature_large[0][threadIdx.y + 2][threadIdx.x + 4] * mask_large(m, 0, 2, 4);

        result += inputFeature_large[0][threadIdx.y + 3][threadIdx.x + 0] * mask_large(m, 0, 3, 0);
        result += inputFeature_large[0][threadIdx.y + 3][threadIdx.x + 1] * mask_large(m, 0, 3, 1);
        result += inputFeature_large[0][threadIdx.y + 3][threadIdx.x + 2] * mask_large(m, 0, 3, 2);
        result += inputFeature_large[0][threadIdx.y + 3][threadIdx.x + 3] * mask_large(m, 0, 3, 3);
        result += inputFeature_large[0][threadIdx.y + 3][threadIdx.x + 4] * mask_large(m, 0, 3, 4);

        result += inputFeature_large[0][threadIdx.y + 4][threadIdx.x + 0] * mask_large(m, 0, 4, 0);
        result += inputFeature_large[0][threadIdx.y + 4][threadIdx.x + 1] * mask_large(m, 0, 4, 1);
        result += inputFeature_large[0][threadIdx.y + 4][threadIdx.x + 2] * mask_large(m, 0, 4, 2);
        result += inputFeature_large[0][threadIdx.y + 4][threadIdx.x + 3] * mask_large(m, 0, 4, 3);
        result += inputFeature_large[0][threadIdx.y + 4][threadIdx.x + 4] * mask_large(m, 0, 4, 4);

        //c == 1
        result += inputFeature_large[1][threadIdx.y + 0][threadIdx.x + 0] * mask_large(m, 1, 0, 0);
        result += inputFeature_large[1][threadIdx.y + 0][threadIdx.x + 1] * mask_large(m, 1, 0, 1);
        result += inputFeature_large[1][threadIdx.y + 0][threadIdx.x + 2] * mask_large(m, 1, 0, 2);
        result += inputFeature_large[1][threadIdx.y + 0][threadIdx.x + 3] * mask_large(m, 1, 0, 3);
        result += inputFeature_large[1][threadIdx.y + 0][threadIdx.x + 4] * mask_large(m, 1, 0, 4);

        result += inputFeature_large[1][threadIdx.y + 1][threadIdx.x + 0] * mask_large(m, 1, 1, 0);
        result += inputFeature_large[1][threadIdx.y + 1][threadIdx.x + 1] * mask_large(m, 1, 1, 1);
        result += inputFeature_large[1][threadIdx.y + 1][threadIdx.x + 2] * mask_large(m, 1, 1, 2);
        result += inputFeature_large[1][threadIdx.y + 1][threadIdx.x + 3] * mask_large(m, 1, 1, 3);
        result += inputFeature_large[1][threadIdx.y + 1][threadIdx.x + 4] * mask_large(m, 1, 1, 4);

        result += inputFeature_large[1][threadIdx.y + 2][threadIdx.x + 0] * mask_large(m, 1, 2, 0);
        result += inputFeature_large[1][threadIdx.y + 2][threadIdx.x + 1] * mask_large(m, 1, 2, 1);
        result += inputFeature_large[1][threadIdx.y + 2][threadIdx.x + 2] * mask_large(m, 1, 2, 2);
        result += inputFeature_large[1][threadIdx.y + 2][threadIdx.x + 3] * mask_large(m, 1, 2, 3);
        result += inputFeature_large[1][threadIdx.y + 2][threadIdx.x + 4] * mask_large(m, 1, 2, 4);

        result += inputFeature_large[1][threadIdx.y + 3][threadIdx.x + 0] * mask_large(m, 1, 3, 0);
        result += inputFeature_large[1][threadIdx.y + 3][threadIdx.x + 1] * mask_large(m, 1, 3, 1);
        result += inputFeature_large[1][threadIdx.y + 3][threadIdx.x + 2] * mask_large(m, 1, 3, 2);
        result += inputFeature_large[1][threadIdx.y + 3][threadIdx.x + 3] * mask_large(m, 1, 3, 3);
        result += inputFeature_large[1][threadIdx.y + 3][threadIdx.x + 4] * mask_large(m, 1, 3, 4);

        result += inputFeature_large[1][threadIdx.y + 4][threadIdx.x + 0] * mask_large(m, 1, 4, 0);
        result += inputFeature_large[1][threadIdx.y + 4][threadIdx.x + 1] * mask_large(m, 1, 4, 1);
        result += inputFeature_large[1][threadIdx.y + 4][threadIdx.x + 2] * mask_large(m, 1, 4, 2);
        result += inputFeature_large[1][threadIdx.y + 4][threadIdx.x + 3] * mask_large(m, 1, 4, 3);
        result += inputFeature_large[1][threadIdx.y + 4][threadIdx.x + 4] * mask_large(m, 1, 4, 4);

        //c == 2
        result += inputFeature_large[2][threadIdx.y + 0][threadIdx.x + 0] * mask_large(m, 2, 0, 0);
        result += inputFeature_large[2][threadIdx.y + 0][threadIdx.x + 1] * mask_large(m, 2, 0, 1);
        result += inputFeature_large[2][threadIdx.y + 0][threadIdx.x + 2] * mask_large(m, 2, 0, 2);
        result += inputFeature_large[2][threadIdx.y + 0][threadIdx.x + 3] * mask_large(m, 2, 0, 3);
        result += inputFeature_large[2][threadIdx.y + 0][threadIdx.x + 4] * mask_large(m, 2, 0, 4);

        result += inputFeature_large[2][threadIdx.y + 1][threadIdx.x + 0] * mask_large(m, 2, 1, 0);
        result += inputFeature_large[2][threadIdx.y + 1][threadIdx.x + 1] * mask_large(m, 2, 1, 1);
        result += inputFeature_large[2][threadIdx.y + 1][threadIdx.x + 2] * mask_large(m, 2, 1, 2);
        result += inputFeature_large[2][threadIdx.y + 1][threadIdx.x + 3] * mask_large(m, 2, 1, 3);
        result += inputFeature_large[2][threadIdx.y + 1][threadIdx.x + 4] * mask_large(m, 2, 1, 4);

        result += inputFeature_large[2][threadIdx.y + 2][threadIdx.x + 0] * mask_large(m, 2, 2, 0);
        result += inputFeature_large[2][threadIdx.y + 2][threadIdx.x + 1] * mask_large(m, 2, 2, 1);
        result += inputFeature_large[2][threadIdx.y + 2][threadIdx.x + 2] * mask_large(m, 2, 2, 2);
        result += inputFeature_large[2][threadIdx.y + 2][threadIdx.x + 3] * mask_large(m, 2, 2, 3);
        result += inputFeature_large[2][threadIdx.y + 2][threadIdx.x + 4] * mask_large(m, 2, 2, 4);

        result += inputFeature_large[2][threadIdx.y + 3][threadIdx.x + 0] * mask_large(m, 2, 3, 0);
        result += inputFeature_large[2][threadIdx.y + 3][threadIdx.x + 1] * mask_large(m, 2, 3, 1);
        result += inputFeature_large[2][threadIdx.y + 3][threadIdx.x + 2] * mask_large(m, 2, 3, 2);
        result += inputFeature_large[2][threadIdx.y + 3][threadIdx.x + 3] * mask_large(m, 2, 3, 3);
        result += inputFeature_large[2][threadIdx.y + 3][threadIdx.x + 4] * mask_large(m, 2, 3, 4);

        result += inputFeature_large[2][threadIdx.y + 4][threadIdx.x + 0] * mask_large(m, 2, 4, 0);
        result += inputFeature_large[2][threadIdx.y + 4][threadIdx.x + 1] * mask_large(m, 2, 4, 1);
        result += inputFeature_large[2][threadIdx.y + 4][threadIdx.x + 2] * mask_large(m, 2, 4, 2);
        result += inputFeature_large[2][threadIdx.y + 4][threadIdx.x + 3] * mask_large(m, 2, 4, 3);
        result += inputFeature_large[2][threadIdx.y + 4][threadIdx.x + 4] * mask_large(m, 2, 4, 4);

        //c == 3
        result += inputFeature_large[3][threadIdx.y + 0][threadIdx.x + 0] * mask_large(m, 3, 0, 0);
        result += inputFeature_large[3][threadIdx.y + 0][threadIdx.x + 1] * mask_large(m, 3, 0, 1);
        result += inputFeature_large[3][threadIdx.y + 0][threadIdx.x + 2] * mask_large(m, 3, 0, 2);
        result += inputFeature_large[3][threadIdx.y + 0][threadIdx.x + 3] * mask_large(m, 3, 0, 3);
        result += inputFeature_large[3][threadIdx.y + 0][threadIdx.x + 4] * mask_large(m, 3, 0, 4);

        result += inputFeature_large[3][threadIdx.y + 1][threadIdx.x + 0] * mask_large(m, 3, 1, 0);
        result += inputFeature_large[3][threadIdx.y + 1][threadIdx.x + 1] * mask_large(m, 3, 1, 1);
        result += inputFeature_large[3][threadIdx.y + 1][threadIdx.x + 2] * mask_large(m, 3, 1, 2);
        result += inputFeature_large[3][threadIdx.y + 1][threadIdx.x + 3] * mask_large(m, 3, 1, 3);
        result += inputFeature_large[3][threadIdx.y + 1][threadIdx.x + 4] * mask_large(m, 3, 1, 4);

        result += inputFeature_large[3][threadIdx.y + 2][threadIdx.x + 0] * mask_large(m, 3, 2, 0);
        result += inputFeature_large[3][threadIdx.y + 2][threadIdx.x + 1] * mask_large(m, 3, 2, 1);
        result += inputFeature_large[3][threadIdx.y + 2][threadIdx.x + 2] * mask_large(m, 3, 2, 2);
        result += inputFeature_large[3][threadIdx.y + 2][threadIdx.x + 3] * mask_large(m, 3, 2, 3);
        result += inputFeature_large[3][threadIdx.y + 2][threadIdx.x + 4] * mask_large(m, 3, 2, 4);

        result += inputFeature_large[3][threadIdx.y + 3][threadIdx.x + 0] * mask_large(m, 3, 3, 0);
        result += inputFeature_large[3][threadIdx.y + 3][threadIdx.x + 1] * mask_large(m, 3, 3, 1);
        result += inputFeature_large[3][threadIdx.y + 3][threadIdx.x + 2] * mask_large(m, 3, 3, 2);
        result += inputFeature_large[3][threadIdx.y + 3][threadIdx.x + 3] * mask_large(m, 3, 3, 3);
        result += inputFeature_large[3][threadIdx.y + 3][threadIdx.x + 4] * mask_large(m, 3, 3, 4);

        result += inputFeature_large[3][threadIdx.y + 4][threadIdx.x + 0] * mask_large(m, 3, 4, 0);
        result += inputFeature_large[3][threadIdx.y + 4][threadIdx.x + 1] * mask_large(m, 3, 4, 1);
        result += inputFeature_large[3][threadIdx.y + 4][threadIdx.x + 2] * mask_large(m, 3, 4, 2);
        result += inputFeature_large[3][threadIdx.y + 4][threadIdx.x + 3] * mask_large(m, 3, 4, 3);
        result += inputFeature_large[3][threadIdx.y + 4][threadIdx.x + 4] * mask_large(m, 3, 4, 4);

        //c == 4
        result += inputFeature_large[4][threadIdx.y + 0][threadIdx.x + 0] * mask_large(m, 4, 0, 0);
        result += inputFeature_large[4][threadIdx.y + 0][threadIdx.x + 1] * mask_large(m, 4, 0, 1);
        result += inputFeature_large[4][threadIdx.y + 0][threadIdx.x + 2] * mask_large(m, 4, 0, 2);
        result += inputFeature_large[4][threadIdx.y + 0][threadIdx.x + 3] * mask_large(m, 4, 0, 3);
        result += inputFeature_large[4][threadIdx.y + 0][threadIdx.x + 4] * mask_large(m, 4, 0, 4);

        result += inputFeature_large[4][threadIdx.y + 1][threadIdx.x + 0] * mask_large(m, 4, 1, 0);
        result += inputFeature_large[4][threadIdx.y + 1][threadIdx.x + 1] * mask_large(m, 4, 1, 1);
        result += inputFeature_large[4][threadIdx.y + 1][threadIdx.x + 2] * mask_large(m, 4, 1, 2);
        result += inputFeature_large[4][threadIdx.y + 1][threadIdx.x + 3] * mask_large(m, 4, 1, 3);
        result += inputFeature_large[4][threadIdx.y + 1][threadIdx.x + 4] * mask_large(m, 4, 1, 4);

        result += inputFeature_large[4][threadIdx.y + 2][threadIdx.x + 0] * mask_large(m, 4, 2, 0);
        result += inputFeature_large[4][threadIdx.y + 2][threadIdx.x + 1] * mask_large(m, 4, 2, 1);
        result += inputFeature_large[4][threadIdx.y + 2][threadIdx.x + 2] * mask_large(m, 4, 2, 2);
        result += inputFeature_large[4][threadIdx.y + 2][threadIdx.x + 3] * mask_large(m, 4, 2, 3);
        result += inputFeature_large[4][threadIdx.y + 2][threadIdx.x + 4] * mask_large(m, 4, 2, 4);

        result += inputFeature_large[4][threadIdx.y + 3][threadIdx.x + 0] * mask_large(m, 4, 3, 0);
        result += inputFeature_large[4][threadIdx.y + 3][threadIdx.x + 1] * mask_large(m, 4, 3, 1);
        result += inputFeature_large[4][threadIdx.y + 3][threadIdx.x + 2] * mask_large(m, 4, 3, 2);
        result += inputFeature_large[4][threadIdx.y + 3][threadIdx.x + 3] * mask_large(m, 4, 3, 3);
        result += inputFeature_large[4][threadIdx.y + 3][threadIdx.x + 4] * mask_large(m, 4, 3, 4);

        result += inputFeature_large[4][threadIdx.y + 4][threadIdx.x + 0] * mask_large(m, 4, 4, 0);
        result += inputFeature_large[4][threadIdx.y + 4][threadIdx.x + 1] * mask_large(m, 4, 4, 1);
        result += inputFeature_large[4][threadIdx.y + 4][threadIdx.x + 2] * mask_large(m, 4, 4, 2);
        result += inputFeature_large[4][threadIdx.y + 4][threadIdx.x + 3] * mask_large(m, 4, 4, 3);
        result += inputFeature_large[4][threadIdx.y + 4][threadIdx.x + 4] * mask_large(m, 4, 4, 4);

        //c == 5
        result += inputFeature_large[5][threadIdx.y + 0][threadIdx.x + 0] * mask_large(m, 5, 0, 0);
        result += inputFeature_large[5][threadIdx.y + 0][threadIdx.x + 1] * mask_large(m, 5, 0, 1);
        result += inputFeature_large[5][threadIdx.y + 0][threadIdx.x + 2] * mask_large(m, 5, 0, 2);
        result += inputFeature_large[5][threadIdx.y + 0][threadIdx.x + 3] * mask_large(m, 5, 0, 3);
        result += inputFeature_large[5][threadIdx.y + 0][threadIdx.x + 4] * mask_large(m, 5, 0, 4);

        result += inputFeature_large[5][threadIdx.y + 1][threadIdx.x + 0] * mask_large(m, 5, 1, 0);
        result += inputFeature_large[5][threadIdx.y + 1][threadIdx.x + 1] * mask_large(m, 5, 1, 1);
        result += inputFeature_large[5][threadIdx.y + 1][threadIdx.x + 2] * mask_large(m, 5, 1, 2);
        result += inputFeature_large[5][threadIdx.y + 1][threadIdx.x + 3] * mask_large(m, 5, 1, 3);
        result += inputFeature_large[5][threadIdx.y + 1][threadIdx.x + 4] * mask_large(m, 5, 1, 4);

        result += inputFeature_large[5][threadIdx.y + 2][threadIdx.x + 0] * mask_large(m, 5, 2, 0);
        result += inputFeature_large[5][threadIdx.y + 2][threadIdx.x + 1] * mask_large(m, 5, 2, 1);
        result += inputFeature_large[5][threadIdx.y + 2][threadIdx.x + 2] * mask_large(m, 5, 2, 2);
        result += inputFeature_large[5][threadIdx.y + 2][threadIdx.x + 3] * mask_large(m, 5, 2, 3);
        result += inputFeature_large[5][threadIdx.y + 2][threadIdx.x + 4] * mask_large(m, 5, 2, 4);

        result += inputFeature_large[5][threadIdx.y + 3][threadIdx.x + 0] * mask_large(m, 5, 3, 0);
        result += inputFeature_large[5][threadIdx.y + 3][threadIdx.x + 1] * mask_large(m, 5, 3, 1);
        result += inputFeature_large[5][threadIdx.y + 3][threadIdx.x + 2] * mask_large(m, 5, 3, 2);
        result += inputFeature_large[5][threadIdx.y + 3][threadIdx.x + 3] * mask_large(m, 5, 3, 3);
        result += inputFeature_large[5][threadIdx.y + 3][threadIdx.x + 4] * mask_large(m, 5, 3, 4);

        result += inputFeature_large[5][threadIdx.y + 4][threadIdx.x + 0] * mask_large(m, 5, 4, 0);
        result += inputFeature_large[5][threadIdx.y + 4][threadIdx.x + 1] * mask_large(m, 5, 4, 1);
        result += inputFeature_large[5][threadIdx.y + 4][threadIdx.x + 2] * mask_large(m, 5, 4, 2);
        result += inputFeature_large[5][threadIdx.y + 4][threadIdx.x + 3] * mask_large(m, 5, 4, 3);
        result += inputFeature_large[5][threadIdx.y + 4][threadIdx.x + 4] * mask_large(m, 5, 4, 4);

        y4d(n, m, h, w) = result;   //update the output element
    }


    #undef y4d
    #undef x4d
    #undef k4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    int B = x.shape_[0];  //additional dimension to the tensors to support an entire mini-batch
    int M = y.shape_[1];  //number of output feature maps
    int C = x.shape_[1];  //number of input feature maps
    int H = x.shape_[2];  //height of each input feature maps
    int W = x.shape_[3];  //width of each input feature maps
    int K = w.shape_[3];  //height/width of each filter bank

    int size_flag;   //0 for no constant memory, 1 for mask_small, 2 for mask_large;

    int H_OUT = H - K + 1;  // output feature height
    int W_OUT = W - K + 1;  // output feature width

    // Set the kernel dimensions
    int W_GRID = (W_OUT-1) / TILE_WIDTH + 1;
    int H_GRID = (H_OUT-1) / TILE_WIDTH + 1;
    int Z = H_GRID * W_GRID;                    //number of blocks needed for one output feature processing
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);                      //minibatch, output feature, tile in that output feature

    //copy mask data into constant memory
    if (M == SMALL_OUTPUT) {
        cudaMemcpyToSymbol(mask_small, w.dptr_, SMALL_OUTPUT*SMALL_INPUT*MASK_WIDTH*MASK_WIDTH*sizeof(float));
        size_flag = SMALL_FLAG;
    }
    else if (M == LARGE_OUTPUT) {
        cudaMemcpyToSymbol(mask_large, w.dptr_, LARGE_OUTPUT*LARGE_INPUT*MASK_WIDTH*MASK_WIDTH*sizeof(float));
        size_flag = LARGE_FLAG;
    }
    else {
        size_flag = 0;
    }

    // printf("B: %d\nM: %d\nC: %d\nH: %d\nW: %d\nK: %d\n", B, M, C, H, W, K);
    // printf("size_flag is: %d\n", size_flag);

    // Call the kernel
    //forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    if (size_flag == SMALL_FLAG){
        forward_kernel_small<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,B,M,C,H,W,K);
    }
    else if (size_flag == LARGE_FLAG) {
        forward_kernel_large<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,B,M,C,H,W,K);
    }

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
