
#ifndef MXNET_OPERATOR_NEW_FORWARD_H_
#define MXNET_OPERATOR_NEW_FORWARD_H_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


template <typename cpu, typename DType>
void forward(mshadow::Tensor<cpu, 4, DType> &y, const mshadow::Tensor<cpu, 4, DType> &x, const mshadow::Tensor<cpu, 4, DType> &k)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct, not fast (this is the CPU implementation.)
    */

    const int B = x.shape_[0];  //additional dimension to the tensors to support an entire mini-batch
    const int M = y.shape_[1];  //number of output feature maps
    const int C = x.shape_[1];  //number of input feature maps
    const int H = x.shape_[2];  //height of each input feature maps
    const int W = x.shape_[3];  //width of each input feature maps
    const int K = k.shape_[3];  //height/width of each filter bank

    int H_OUT = H - K + 1;  //height of each output feature maps
    int W_OUT = W - K + 1;  //width of each output feature maps

    for (int b = 0; b < B; ++b) { //outmost loop for each mini-batch
      for (int m = 0; m < M; m++) {               //loop for each output feature map
        for (int h = 0; h < H_OUT; h++) {         //loop for every pixel of one output feature map
          for (int w = 0; w < W_OUT; w++) {
            y[b][m][h][w] = 0;                    //clear to 0
            for (int c = 0; c < C; c++) {         //loop for each input feature maps
              for (int p = 0; p < K; p++) {       //loop for each element in the filter bank
                for (int q = 0; q < K; q++) {
                  y[b][m][h][w] += x[b][c][h+p][w+q] * k[m][c][p][q]; //update the output pixel
                }
              }
            }
          }
        }
      }
    }
    //original comments:
    //CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
    /* ... a bunch of nested loops later...
        y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
    */

}
}
}

#endif
