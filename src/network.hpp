#ifndef network_hpp
#define network_hpp

#include <iostream>
#include <tuple>
#include <vector>

#include <torch/torch.h>

struct ConvBlockImpl : torch::nn::Module {
    ConvBlockImpl(int in_channel, int C /* channels out */, int K, int P)
      : conv(register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, C, K).padding(P)))),
        batch_norm(register_module("bn", torch::nn::BatchNorm(C))) { /* pass */ }

    torch::Tensor forward(const torch::Tensor& x){
        return torch::relu(batch_norm(conv(x)));
    }

    torch::nn::Conv2d conv;
    torch::nn::BatchNorm batch_norm;
};
TORCH_MODULE(ConvBlock);


struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(int C, int K, int P)
      : conv(register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(C, C, K).padding(P)))),
        batch_norm(register_module("bn", torch::nn::BatchNorm(C))) { /* pass */ }

    torch::Tensor forward(const torch::Tensor& input)
    {
        torch::Tensor x = torch::relu(batch_norm(conv(input)));
        x = batch_norm(conv(x));
        x = torch::relu(x + input);
        return x;
    }

    torch::nn::Conv2d conv;
    torch::nn::BatchNorm batch_norm;
};
TORCH_MODULE(ResBlock);


struct PolicyHeadImpl : torch::nn::Module {
    PolicyHeadImpl(int C, int out)
      : conv(register_module("conv", torch::nn::Conv2d(C, out, 1))) { /* pass */ }
    
    torch::Tensor forward(const torch::Tensor& input)
    {
        torch::Tensor x = conv(input);
        auto shape = x.sizes();
        x = torch::softmax(x.flatten(1), 1).view(shape);
        return x;
    }
    torch::nn::Conv2d conv;
};
TORCH_MODULE(PolicyHead);


struct ValueHeadImpl : torch::nn::Module {
    ValueHeadImpl(int C, int size_2)
      : value_conv(register_module("conv", torch::nn::Conv2d(C, 1, 1))),
        value_bn(register_module("bn", torch::nn::BatchNorm(1))),
        value_fc1(register_module("fc1", torch::nn::Linear(size_2, 64))),
        value_fc2(register_module("fc2", torch::nn::Linear(64, 2))) { /* pass */ }
    
    torch::Tensor forward(const torch::Tensor& input)
    {
        torch::Tensor x = torch::relu(value_bn(value_conv(input)));  // size x size
        x = x.flatten(1);  // size^2
        x = torch::relu(value_fc1(x));
        x = value_fc2(x);
        x = torch::softmax(x, 1) * 2 - 1;
        return x;
    }

    torch::nn::Conv2d value_conv;
    torch::nn::BatchNorm value_bn;
    torch::nn::Linear value_fc1;
    torch::nn::Linear value_fc2;
};
TORCH_MODULE(ValueHead);

/*
 * Actual network used in AGZ.
 *
 * board_size: Size of the board (3 in case of TicTacToe)
 * n_res     : # of ResBlock
 * in        : # of input channels
 * C         : # of intermediate channels
 * out       : # of output channels
 * K         : Kernel size
 * P         : Padding (only applied where K == 3)
 */
struct PVNetworkImpl : torch::nn::Module {
    PVNetworkImpl(int board_size, const std::vector<int>& Cs, int in, int out, int K = 3, int P = 1)
      : num_res(Cs.size()),
        CBlock(register_module("CBlock", ConvBlock(in, Cs[0], K, P))),
        PHead(register_module("PHead", PolicyHead(Cs[num_res - 1], out))),
        VHead(register_module("VHead", ValueHead(Cs[num_res - 1], board_size*board_size)))
    {
        for (int i = 0; i < num_res; i++) {
            int C = Cs[i];
            RBlocks.emplace_back(register_module("RBlock_" + std::to_string(i), ResBlock(C, K, P)));
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input)
    {
        torch::Tensor x = CBlock(input);
        for (int i = 0; i < num_res; i++)
            x = RBlocks[i](x);
        return std::make_tuple(PHead(x), VHead(x));
    }

    int num_res;

    ConvBlock CBlock;
    std::vector<ResBlock> RBlocks;
    PolicyHead PHead;
    ValueHead VHead;
};
TORCH_MODULE(PVNetwork);

#endif // network_hpp