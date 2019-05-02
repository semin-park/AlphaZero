#ifndef netconfig_hpp
#define netconfig_hpp

#include <vector>

class NetConfig
{
public:
    static NetConfig& get(int n)
    {
        static NetConfig singleton(n);
        return singleton;
    }

    std::vector<int> resblocks()
    {
        return Cs[id];
    }
private:
    NetConfig(int n) : id(n) {
        /*
         * 0. [  32  32 32 32 32 32 ... ] => 20 ResBlocks == previous default
         * 1. [  32  32 32 32 32 32 ... ] => 10 ResBlocks
         * 2. [ 128 128 128 128 128 ... ] => 10 ResBlocks
         * 3. [ 256 128 64 32 16  8  4  2 ] => 8 ResBlocks
         * 4. [ 128 128 64 64 32 32 16 16 ] => 8 ResBlocks
         */

        std::vector<int> C0, C1, C2, C3, C4;

        // Number 0
        for (int i = 0; i < 20; i++)
            C0.emplace_back(32);

        // Number 1
        for (int i = 0; i < 10; i++)
            C1.emplace_back(32);

        // Number 2
        for (int i = 0; i < 10; i++)
            C2.emplace_back(128);

        // Number 3
        C3 = { 256, 128, 64, 32, 16, 8, 4, 2 };

        // Number 4
        C4 = { 128, 128, 64, 64, 32, 32, 16, 16 };

        Cs = { C0, C1, C2, C3, C4 };

    }
    int id;
    std::vector<std::vector<int>> Cs;
};

#endif // netconfig_hpp