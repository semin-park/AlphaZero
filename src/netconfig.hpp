#ifndef netconfig_hpp
#define netconfig_hpp

class NetConfig
{
public:
    static NetConfig& get()
    {
        static NetConfig singleton;
        return singleton;
    }

    int resblocks()
    {
        return n_res;
    }

    int channels()
    {
        return C;
    }
private:
    NetConfig() {
        n_res = 20;
        C = 32;
    }
    int n_res, C;
};

#endif // netconfig_hpp