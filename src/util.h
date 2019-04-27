//
//  util.h
//  AlphaZero
//
//  Created by Semin Park on 09/03/2019.
//  Copyright © 2019 Semin Park. All rights reserved.
//

#ifndef util_h
#define util_h

#include <algorithm>
#include <csignal>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <thread>
#include <vector>

#include <torch/torch.h>

#include "network.hpp"


void msg(std::string str) {
    std::cout << str << std::endl;
}

void sighandler(int n)
{
    std::stringstream ss;
    ss << "Signal: " << n << " | Thread ID: " << std::this_thread::get_id() << std::endl;
    std::cout << ss.str();
    std::abort();
}

std::vector<float> dirichlet(int size, float alpha = 0.05)
{
    static std::mt19937 rng = std::mt19937(std::random_device{}());
    std::gamma_distribution<float> dist(alpha);
    
    std::vector<float> vec;
    vec.reserve(size);
    
    for (int i = 0; i < size; i++) {
        vec.push_back(dist(rng));
    }
    
    double sum = std::accumulate(vec.begin(), vec.end(), 0.);
    for (float& f : vec)
        f /= sum;
    return vec;
}

std::string load_network(PVNetwork& net, const std::string& path = "")
{
    std::cout << "Loading network." << std::endl;
    int fd = open("ckpt_location.txt", O_RDONLY);
    flock(fd, LOCK_SH);
    
    std::ifstream dir("ckpt_location.txt");

    if (!dir.is_open()) {
        char *cwd = getcwd(nullptr, 0);

        std::stringstream ss;
        ss << std::endl << "* ERROR *" << std::endl
           << "\"ckpt_location.txt\" must be in the working directory." << std::endl
           << "Furthermore, the path that \"ckpt_location.txt\" points to must exist." << std::endl
           << "Current working directory: " << cwd << std::endl;

        free(cwd);
        throw std::runtime_error(ss.str());
    }

    std::string export_path;
    getline(dir, export_path);
    dir.close();
    flock(fd, LOCK_UN);

    if (export_path == path) {
        std::cout << "No change in network. Resuming with network " << path << std::endl;
        return path;
    }

    std::ifstream file(export_path);
    if (!file.good()) {
        std::cout << "Model path provided by 'ckpt_location.txt' doesn't exist. Creating a new one." << std::endl;
        torch::save(net, export_path);
    } else {
        torch::load(net, export_path);
        std::cout << "Network loaded." << std::endl << "Path: " << export_path << std::endl;
    }
    return export_path;
}


std::string save_network(PVNetwork& net, const std::string& path)
{
    int idx = path.find("_");
    auto dot = path.find(".") - 1;
    auto diff = dot - idx;

    auto ver = path.substr(idx+1, diff);
    int version = std::atoi(ver.c_str()) + 1;
    std::cout << std::endl << "version: " << version << std::endl;


    std::string new_path = path.substr(0, idx + 1) + std::to_string(version) + ".pt";
    torch::save(net, new_path);
    std::cout << "Saved at " << new_path << std::endl;

    int fd = open("ckpt_location.txt", O_WRONLY);
    flock(fd, LOCK_EX);
    
    std::ofstream file("ckpt_location.txt");
    if (!file.is_open()) {
        flock(fd, LOCK_UN);
        throw std::runtime_error("ckpt_location does not exist. Did you remove it by accident?");
    }

    file << new_path;
    flock(fd, LOCK_UN);

    return new_path;
}

void visualize(torch::Tensor policy)
{
    static std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    int size = policy.sizes()[1];

    std::cout << "   ";
    for (char c : alphabet.substr(0,size))
        std::cout << c << ' ';
    std::cout << std::endl;

    auto policy_a = policy.accessor<float, 3>();
    for (int i = 0; i < size; i++) {
        std::cout << std::setw(2) << i << ' ';
        for (int j = 0; j < size; j++) {
            int lvl = (int) (policy_a[0][i][j] * 10);
            if (lvl != 0)
                std::cout << lvl << ' ';
            else
                std::cout << '.' << ' ';
        }
        std::cout << std::endl;
    }
}


#endif /* util_h */