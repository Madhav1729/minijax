#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace minijax {

inline int run_cli(const std::vector<std::string>& args) {
    std::cout << "minijax-cpp CLI skeleton\n";
    for (const auto& arg : args) {
        std::cout << "arg: " << arg << '\n';
    }
    return 0;
}

}  // namespace minijax
