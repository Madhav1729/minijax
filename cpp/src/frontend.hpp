#pragma once

#include <cctype>
#include <string>
#include <vector>

#include "ir.hpp"

namespace minijax {

inline Graph parse_text_program(const std::string& text) {
    (void)text;
    return Graph{};
}

inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> out;
    std::string token;
    for (char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!token.empty()) {
                out.push_back(token);
                token.clear();
            }
        } else {
            token.push_back(ch);
        }
    }
    if (!token.empty()) out.push_back(token);
    return out;
}

}  // namespace minijax
