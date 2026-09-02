#include "minijax/frontend.hpp"

#include <cctype>
#include <cstdlib>
#include <map>

namespace minijax::frontend {

const char* tok_kind_name(TokKind k) {
    switch (k) {
        case TokKind::Ident: return "identifier";
        case TokKind::Number: return "number";
        case TokKind::KwLet: return "'let'";
        case TokKind::KwOutput: return "'output'";
        case TokKind::LParen: return "'('";
        case TokKind::RParen: return "')'";
        case TokKind::LBracket: return "'['";
        case TokKind::RBracket: return "']'";
        case TokKind::Comma: return "','";
        case TokKind::Semi: return "';'";
        case TokKind::Equal: return "'='";
        case TokKind::Plus: return "'+'";
        case TokKind::Minus: return "'-'";
        case TokKind::Star: return "'*'";
        case TokKind::Slash: return "'/'";
        case TokKind::At: return "'@'";
        case TokKind::Apostrophe: return "'";
        case TokKind::End: return "end of input";
    }
    return "?";
}

std::vector<Token> lex(const std::string& src) {
    std::vector<Token> toks;
    int line = 1;
    size_t i = 0;
    auto err = [&](char c) -> std::invalid_argument {
        return std::invalid_argument("lex error at line " + std::to_string(line) +
                                     ": unexpected character '" + std::string(1, c) + "'");
    };

    while (i < src.size()) {
        char c = src[i];
        if (c == '\n') { ++line; ++i; continue; }
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        if (c == '#') {
            while (i < src.size() && src[i] != '\n') ++i;
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t start = i;
            while (i < src.size() &&
                   (std::isalnum(static_cast<unsigned char>(src[i])) || src[i] == '_')) ++i;
            std::string word = src.substr(start, i - start);
            if (word == "let") toks.push_back({TokKind::KwLet, word, 0.0, line});
            else if (word == "output") toks.push_back({TokKind::KwOutput, word, 0.0, line});
            else toks.push_back({TokKind::Ident, word, 0.0, line});
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) ||
            (c == '.' && i + 1 < src.size() && std::isdigit(static_cast<unsigned char>(src[i + 1])))) {
            size_t start = i;
            while (i < src.size() && (std::isdigit(static_cast<unsigned char>(src[i])) || src[i] == '.')) ++i;

            if (i < src.size() && (src[i] == 'e' || src[i] == 'E')) {
                size_t save = i;
                ++i;
                if (i < src.size() && (src[i] == '+' || src[i] == '-')) ++i;
                if (i < src.size() && std::isdigit(static_cast<unsigned char>(src[i]))) {
                    while (i < src.size() && std::isdigit(static_cast<unsigned char>(src[i]))) ++i;
                } else {
                    i = save;
                }
            }
            std::string text = src.substr(start, i - start);
            toks.push_back({TokKind::Number, text, std::strtod(text.c_str(), nullptr), line});
            continue;
        }
        auto push1 = [&](TokKind k) { toks.push_back({k, std::string(1, c), 0.0, line}); ++i; };
        switch (c) {
            case '(': push1(TokKind::LParen); break;
            case ')': push1(TokKind::RParen); break;
            case '[': push1(TokKind::LBracket); break;
            case ']': push1(TokKind::RBracket); break;
            case ',': push1(TokKind::Comma); break;
            case ';': push1(TokKind::Semi); break;
            case '=': push1(TokKind::Equal); break;
            case '+': push1(TokKind::Plus); break;
            case '-': push1(TokKind::Minus); break;
            case '*': push1(TokKind::Star); break;
            case '/': push1(TokKind::Slash); break;
            case '@': push1(TokKind::At); break;
            case '\'': push1(TokKind::Apostrophe); break;
            default: throw err(c);
        }
    }
    toks.push_back({TokKind::End, "", 0.0, line});
    return toks;
}

}
