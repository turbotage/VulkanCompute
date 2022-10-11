module;

export module util;

import <string>;
import <vector>;
import <algorithm>;
import <locale>;
import <functional>;
import <random>;
import <chrono>;
import <memory>;
import <fstream>;
import <ostream>;
import <map>;
import <iomanip>;

namespace util {

    export std::size_t replace_all(std::string& inout, std::string_view what, std::string_view with)
    {
        std::size_t count{};
        for (std::string::size_type pos{};
            inout.npos != (pos = inout.find(what.data(), pos, what.length()));
            pos += with.length(), ++count) {
            inout.replace(pos, what.length(), with.data(), with.length());
        }
        return count;
    }

    export std::size_t remove_all(std::string& inout, std::string_view what) {
        return replace_all(inout, what, "");
    }

    export template<typename KeyType, typename HashFunc = std::hash<KeyType>>
    concept Hashable = std::regular_invocable<HashFunc, KeyType>
        && std::convertible_to<std::invoke_result_t<HashFunc, KeyType>, std::size_t>;

    export template <typename KeyType, typename HashFunc = std::hash<KeyType>> requires Hashable<KeyType, HashFunc>
    inline std::size_t hash_combine(const std::size_t& seed, const KeyType& v)
    {
        HashFunc hasher;
        std::size_t ret = seed;
        ret ^= hasher(v) + 0x9e3779b9 + (ret << 6) + (ret >> 2);
        return ret;
    }

    export template <typename KeyType, typename HashFunc = std::hash<KeyType>> requires Hashable<KeyType, HashFunc>
    std::size_t hash_combine(const std::vector<KeyType>& hashes)
    {
        std::size_t ret = hashes.size() > 0 ? hashes.front() : throw std::runtime_error("Can't hash_combine an empty vector");
        for (int i = 1; i < hashes.size(); ++i) {
            ret = hash_combine(ret, hashes[i]);
        }
        return ret;
    }

    export std::string to_lower_case(const std::string& str) {
        std::string ret = str;
        std::transform(ret.begin(), ret.end(), ret.begin(), [](unsigned char c) { return std::tolower(c); });
        return ret;
    }

    export std::string add_whitespace_until(const std::string& str, int until) {
        if (str.size() > until) {
            return std::string(str.begin(), str.begin() + until);
        }

        std::string ret = str;
        ret.reserve(until);
        for (int i = str.size(); i <= until; ++i) {
            ret += ' ';
        }
        return ret;
    }

    export std::string add_after_newline(const std::string& str, const std::string& adder, bool add_start = true)
    {
        std::string ret = str;
        if (add_start) {
            ret.insert(0, adder);
        }
        for (int i = 0; i < ret.size(); ++i) {
            if (ret[i] == '\n') {
                if (i + 1 > ret.size())
                    return ret;
                ret.insert(i + 1, adder);
                i += adder.size() + 2;
            }
        }
        return ret;
    }

    export std::string add_line_numbers(const std::string& str, int max_number_length = 5) {
        int until = max_number_length;
        std::string ret = str;
        ret.insert(0, util::add_whitespace_until(std::to_string(1), until) + "\t|");
        int k = 2;
        for (int i = until; i < ret.size(); ++i) {
            if (ret[i] == '\n') {
                if (i + 1 > ret.size())
                    return ret;
                ret.insert(i+1, util::add_whitespace_until(std::to_string(k), until) + "\t|");
                i += until+2;
                ++k;
            }
        }
        return ret;
    }

    export std::string remove_whitespace(const std::string& str) {
        std::string ret = str;
        ret.erase(std::remove_if(ret.begin(), ret.end(),
            [](char& c) {
                return std::isspace<char>(c, std::locale::classic());
            }),
            ret.end());
        return ret;
    }

    export template<typename Container> requires std::ranges::range<Container>
    bool container_contains(const Container& c, typename Container::const_reference v)
    {
        return std::find(c.begin(), c.end(), v) != c.end();
    }

    export std::string stupid_compress(uint64_t num) 
    {
        std::string basec = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        std::string ret;

        auto powlam = [](uint64_t base, uint32_t exponent) {
            uint64_t retnum = 1;
            for (int i = 0; i < exponent; ++i) {
                retnum *= base;
            }
            return retnum;
        };

        uint64_t base = std::numeric_limits<uint64_t>::max();
        uint64_t c = (uint64_t)num / base;
        uint64_t rem = num % base;

        for (int i = 10; i >= 0; --i) {
            base = powlam(basec.size(), i);
            c = (uint64_t)num / base;
            rem = num % base;

            if (c > 0)
                ret += basec[c];
            num = rem;
        }

        return ret;
    }

    export void add_n_str(std::string& str, const std::string& adder, int n)
    {
        for (int i = 0; i < n; ++i) {
            str += adder;
        }
    }

}
