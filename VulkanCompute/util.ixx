module;

#include <string>
#include <vector>

export module util;

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
        std::size_t ret = hashes.size();
        for (int i = 0; i < hashes.size(); ++i) {
            ret = hash_combine(ret, hashes[i]);
        }
        return ret;
    }

}
