module;

#include <string>

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

    std::size_t remove_all(std::string& inout, std::string_view what) {
        return replace_all(inout, what, "");
    }

}
