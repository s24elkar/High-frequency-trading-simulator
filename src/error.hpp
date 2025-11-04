#pragma once

#include <cstdlib>
#include <exception>
#include <iostream>
#include <utility>

namespace simulator::detail {

template <typename Exception>
[[noreturn]] void throw_or_abort(Exception&& ex, const char* file, int line) {
#if defined(__cpp_exceptions)
    throw std::forward<Exception>(ex);
#else
    std::cerr << "fatal: " << ex.what() << " (" << file << ':' << line << ")\n";
    std::abort();
#endif
}

} // namespace simulator::detail

#if defined(__cpp_exceptions)
#define HFT_THROW(expr) throw expr
#else
#define HFT_THROW(expr) ::simulator::detail::throw_or_abort((expr), __FILE__, __LINE__)
#endif

