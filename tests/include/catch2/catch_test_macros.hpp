#pragma once

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace minimal_catch {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(std::string name, std::function<void()> fn) {
        registry().push_back(TestCase{std::move(name), std::move(fn)});
    }
};

inline void fail(const char* expr, const char* file, int line) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " assertion failed: " + expr);
}

inline void log_failure(const char* expr, const char* file, int line) {
    std::cerr << file << ":" << line << " CHECK failed: " << expr << std::endl;
}

inline int run_all() {
    int failures = 0;
    for (const auto& test : registry()) {
        try {
            test.fn();
        } catch (const std::exception& ex) {
            ++failures;
            std::cerr << "[FAILED] " << test.name << " — " << ex.what() << std::endl;
        } catch (...) {
            ++failures;
            std::cerr << "[FAILED] " << test.name << " — unknown exception" << std::endl;
        }
    }
    const int total = static_cast<int>(registry().size());
    if (failures == 0) {
        std::cout << "[PASSED] " << total << " tests" << std::endl;
    } else {
        std::cerr << "[SUMMARY] " << failures << "/" << total << " tests failed" << std::endl;
    }
    return failures == 0 ? 0 : 1;
}

} // namespace minimal_catch

#define MC_CONCAT_IMPL(a, b) a##b
#define MC_CONCAT(a, b) MC_CONCAT_IMPL(a, b)

#define TEST_CASE(name) \
    static void MC_CONCAT(test_case_fn_, __LINE__)(); \
    static minimal_catch::Registrar MC_CONCAT(test_case_reg_, __LINE__)(name, MC_CONCAT(test_case_fn_, __LINE__)); \
    static void MC_CONCAT(test_case_fn_, __LINE__)()

#define REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            minimal_catch::fail(#condition, __FILE__, __LINE__); \
        } \
    } while (false)

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            minimal_catch::log_failure(#condition, __FILE__, __LINE__); \
            minimal_catch::fail(#condition, __FILE__, __LINE__); \
        } \
    } while (false)

#define CHECK_FALSE(condition) CHECK(!(condition))
