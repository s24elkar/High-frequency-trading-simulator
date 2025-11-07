#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <array>
#include <sstream>
#include <utility>
#include <type_traits>

namespace spdlog {

inline std::string escape_json(std::string_view input) {
    std::string output;
    output.reserve(input.size() + 8);
    for (char ch : input) {
        switch (ch) {
            case '"':
                output.append("\\\"");
                break;
            case '\\':
                output.append("\\\\");
                break;
            case '\b':
                output.append("\\b");
                break;
            case '\f':
                output.append("\\f");
                break;
            case '\n':
                output.append("\\n");
                break;
            case '\r':
                output.append("\\r");
                break;
            case '\t':
                output.append("\\t");
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    std::ostringstream oss;
                    oss << "\\u" << std::hex << std::uppercase << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch));
                    output.append(oss.str());
                } else {
                    output.push_back(ch);
                }
        }
    }
    return output;
}

inline std::string iso_timestamp() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const auto secs = clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &secs);
#else
    gmtime_r(&secs, &tm);
#endif
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1'000'000;
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S")
        << '.' << std::setw(6) << std::setfill('0') << micros.count() << 'Z';
    return oss.str();
}

template <typename T>
std::string to_string_any(T&& value) {
    if constexpr (std::is_convertible_v<T, std::string_view>) {
        return std::string(value);
    } else if constexpr (std::is_convertible_v<T, const char*>) {
        return std::string(value);
    } else {
        std::ostringstream oss;
        oss << std::forward<T>(value);
        return oss.str();
    }
}

template <typename... Args>
std::string basic_format(std::string_view fmt, Args&&... args) {
    std::array<std::string, sizeof...(Args)> formatted{to_string_any(std::forward<Args>(args))...};
    std::string output;
    output.reserve(fmt.size() + 32 * sizeof...(Args));
    std::size_t arg_index = 0;
    for (std::size_t i = 0; i < fmt.size(); ++i) {
        if (fmt[i] == '{' && i + 1 < fmt.size() && fmt[i + 1] == '}' && arg_index < formatted.size()) {
            output.append(formatted[arg_index++]);
            ++i;
        } else {
            output.push_back(fmt[i]);
        }
    }
    return output;
}

class logger {
public:
    logger(std::string name, const std::filesystem::path& target_path)
        : name_(std::move(name)),
          stream_(target_path, std::ios::app) {}

    template <typename... Args>
    void info(std::string_view fmt, Args&&... args) {
        log("info", fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warn(std::string_view fmt, Args&&... args) {
        log("warn", fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error(std::string_view fmt, Args&&... args) {
        log("error", fmt, std::forward<Args>(args)...);
    }

    void flush() {
        std::scoped_lock lock(mutex_);
        stream_.flush();
    }

private:
    template <typename... Args>
    void log(std::string_view level, std::string_view fmt, Args&&... args) {
        if (!stream_.is_open()) {
            return;
        }
        const std::string message = basic_format(fmt, std::forward<Args>(args)...);
        const std::string payload = escape_json(message);
        const std::string ts = iso_timestamp();
        std::scoped_lock lock(mutex_);
        stream_ << "{\"ts\":\"" << ts << "\","
                << "\"logger\":\"" << name_ << "\","
                << "\"level\":\"" << level << "\","
                << "\"message\":\"" << payload << "\"}\n";
    }

    std::string name_;
    std::ofstream stream_;
    std::mutex mutex_;
};

inline std::shared_ptr<logger>& default_logger_ref() {
    static std::shared_ptr<logger> logger_instance;
    return logger_instance;
}

inline void set_default_logger(std::shared_ptr<logger> new_logger) {
    default_logger_ref() = std::move(new_logger);
}

inline std::shared_ptr<logger> default_logger() {
    return default_logger_ref();
}

inline std::shared_ptr<logger> basic_logger_mt(const std::string& name, const std::filesystem::path& filepath) {
    std::filesystem::create_directories(filepath.parent_path());
    return std::make_shared<logger>(name, filepath);
}

template <typename... Args>
void info(std::string_view fmt, Args&&... args) {
    if (auto lg = default_logger()) {
        lg->info(fmt, std::forward<Args>(args)...);
    }
}

template <typename... Args>
void warn(std::string_view fmt, Args&&... args) {
    if (auto lg = default_logger()) {
        lg->warn(fmt, std::forward<Args>(args)...);
    }
}

template <typename... Args>
void error(std::string_view fmt, Args&&... args) {
    if (auto lg = default_logger()) {
        lg->error(fmt, std::forward<Args>(args)...);
    }
}

inline void flush() {
    if (auto lg = default_logger()) {
        lg->flush();
    }
}

} // namespace spdlog
