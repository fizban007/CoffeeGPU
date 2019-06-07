#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <cstdio>
#include <string>
#include <fmt/core.h>

namespace Coffee {

enum class LogLevel : char { info, detail, debug };

class Logger {
 private:
  static int m_rank;
  static LogLevel m_level;
  static std::string m_log_file;
  static std::FILE* m_file;

 public:
  Logger() {}
  ~Logger();

  static void init(int rank, LogLevel level, std::string log_file);
  static bool open_log_file();

  template <typename... Args>
  static void err(const char* str, Args&&... args) {
    fmt::print(stderr, str, std::forward<Args>(args)...);
    fmt::print("\n");
  }

  template <typename... Args>
  static void print_err(const char* str, Args&&... args) {
    if (m_rank == 0) {
      fmt::print(stderr, str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void print_info(const char* str, Args&&... args) {
    if (m_rank == 0) {
      fmt::print(str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void print_detail(const char* str, Args&&... args) {
    if (m_rank == 0 && m_level > LogLevel::info) {
      fmt::print(str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void print_debug(const std::string& str, Args&&... args) {
    if (m_rank == 0 && m_level > LogLevel::detail) {
      fmt::print("Debug: " + str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void print_debug_all(const std::string& str, Args&&... args) {
    if (m_level > LogLevel::detail) {
      fmt::print("Debug: " + str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void log_info(const char* str, Args&&... args) {
    if (m_rank == 0) {
      if (m_file == nullptr)
        if (!open_log_file()) {
          fmt::print("File can't be opened!");
          return;
        }
      fmt::print(m_file, str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void log_detail(const char* str, Args&&... args) {
    if (m_rank == 0 && m_level > LogLevel::info) {
      if (m_file == nullptr)
        if (!open_log_file()) return;
      fmt::print(m_file, str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void log_debug(const std::string& str, Args&&... args) {
    if (m_rank == 0 && m_level > LogLevel::detail) {
      if (m_file == nullptr)
        if (!open_log_file()) return;
      fmt::print(m_file, "Debug: " + str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }

  template <typename... Args>
  static void log_debug_all(const std::string& str, Args&&... args) {
    if (m_file == nullptr)
      if (!open_log_file()) return;
    if (m_level > LogLevel::detail) {
      fmt::print(m_file, "Debug: " + str, std::forward<Args>(args)...);
      fmt::print("\n");
    }
  }
};

}  // namespace Aperture

#endif  // _LOGGER_H_
