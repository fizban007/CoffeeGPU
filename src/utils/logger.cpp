#include "utils/logger.h"

namespace Aperture {

int Logger::m_rank = 0;
LogLevel Logger::m_level = LogLevel::info;
std::string Logger::m_log_file = "";
std::FILE* Logger::m_file = nullptr;

void
Logger::init(int rank, LogLevel level, std::string log_file) {
  m_rank = rank;
  m_level = level;
  m_log_file = log_file;
}

Logger::~Logger() {
  if (m_file != nullptr) {
    fclose(m_file);
  }
}

bool
Logger::open_log_file() {
  m_file = std::fopen(m_log_file.c_str(), "w");
  if (!m_file) {
    print_err("Can't open log file, unable to log to file\n");
    return false;
  }
  return true;
}

}  // namespace Aperture
