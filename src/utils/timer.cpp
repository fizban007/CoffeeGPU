#include "utils/timer.h"

using namespace Aperture;
using namespace std::chrono;

std::unordered_map<std::string, high_resolution_clock::time_point>
    timer::t_stamps;
high_resolution_clock::time_point timer::t_now =
    high_resolution_clock::now();

void
timer::stamp(const std::string& name) {
  t_stamps[name] = high_resolution_clock::now();
}

void
timer::show_duration_since_stamp(const std::string& routine_name,
                                 const std::string& unit,
                                 const std::string& stamp_name) {
  t_now = high_resolution_clock::now();
  if (routine_name == "" && stamp_name == "") {
    std::cout << "--- Time for default clock is ";
  } else if (routine_name == "") {
    std::cout << "--- Time for " << stamp_name << " is ";
  } else {
    std::cout << "--- Time for " << routine_name << " is ";
  }
  if (unit == "second" || unit == "s") {
    auto dur = duration_cast<duration<float, std::ratio<1, 1>>>(
        t_now - t_stamps[stamp_name]);
    std::cout << dur.count() << "s" << std::endl;
  } else if (unit == "millisecond" || unit == "ms") {
    auto dur =
        duration_cast<milliseconds>(t_now - t_stamps[stamp_name]);
    std::cout << dur.count() << "ms" << std::endl;
  } else if (unit == "microsecond" || unit == "us") {
    auto dur =
        duration_cast<microseconds>(t_now - t_stamps[stamp_name]);
    std::cout << dur.count() << "µs" << std::endl;
  } else if (unit == "nanosecond" || unit == "ns") {
    auto dur = duration_cast<nanoseconds>(t_now - t_stamps[stamp_name]);
    std::cout << dur.count() << "ns" << std::endl;
  }
}

float
timer::get_duration_since_stamp(const std::string& unit,
                                const std::string& stamp_name) {
  t_now = high_resolution_clock::now();
  if (unit == "millisecond" || unit == "ms") {
    auto dur =
        duration_cast<milliseconds>(t_now - t_stamps[stamp_name]);
    return dur.count();
  } else if (unit == "microsecond" || unit == "us") {
    auto dur =
        duration_cast<microseconds>(t_now - t_stamps[stamp_name]);
    return dur.count();
  } else if (unit == "nanosecond" || unit == "ns") {
    auto dur = duration_cast<nanoseconds>(t_now - t_stamps[stamp_name]);
    return dur.count();
  } else {
    auto dur = duration_cast<duration<float, std::ratio<1, 1>>>(
        t_now - t_stamps[stamp_name]);
    return dur.count();
  }
}
