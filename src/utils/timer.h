#ifndef _TIMER_H_
#define _TIMER_H_

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

namespace Aperture {

class timer {
 public:
  timer() {}
  ~timer() {}

  static void stamp(const std::string& name = "");
  static void show_duration_since_stamp(
      const std::string& routine_name, const std::string& unit,
      const std::string& stamp_name = "");
  static float get_duration_since_stamp(
      const std::string& unit, const std::string& stamp_name = "");

  static std::unordered_map<
      std::string, std::chrono::high_resolution_clock::time_point>
      t_stamps;
  static std::chrono::high_resolution_clock::time_point t_now;
};  // ----- end of class timer -----

}  // namespace Aperture

#endif  // _TIMER_H_
