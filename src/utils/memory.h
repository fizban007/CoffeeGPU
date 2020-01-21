#ifndef _UTILS_MEMORY_H_
#define _UTILS_MEMORY_H_

#include <cstddef>

namespace Coffee {

/// Malloc an aligned memory region of size size, with specified alignment. It
/// is required that alignment is smaller than 0x8000. Note: This function does
/// not initialize the new allocated memory. Need to call initialize by hand
/// afterwards
void* aligned_malloc(std::size_t size, std::size_t alignment);
void aligned_free(void* p);

}  // namespace Coffee

#endif  // _UTILS_MEMORY_H_
