#include "utils/memory.h"
#include <assert.h>
#include <cstdint>
#include <cstdlib>

namespace Coffee {

void *aligned_malloc(std::size_t size, std::size_t alignment) {
  // assert(alignment <= 0x8000);
  uintptr_t r = (uintptr_t)malloc(size + --alignment + 2);
  uintptr_t o = (r + 2 + alignment) & ~(uintptr_t)alignment;
  if (!r)
    return nullptr;
  ((uint16_t *)o)[-1] = (uint16_t)(o - r);
  return (void *)o;
}

void aligned_free(void *p) {
  if (p == nullptr || !p)
    return;
  free((void *)((uintptr_t)p - ((uint16_t *)p)[-1]));
}

} // namespace Coffee
