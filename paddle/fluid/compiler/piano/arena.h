#ifndef PADDLE_COMPILER_ARENA_H
#define PADDLE_COMPILER_ARENA_H
#include "paddle/fluid/platform/enforce.h"
namespace paddle {
namespace piano {

typedef decltype(sizeof(1)) size_t;
const size_t ARENA_DEFAULT_SIZE = 1 << 20;
const size_t ARENA_MAXIMUM_SIZE = 1 << 30;

static size_t roundup(size_t bytes, size_t alignment) {
  PADDLE_ENFORCE_LE(alignment, ARENA_MAXIMUM_SIZE,
                    "Alignment is larger than maximum arena size.");
  auto x = alignment - 1;
  return (bytes + x) & ~x;
}

class Arena
{
 private:
  // Constants
  // Permits no copy construction
  Arena(const Arena&);
  void operator=(const Arena&);
  bool expand(size_t new_size);

 public:
  explicit Arena(size_t arena_size = ARENA_DEFAULT_SIZE);
  ~Arena();
  inline char* allocate_aligned(const size_t bytes, size_t alignment);

  template<typename T>
  inline char* allocate(size_t numel) {
    return allocate_aligned(numel * sizeof(T), sizeof(T));
  }

 private:
  char *start_;
  char *end_;
  char *pos_;
};

}
}

#endif // PADDLE_COMPILER_ARENA_H