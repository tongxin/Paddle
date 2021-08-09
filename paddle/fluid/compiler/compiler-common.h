#ifndef PADDLE_COMPILER_COMPILER_COMMON_H
#define PADDLE_COMPILER_COMPILER_COMMON_H

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {

class StaticAllocated {
 private:
  // Disallow allocation on the heap
  StaticAllocated() = delete;
  StaticAllocated(const StaticAllocated&) = delete;
  void operator= (const StaticAllocated&) = delete;
  void *operator new(size_t size);
 public:
  void operator delete(void *obj) {
    PADDLE_THROW("StaticAllocated objects are disallowed on the heap "
                 "and hence any deallocation is not sensible.");
  }
};

class Arena;

// Objects in arena are not deallocated individually 
class ArenaAllocated
{
private:
  // Arena allocated objects requires explicit construction
  ArenaAllocated(const ArenaAllocated&) = delete;
  void operator= (const ArenaAllocated&) = delete;
public:
  ArenaAllocated();
  
  void *operator new(const size_t size);
  void *operator new(const size_t size, const Arena *arena);
  
  void operator delete(void *obj) {
    PADDLE_THROW("Arena allocated object should not be individually deallocated.");
  }
};

  
} // namespace piano
} // namespace paddle



#endif // PADDLE_COMPILER_COMPILER_COMMON_H