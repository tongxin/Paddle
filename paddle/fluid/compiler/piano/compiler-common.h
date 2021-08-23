/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

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