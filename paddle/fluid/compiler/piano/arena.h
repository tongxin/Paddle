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

#ifndef PADDLE_COMPILER_ARENA_H
#define PADDLE_COMPILER_ARENA_H
// #include "paddle/fluid/platform/enforce.h"
namespace paddle {
namespace piano {

typedef decltype(sizeof(void *)) size_t;
const size_t ARENA_DEFAULT_SIZE = 1 << 20;
const size_t ARENA_MAXIMUM_SIZE = 1 << 30;

static size_t roundup(size_t bytes, size_t alignment) {
  // PADDLE_ENFORCE_LE(alignment, ARENA_MAXIMUM_SIZE,
  //                   "Alignment is larger than maximum arena size.");
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
  explicit Arena(size_t bytes = ARENA_DEFAULT_SIZE);
  ~Arena();
  char* allocate_aligned(const size_t bytes, size_t alignment);

  inline char* allocate(size_t numel, size_t sz) {
    return allocate_aligned(numel * sz, sz);
  }

 private:
  char *start_;
  char *end_;
  char *pos_;
};

}
}

#endif // PADDLE_COMPILER_ARENA_H