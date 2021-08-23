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

#include "paddle/fluid/compiler/piano/arena.h"
#include <stdlib.h>

namespace paddle {
namespace piano {

Arena::Arena(size_t bytes = ARENA_DEFAULT_SIZE) {
  size = roundup(bytes, ARENA_DEFAULT_SIZE);
  PADDLE_ENFORCE_LE(arena_size, ARENA_MAXIMUM_SIZE,
                    "Rounded Arena size %s (bytes) exceeds maximally allowed.", size);
  char* p = aligned_alloc(ARENA_DEFAULT_SIZE, size);
  if (p == nullptr)
    PADDLE_THROW("Failed to allocate Arena of size %s", size);
  
  start_ = p;
  pos_ = p;
  end_ = p + size;
}

inline char* Arena::allocate_aligned(const size_t bytes, size_t alignment) {
  auto size = roundup(bytes, alignment);
  
  auto new_pos = pos_ + size;

  if (new_pos > end_)
    expand(new_pos - start_);
  
  pos_ += size;
  
  return pos_;
}

}
}