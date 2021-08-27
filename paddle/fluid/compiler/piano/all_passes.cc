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

#include "paddle/fluid/compiler/piano/pass.h"
#include "paddle/fluid/compiler/piano/all_passes.h"
#include "glog/logging.h"

namespace paddle {
namespace piano {

int verify_all_passes() {
  int count = 0;
#define VAR(pass) _##pass
#define CHECK_PASS(pass)                              \
  auto VAR(pass) = PASS_CTOR(pass);                   \
  count++;                                            \
  LOG(INFO) << "Check pass: " << VAR(pass).name();    \
  {
    // Expand the pass list
    PASS_ALL(CHECK_PASS)
  }
  return count;
#undef CHECK_PASS
#undef VAR
}

}
}