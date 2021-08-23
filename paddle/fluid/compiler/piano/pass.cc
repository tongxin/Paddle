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

namespace paddle {
namespace piano {

class ExpandBatchNormPass : Pass {
 public:
  ExpandBatchNormPass(CompilerContext *cc) : Pass(cc) {}
  ~ExpandBatchNormPass() override = default; 
  bool run(const void *ir) override {
    bool changed = false;

    return changed;
  }
  std::string name() const override {
    return "expand_batchnorm_pass";
  }
};

void verify_all_passes() {
#define VAR(pass) _##pass
#define DECLARE(pass)                   \
  PASSDEF_CLASSNAME(pass) VAR(pass)();
  // Expand the pass list
  PASSDEF_ALL(DECLARE)

#undef DECLARE
#undef VAR
}

}
}
