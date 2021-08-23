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

#pragma once
#include "paddle/fluid/platform/enforce.h"
#include <functional>
#include <memory>
#include <string>

namespace paddle {
namespace piano {

// All PIANO optimization passes must be present here.
// Use make_pass<SpecificPassType>(CompilerContext*) to create
// a concrete pass object. 
#define PASSDEF_ALL(__macro)    \
  __macro(ExpandBatchNorm)      \
  __macro(TransposeFolding)

#define PASSDEF_ID        (pass) PASS_##pass
#define PASSDEF_ID_       (pass) PASS_##pass,
#define PASSDEF_CLASSNAME (pass) PASS##Pass

// Pass id enum is used as key for dispatching pass classes
enum class PassId {
  PASS_NA,
  PASSDEF_ALL(PASSDEF_ID_)
};

#define INC(name) +1
  constexpr int Total_Num_Passes = PASSDEF_ALL(INC);
#undef INC

class CompilerContext;

class Pass {
  CompilerContext *cc_;
 public:
  Pass() { cc_ = nullptr;}
  Pass(CompilerContext *cc) : cc_(cc) {}
  virtual ~Pass();
  virtual bool run(void *ir) = 0;
  virtual std::string name() const = 0;
};

template<typename PassT>
PassT *do_make_pass(CompilerContext *cc) {
  char *p = cc.arena().allocate(sizeof(PassT));
  return new(p) PassT(cc);
}

#define make_pass(pass, cc)     \
  {
    static_assert(PassId::PASSDEF_ID(pass) > PassId::PASS_NA);
    do_make_pass<PASSDEF_CLASSNAME(pass)>(cc);
  }

#undef PASSDEF_ID_

}
}
