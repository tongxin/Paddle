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
#include "paddle/fluid/compiler/piano/compiler-thread.h"
#include "paddle/fluid/compiler/piano/arena.h"
#include "paddle/fluid/compiler/piano/note/instruction.h"
#include "paddle/fluid/compiler/piano/note/function.h"
#include "paddle/fluid/platform/enforce.h"
#include <functional>
#include <memory>
#include <string>

namespace paddle {
namespace piano {

using Function = note::Function;
using Instruction = note::Instruction;
using OpCode = note::OpCode;

// All PIANO optimization passes must be present here.
// Use make_pass<SpecificPassType>(CompilerContext*) to create
// a concrete pass object. 
#define PASSDEF_ALL(__macro)    \
  __macro(ExpandBatchNorm)      \
  __macro(ATest)


#define PASSDEF_ID(pass)        PASS_##pass
#define PASSDEF_ID_(pass)       PASS_##pass,
#define PASSDEF_CLASSNAME(pass) pass##Pass

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
 private:
  CompilerContext *cc_;
 public:
  Pass() { cc_ = nullptr;}
  Pass(CompilerContext *cc) : cc_(cc) {}
  virtual ~Pass() {};
  virtual bool run(void *ir) = 0;
  virtual std::string name() const = 0;
};

template<PassId pass_id, typename PassT>
PassT *do_make_pass(CompilerContext *cc) {
  if (!cc)
    return new PassT(cc);
  char *p = (*cc->arena()).allocate(1, sizeof(PassT));
  return new(p) PassT(cc);
}

#define make_pass(pass, cc)                                   \
  do_make_pass<PassId::PASSDEF_ID(pass), PASSDEF_CLASSNAME(pass)>(cc);

#define DECL(pass)      \
class PASSDEF_CLASSNAME(pass);
PASSDEF_ALL(DECL)
#undef DECL

void verify_all_passes();

class ATestPass : Pass {
 public:
  ATestPass(CompilerContext *cc) : Pass(cc) {}
  ~ATestPass() override = default; 
  bool run(void *fn) override {
    bool changed = false;
    auto* ir = static_cast<Function*>(fn);
    auto dead_ins = std::vector<Instruction*>();
    for (auto *instruction : ir->instructions()) {
      if (instruction->ctrl_predecessors().empty() &&
          instruction->ctrl_successors().empty()   &&
          instruction->opcode() != OpCode::kParameter)
        dead_ins.push_back(instruction);
    }
    // (TODO) remove dead instructions from function
    changed = !dead_ins.empty();
    return changed;
  }
  std::string name() const override {
    return "a_test_pass";
  }
};


#undef PASSDEF_ID_

}
}
