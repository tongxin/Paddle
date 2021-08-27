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

#include "paddle/fluid/compiler/piano/pass.h"
#include "paddle/fluid/compiler/piano/arena.h"
#include "paddle/fluid/compiler/piano/compiler-thread.h"
#include "paddle/fluid/compiler/piano/note/instruction.h"
#include "paddle/fluid/compiler/piano/note/function.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include <type_traits>

// include all pass class definition headers here


namespace paddle {
namespace piano {

class ATestPass : public Pass {
  using Function = note::Function;
  using Instruction = note::Instruction;
  using OpCode = note::OpCode;
 public:
  ATestPass(CompilerContext *cc) : Pass(cc) {}
  ~ATestPass() override = default; 
  bool run(void *fn) override {
    bool changed = false;
    auto* ir = static_cast<Function*>(fn);
    auto dead_ins = std::vector<Instruction*>(); 
    for (auto& instruction : ir->instructions()) {
      if (instruction.ctrl_predecessors().empty() &&
          instruction.ctrl_successors().empty()   &&
          instruction.opcode() != OpCode::kParameter)
        dead_ins.push_back(&instruction);
    }
    // (TODO) remove dead instructions from function
    changed = !dead_ins.empty();
    return changed;
  }
  std::string name() const override {
    return "a_test_pass";
  }
};

class ExpandBatchNormPass : Pass {
  using Function = note::Function;
  using Instruction = note::Instruction;
  using OpCode = note::OpCode;
 public:
  ExpandBatchNormPass(CompilerContext *cc) : Pass(cc) {}
  ~ExpandBatchNormPass() override = default; 
  bool run(void *ir) override {
    bool changed = false;

    return changed;
  }
  std::string name() const override {
    return "expand_batchnorm_pass";
  }
};

// All PIANO optimization passes must be present here.
// Use make_pass(pass_id, CompilerContext*) to create a concrete pass object. 
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

#define DECL(pass)      \
class PASSDEF_CLASSNAME(pass);
PASSDEF_ALL(DECL)
#undef DECL


// Following are basic utilities for constructing pass objects

template<typename P>
static P *do_make_pass(CompilerContext *cc) {  
  static_assert(std::is_base_of<Pass, P>::value);
  if (!cc)
    return new P(cc);
  char *p = (*cc->arena()).allocate(1, sizeof(P));
  return new(p) P(cc);
}

template<PassId T>
struct PassClass {};


#define SPECIALIZE_PASSCLASS(pass)                  \
template<>                                          \
struct PassClass<PassId::PASSDEF_ID(pass)> {        \
  using type = PASSDEF_CLASSNAME(pass);             \
};
PASSDEF_ALL(SPECIALIZE_PASSCLASS)

#define make_pass(pass, cc)                         \
  do_make_pass<PassClass<PassId::PASSDEF_ID(pass)>::type>(cc);

void verify_all_passes();

#undef PASSDEF_ID_
#undef DEF_PASS_MAKER

}
}
