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
  ATestPass() : Pass() {}
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

// Put all the piano optimization passes here so that they can be hooked
// with the make_pass function.
#define PASS_ALL(__macro)        \
  __macro(ATest)

// Pass id enum is used as key for dispatching pass classes
enum class PassId {
#define ID(pass) pass,
  PASS_NA,
  PASS_ALL(ID)
#undef ID
};

#define INC(name) +1
  constexpr int Total_Num_Passes = PASS_ALL(INC);
#undef INC

// Following are basic utilities for constructing pass objects

template<typename P>
static P *do_make_pass() { 
  static_assert(std::is_base_of<Pass, P>::value);
  return new P();
}

template<PassId T>
struct PassClass {};

#define PASS_ID(pass)        PassId::pass
#define PASS_CLASS(pass)     pass##Pass

#define SPECIALIZE_PASSCLASS(pass)                  \
template<>                                          \
struct PassClass<PASS_ID(pass)> {                   \
  using type = PASS_CLASS(pass);                    \
};
PASS_ALL(SPECIALIZE_PASSCLASS)

// Use this macro as the public interface for constructing heap allocated
// pass object. 
// Code example:
// {
//    auto* dce_pass = make_pass(ModuleDCE);
//    dce->run(module_ir);
// }
#define make_pass(pass)                             \
  do_make_pass<PassClass<PASS_ID(pass)>::type>();

// use this macro as the public interface for constructing stack allocated
// pass object.
// Code example:
// {
//    auto dce_pass = PASS_CTOR(ModuleDCE);
//    dce.run(module_ir);
// } 
#define PASS_CTOR(pass)                            \
  PassClass<PASS_ID(pass)>::type();

int verify_all_passes();

#undef DEF_PASS_MAKER
#undef SPECIALIZE_PASSCLASS

}
}
