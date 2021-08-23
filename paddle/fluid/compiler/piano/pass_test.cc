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

#include "paddle/fluid/compiler/note/pass.h"
#include "paddle/fluid/compiler/note/instruction.h"
#include "paddle/fluid/compiler/note/function.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddle {
namespace piano {

using Function = note::Function;
using OpCode = note::OpCode;

#undef PASSDEF_ALL
#define PASSDEF_ALL(__macro)      \
  __macro(ATest)


void verify_all_passes() {
#define VAR(pass) _##pass
#define DECLARE(pass)                   \
  PASSDEF_CLASSNAME(pass) VAR(pass)();
  // Expand the pass list
  PASSDEF_ALL(DECLARE)

#undef DECLARE
#undef VAR
}

class ATestPass : Pass {
 public:
  ATestPass(CompilerContext *cc) : Pass(cc) {}
  ~ATestPass() override = default; 
  bool run(const void *ir) override {
    bool changed = false;
    auto* ir = static_cast<Function*>(ir);
    auto dead_ins = std::vector<Instruction*>();
    for (const auto *instruction : ir->instructions()) {
      if (instruction->ctrl_predecessors().empty() &&
          instruction->ctrl_successors().empty()   &&
          instruction->opcode() != OpCode.kParameter)
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

class BTestPass : Pass {
 public:
  BTestPass(CompilerContext *cc) : Pass(cc) {}
  ~BTestPass() override = default; 
  bool run(const void *ir) override {
    bool changed = false;
    auto* ir = static_cast<Function*>(ir);
    auto dead_ins = std::vector<Instruction*>();
    for (const auto *instruction : ir->instructions()) {
      if (instruction->ctrl_predecessors().empty() &&
          instruction->ctrl_successors().empty()   &&
          instruction->opcode() != OpCode.kParameter)
        dead_ins.push_back(instruction);
    }
    // (TODO) remove dead instructions from function
    changed = !dead_ins.empty();
    return changed;
  }
  std::string name() const override {
    return "b_test_pass";
  }
};

class PassClassTest : public ::testing::Test {
  virtual void SetUp() {
    // input shapes
    const Shape arg1_shape(note::F32, {3, 6});
    const Shape arg2_shape(note::F32, {3, 6});
    // output shape
    const Shape result_shape(note::F32, {3, 6});
    // function signature
    const Signature signature({arg1_shape, arg2_shape}, {"arg1.1", "arg2.2"},
                              result_shape);
    SignatureProto signature_proto = signature.ToProto();
    signature_proto_.Swap(&signature_proto);

    // set instr1_proto_
    instr1_proto_.set_name("arg1.1");
    instr1_proto_.set_opcode(GetOpName(OpCode::kParameter));
    instr1_proto_.set_id(1);
    instr1_proto_.set_parameter_number(0);
    *instr1_proto_.mutable_shape() = arg1_shape.ToProto();
    auto* attrs1_map = instr1_proto_.mutable_attrs();
    AttrValueProto val1_proto;
    val1_proto.set_d(3.141);
    attrs1_map->insert(ProtoMapType::value_type("test_double", val1_proto));
    auto* strings = val1_proto.mutable_strings()->mutable_value();
    *strings->Add() = "hello";
    *strings->Add() = "world";
    attrs1_map->insert(ProtoMapType::value_type("test_strings", val1_proto));
    auto* bools = val1_proto.mutable_bools()->mutable_value();
    bools->Add(true);
    bools->Add(false);
    attrs1_map->insert(ProtoMapType::value_type("test_bools", val1_proto));
    auto* ints = val1_proto.mutable_ints()->mutable_value();
    ints->Add(8);
    ints->Add(26);
    attrs1_map->insert(ProtoMapType::value_type("test_ints", val1_proto));

    // set instr2_proto_
    instr2_proto_.set_name("arg2.2");
    instr2_proto_.set_opcode(GetOpName(OpCode::kParameter));
    instr2_proto_.set_id(2);
    instr2_proto_.set_parameter_number(0);
    *instr2_proto_.mutable_shape() = arg2_shape.ToProto();
    auto* attrs2_map = instr2_proto_.mutable_attrs();
    AttrValueProto val2_proto;
    val2_proto.set_b(true);
    attrs2_map->insert(ProtoMapType::value_type("test_bool", val2_proto));
    auto* longs = val2_proto.mutable_longs()->mutable_value();
    longs->Add(8l);
    longs->Add(16l);
    attrs2_map->insert(ProtoMapType::value_type("test_longs", val2_proto));
    auto* floats = val2_proto.mutable_floats()->mutable_value();
    floats->Add(8.6f);
    floats->Add(7.6f);
    attrs2_map->insert(ProtoMapType::value_type("test_floats", val2_proto));
    auto* doubles = val2_proto.mutable_doubles()->mutable_value();
    doubles->Add(5.66);
    doubles->Add(6.66);
    attrs2_map->insert(ProtoMapType::value_type("test_doubles", val2_proto));

    // set instr3_proto_
    instr3_proto_.set_name("add.3");
    instr3_proto_.set_opcode(GetOpName(OpCode::kAdd));
    instr3_proto_.set_id(3);
    instr3_proto_.set_parameter_number(2);
    *instr3_proto_.mutable_shape() = result_shape.ToProto();
    instr3_proto_.add_operand_ids(1);
    instr3_proto_.add_operand_ids(2);
    auto* attrs3_map = instr3_proto_.mutable_attrs();
    AttrValueProto val3_proto;
    val3_proto.set_s("Add");
    attrs3_map->insert(ProtoMapType::value_type("test_string", val3_proto));
    val3_proto.set_i(-1);
    attrs3_map->insert(ProtoMapType::value_type("test_int", val3_proto));
    val3_proto.set_l(-100l);
    attrs3_map->insert(ProtoMapType::value_type("test_long", val3_proto));
    val3_proto.set_f(-1.414f);
    attrs3_map->insert(ProtoMapType::value_type("test_float", val3_proto));

    // set func_proto_
    func_proto_.set_name(func_name_);
    *func_proto_.mutable_signature() = signature_proto_;
    func_proto_.set_return_id(instr3_proto_.id());
    function_id_ = instr3_proto_.id() + 1;
    func_proto_.set_id(function_id_);
    *func_proto_.add_instructions() = instr1_proto_;
    *func_proto_.add_instructions() = instr2_proto_;
    *func_proto_.add_instructions() = instr3_proto_;
  }
 protected:
  std::string func_name_{"union_12510013719728903619"};
  FunctionProto func_proto_;
  std::int64_t function_id_;
  SignatureProto signature_proto_;
  InstructionProto instr1_proto_;
  InstructionProto instr2_proto_;
  InstructionProto instr3_proto_;
};

TEST_F(PassClassTest, VerifyPasses) {
  verify_all_passes();
  // Control reaches here if compilation succeeds 
  EXPECT_TRUE(true);
}

TEST_F(PassClassTest, SimpleFunctionPass) {
  std::unordered_map<std::int64_t, Function*> func_index;
  Function func(func_proto_, func_index);
  auto* a_pass = make_pass(ATest, nullptr);
  EXPECT_EQ(a_pass(&func, nullptr), false);
  LOG(INFO) << "A simple function pass detecting dead instructions.";
}

TEST_F(PassClassTest, FailUnregisteredPass) {
  auto* b_pass = make_pass(BTest, nullptr);
  EXPECT_THROW();
}

}
}