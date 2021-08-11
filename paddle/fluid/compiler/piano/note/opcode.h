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

#include <cstdint>
#include <string>

namespace paddle {
namespace piano {
namespace note {

// HANDLE(ID, DESC, PARAM_NUM, ...)
#define OPCODE_HANDLER(HANDLE)                           \
  HANDLE(kConstant, "constant", 0)                       \
  HANDLE(kBatchNormGrad, "batch_norm_grad", 5)           \
  HANDLE(kBatchNormInference, "batch_norm_inference", 5) \
  HANDLE(kBatchNormTraining, "batch_norm_training", 3)   \
  HANDLE(kConvolution, "convolution", 2)                 \
  HANDLE(kDot, "dot", 2)                                 \
  HANDLE(kBroadcast, "broadcast", 1)                     \
  HANDLE(kCast, "cast", 1)                               \
  HANDLE(kCopy, "copy", 1)                               \
  HANDLE(kExp, "exp", 1)                                 \
  HANDLE(kLog, "log", 1)                                 \
  HANDLE(kNegative, "negative", 1)                       \
  HANDLE(kNot, "not", 1)                                 \
  HANDLE(kReshape, "reshape", 1)                         \
  HANDLE(kReverse, "reverse", 1)                         \
  HANDLE(kRsqrt, "rsqrt", 1)                             \
  HANDLE(kSlice, "slice", 1)                             \
  HANDLE(kSqrt, "sqrt", 1)                               \
  HANDLE(kTranspose, "transpose", 1)                     \
  HANDLE(kAdd, "add", 2)                                 \
  HANDLE(kAnd, "and", 2)                                 \
  HANDLE(kCompare, "compare", 2)                         \
  HANDLE(kDivide, "divide", 2)                           \
  HANDLE(kMaximum, "maximum", 2)                         \
  HANDLE(kMinimum, "minimum", 2)                         \
  HANDLE(kMultiply, "multiply", 2)                       \
  HANDLE(kOr, "or", 2)                                   \
  HANDLE(kSubtract, "subtract", 2)                       \
  HANDLE(kXor, "xor", 2)                                 \
  HANDLE(kSelect, "select", 3)                           \
  HANDLE(kConcatenate, "concatenate", kVariadicParamNum) \
  HANDLE(kReduce, "reduce", kVariadicParamNum)           \
  HANDLE(kRng, "rng", kVariadicParamNum)                 \
  HANDLE(kSort, "sort", kVariadicParamNum)               \
  HANDLE(kTuple, "tuple", kVariadicParamNum)

constexpr std::int32_t kVariadicParamNum = -1;

enum class OpCode : std::uint32_t {
#define HANDLE_ENUM(enum_id, ...) enum_id,
  OPCODE_HANDLER(HANDLE_ENUM)
#undef HANDLE_ENUM
      kNumOps
};

OpCode GetOpCode(const std::string& op_name);

const std::string& GetOpName(OpCode code);

int GetOpParamNum(OpCode code);

}  // namespace note
}  // namespace piano
}  // namespace paddle
