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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_primitive_ir_emitter.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddle {
namespace piano {
namespace backends {

TEST(NvptxPrimitiveIrEmitter, GetBinaryOp) {
  NvptxPrimitiveIrEmitter nvptx_primitive_ir_emitter;
  auto op = nvptx_primitive_ir_emitter.GetUnaryOp(nullptr);
}

TEST(NvptxPrimitiveIrEmitter, GetUnaryOp) {
  NvptxPrimitiveIrEmitter nvptx_primitive_ir_emitter;
  auto op = nvptx_primitive_ir_emitter.GetBinaryOp(nullptr);
}

TEST(NvptxPrimitiveIrEmitter, DeviceBaseOp) {
  NvptxPrimitiveIrEmitter nvptx_primitive_ir_emitter;
  ASSERT_EQ(nvptx_primitive_ir_emitter.ThreadIdx(nullptr), nullptr);
  ASSERT_EQ(nvptx_primitive_ir_emitter.ThreadIdy(nullptr), nullptr);
  ASSERT_EQ(nvptx_primitive_ir_emitter.ThreadIdz(nullptr), nullptr);
  ASSERT_EQ(nvptx_primitive_ir_emitter.BlockDimx(nullptr), nullptr);
  ASSERT_EQ(nvptx_primitive_ir_emitter.BlockDimy(nullptr), nullptr);
  ASSERT_EQ(nvptx_primitive_ir_emitter.BlockDimz(nullptr), nullptr);
  ASSERT_EQ(nvptx_primitive_ir_emitter.BlockIdx(nullptr), nullptr);
  ASSERT_EQ(nvptx_primitive_ir_emitter.BlockIdy(nullptr), nullptr);
  ASSERT_EQ(nvptx_primitive_ir_emitter.BlockIdz(nullptr), nullptr);
  ASSERT_EQ(nvptx_primitive_ir_emitter.Alloca(nullptr), nullptr);
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
