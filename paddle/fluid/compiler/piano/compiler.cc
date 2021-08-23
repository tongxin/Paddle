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

#include "paddle/fluid/compiler/compiler.h"
#include "paddle/fluid/compiler/compiler-common.h"
#include "paddle/fluid/compiler/compiler-thread.h"

namespace paddle {
namespace piano {

class CompilerTask : public ThreadPool::Task {
 public:
  explicit CompilerTask(Compiler* compiler)
      : compiler_(compiler) {}
  virtual ~CompilerTask() {}

 private:
  virtual void Run() { ; }

  Compiler* compiler_;

  CompilerTask(const CompilerTask &) = delete;
  void operator=(const CompilerTask& ) = delete;
};



bool Compiler::CompileModule(const Module& ir) {
  
}

} // namespace piano
} // namespace paddle