#pragma once
#include "paddle/fluid/compiler/note/pass.h"

namespace paddle {
namespace piano {

using note::Module;

// This pass removes dead code in the module
class ModuleDCE : Pass<Module> {
  std::unique_ptr<LivenessAnalysis> liveness;
  std::unique_ptr<SomeOtherAnalysis> analysis1;
  std::unique_ptr<DCE> dce;
 public:
  ModuleDCE(CompilerContext *cc) {
    liveness = std::make_unique<LivenessAnalysis>(cc);
    analysis1 = std::make_unique<SomeOtherAnalysis>(cc);
  }
  ~ModuleDCE() override {}
  PassKind kind() override { return PASSKIND_MOD;      }
  String&& name() override { return "module_dce_pass"; }
  bool run(CompilerContext *cc);
};
 
} // namespace piano
} // namespace paddle