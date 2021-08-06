#include "paddle/fluid/compiler/note/module_dce.h"

namespace paddle {
namespace piano {

bool ModuleDCE::run(CompilerContext *cc) {
  auto ir = cc->GetModule();
  liveness.analyze(cc, ir);
  analysis1.analyze(cc, ir);
  return dce.transform(cc, ir);
}

} // namespace piano
} // namespace paddle
