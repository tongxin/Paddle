#ifndef PADDLE_COMPILER_COMPILER_H
#define PADDLE_COMPILER_COMPILER_H

#include "paddle/fluid/compiler/compiler-common.h"
#include "paddle/fluid/compiler/piano/note/module.h"

namespace paddle {
namespace piano {

class Compiler : StaticAllocated
{
 public:
  bool CompileModule(const Module& module_ir);
};


} // namespace piano
} // namespace paddle


#endif // PADDLE_COMPILER_COMPILER_H