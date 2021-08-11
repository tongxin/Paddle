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