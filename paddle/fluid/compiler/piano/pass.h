
#pragma once
#include <functional>
#include <memory>
#include <string>

namespace paddle {
namespace piano {

using note::Function;
using note::Module;

template<typename T>
using StringMap = std::unordered_map<std::string, T>;
using String = std::string;

enum PassKind : char {
  // Function pass
  PASSKIND_FUN,
  // Module pass
  PASSKIND_MOD,
  // Virtual pass
  PASSKIND_VIR
};

template<typename ir_type>
class Pass {
  PassKind kind_;
 public:
  Pass(CompilerContext *cc);
  virtual ~Pass();
  virtual PassKind kind() = 0;
  virtual String&& name() = 0;
  virtual bool analyze(CompilerContext *cc, std::shared_ptr<ir_type>&& ir) = 0;
  virtual bool transform(CompilerContext *cc, std::unique_ptr<ir_type>&& ir) = 0;
  virtual bool run(CompilerContext *cc) = 0;
};

}
}
