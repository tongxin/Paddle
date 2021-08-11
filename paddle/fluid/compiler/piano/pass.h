
#pragma once
#include <functional>
#include <memory>
#include <string>

namespace paddle {
namespace piano {

#define PIANO_PASS_LIST(__macro)              \
  __macro(InliningPass)

template<typename T>
using StringMap = std::unordered_map<std::string, T>;
using String = std::string;

#define PASSID(name) PASS__##name

enum PassId {
  #define ID(name) PASS_ID(name),
    PIANO_PASS_LIST(ID)
  #undef ID
};

#define INC(name) +1
  constexpr int PIANO_NUM_PASSES = PIANO_PASS_LIST(INC);
#undef INC

class Pass {
 public:
  Pass(CompilerContext *cc);
  virtual ~Pass();
  virtual bool run(CompilerContext *cc) = 0;
  virtual PassId PassId() = 0;
};

#define DECL(name)              \
class name : Pass {};

PIANO_PASS_LIST(DECL)

#undef DECL

}
}
