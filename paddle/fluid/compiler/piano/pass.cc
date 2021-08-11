#include "paddle/fluid/compiler/note/pass.h"


namespace paddle {
namespace piano {

#define PASS_CLASS(name) name
#define SETID(name)           \
  PassId PASS_CLASS(name)::PassId() { return PASSID(name); }

PIANO_PASS_LIST(SETID)

#undef SETID
#undef PASS_CLASS

}
}
