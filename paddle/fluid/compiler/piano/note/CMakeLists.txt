cc_library(note_opcode SRCS opcode.cc DEPS errors enforce)
cc_test(note_opcode_test SRCS opcode_test.cc DEPS note_opcode)

proto_library(note_proto SRCS note.proto)
target_compile_options(note_proto PUBLIC "-Wno-extra")
