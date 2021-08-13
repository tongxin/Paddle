cc_library(primitive_ir_emitter SRCS primitive_ir_emitter.cc DEPS llvm)
cc_library(gpu_primitive_ir_emitter SRCS gpu_primitive_ir_emitter.cc DEPS llvm)
cc_library(nvptx_primitive_ir_emitter SRCS nvptx_primitive_ir_emitter.cc DEPS llvm)

cc_test(nvptx_primitive_ir_emitter_test SRCS nvptx_primitive_ir_emitter_test.cc
        DEPS nvptx_primitive_ir_emitter gpu_primitive_ir_emitter primitive_ir_emitter)
target_link_libraries(nvptx_primitive_ir_emitter_test ${LLVM_LIBS})
