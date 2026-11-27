# DeepSeek V4.1 Flash external model extension.

# Host side: artifact probe, chat template, tool syntax, adapter files and
# registration.
gufo_extension_target(
  NAME gufo_deepseek41flash
  SOURCES
    chat_template.cpp
    direction_adapter.cpp
    extension.cpp
    lora_adapter.cpp
    probe.cpp
    text_contract.cpp
    tool_calls.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
)

if(TARGET gufo_deepseek41flash)
  target_include_directories(gufo_deepseek41flash PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  # Registration leaves the provider to gufo_deepseek41flash_text in HIP builds.
  if(ENGINE_ENABLE_HIP)
    set_property(SOURCE "${GUFO_EXTENSION_BUILD_ROOT}/extension.cpp"
      APPEND PROPERTY COMPILE_DEFINITIONS ENGINE_ENABLE_HIP=1)
  endif()
endif()

if(ENGINE_ENABLE_HIP)
  set(DEEPSEEK41FLASH_RUNTIME_DIR "${GUFO_EXTENSION_BUILD_ROOT}/runtime")
  set(DEEPSEEK41FLASH_ROCM_DIR "${GUFO_EXTENSION_BUILD_ROOT}/kernels/rocm")
  set(DEEPSEEK41FLASH_MMQ_DIR "${DEEPSEEK41FLASH_ROCM_DIR}/mmq")

  # Vendored llama.cpp quantized-matmul tier: its own C++17 dialect, GGML
  # feature macros and tile size, so it stays a separate target. See
  # kernels/rocm/mmq/VENDOR.md.
  set(DEEPSEEK41FLASH_MMQ_SOURCES
    ${DEEPSEEK41FLASH_MMQ_DIR}/ds41_ggml_stubs.hip.cpp
    ${DEEPSEEK41FLASH_MMQ_DIR}/ds41_mmq.hip.cpp
    ${DEEPSEEK41FLASH_MMQ_DIR}/ds41_mmq_d2r.hip.cpp
    ${DEEPSEEK41FLASH_MMQ_DIR}/mmid.hip.cpp
    ${DEEPSEEK41FLASH_MMQ_DIR}/mmvq.hip.cpp
    ${DEEPSEEK41FLASH_MMQ_DIR}/quantize.hip.cpp)
  add_library(gufo_deepseek41flash_mmq STATIC ${DEEPSEEK41FLASH_MMQ_SOURCES})
  # One object per V4.1 weight type outside the V4 set, from one source.
  foreach(ds41_mmq_type Q5_K Q6_K IQ1_S IQ1_M IQ2_XS IQ2_S)
    set(ds41_mmq_type_source
      "${CMAKE_CURRENT_BINARY_DIR}/ds41_mmq_v41_${ds41_mmq_type}.hip.cpp")
    file(GENERATE OUTPUT ${ds41_mmq_type_source} CONTENT
      "#define DS41_MMQ_TYPE GGML_TYPE_${ds41_mmq_type}\n#include \"ds41_mmq_v41_types.hip.cpp\"\n")
    target_sources(gufo_deepseek41flash_mmq PRIVATE ${ds41_mmq_type_source})
    list(APPEND DEEPSEEK41FLASH_MMQ_SOURCES ${ds41_mmq_type_source})
  endforeach()
  set_source_files_properties(${DEEPSEEK41FLASH_MMQ_SOURCES}
    PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_deepseek41flash_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_deepseek41flash_mmq PRIVATE
    GGML_USE_HIP
    GGML_HIP_NO_VMM
    DS41_ROCM_WMMA_MMQ_Y=64)
  # ${GUFO_CLANG_OPTION} is /clang: under clang-cl, which otherwise drops
  # these flags (or keeps them from the device compile).
  target_compile_options(gufo_deepseek41flash_mmq PRIVATE
    ${GUFO_CLANG_OPTION}-O3 ${GUFO_CLANG_OPTION}-ffast-math
    ${GUFO_CLANG_OPTION}-fno-finite-math-only -Wno-unused-value
    -Wno-unused-command-line-argument)
  target_include_directories(gufo_deepseek41flash_mmq
    SYSTEM PUBLIC ${DEEPSEEK41FLASH_MMQ_DIR})
  target_link_libraries(gufo_deepseek41flash_mmq PUBLIC hip::host roc::hipblas)
endif()

# Device side: engine, decoder, DSpark, expert store, kernels, vision encoder
# and continuation adapter. The whole package keeps -ffast-math: moving code
# between these flags changes rounding, and outputs must stay bit-identical.
gufo_extension_target(
  NAME gufo_deepseek41flash_hip
  SOURCES
    adapter.cpp
    continuation_adapter.cpp
    engine.cpp
    runtime/ds41_platform.cpp
    runtime/expert_store.cpp
    runtime/model_data.cpp
    runtime/sampling.cpp
    runtime/tokenizer.cpp
    runtime/v41_decode.cpp
    runtime/v41_dspark.cpp
    runtime/v41_session.cpp
    vision/prompt.cpp
  LINK_LIBRARIES
    gufo_deepseek41flash
    gufo_deepseek41flash_mmq
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_cache
    gufo_ext_hip
    roc::hipblas
    roc::hipblaslt
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

if(TARGET gufo_deepseek41flash_hip)
  gufo_extension_hip_sources(gufo_deepseek41flash_hip
    SOURCES
      kernels/rocm/backend.hip.cpp
      kernels/rocm/ds41_expert_gemv.hip.cpp
      kernels/rocm/ds41_gemv.hip.cpp
      kernels/rocm/ds41_v41.hip.cpp
      vision/encoder.hip
      vision/lt_gemm.hip
  )
  gufo_extension_hip_sources(gufo_deepseek41flash_hip
    SOURCES
      kernels/rocm/ds41_dense_f16.hip.cpp
      kernels/rocm/ds41_routed_f16.hip.cpp
  )
  target_compile_definitions(gufo_deepseek41flash_hip PRIVATE
    _GNU_SOURCE
    DS41_ROCM_BUILD=1)
  target_compile_options(gufo_deepseek41flash_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:${GUFO_CLANG_OPTION}-O3>
    ${GUFO_CLANG_OPTION}-ffast-math ${GUFO_CLANG_OPTION}-fno-finite-math-only
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>)
  target_include_directories(gufo_deepseek41flash_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>")
  target_include_directories(gufo_deepseek41flash_hip SYSTEM PUBLIC
    "$<BUILD_INTERFACE:${DEEPSEEK41FLASH_RUNTIME_DIR}>"
    "$<BUILD_INTERFACE:${DEEPSEEK41FLASH_ROCM_DIR}>"
    ${ROCWMMA_INCLUDE_DIR})
endif()

# Serving runner, prompt/chat/bench drivers and the provider registration.
gufo_extension_target(
  NAME gufo_deepseek41flash_text
  SOURCES
    drivers.cpp
    provider.cpp
    runner.cpp
  LINK_LIBRARIES
    gufo_deepseek41flash_hip
    gufo_deepseek41flash
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)
if(TARGET gufo_deepseek41flash_text)
  target_compile_options(gufo_deepseek41flash_text PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
endif()

install(FILES
  "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  "${GUFO_EXTENSION_BUILD_ROOT}/LICENSES/gufo.txt"
  "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE.ds4"
  DESTINATION share/licenses/gufo/extensions/deepseek41flash)

# Tests. Host contracts and kernel checks need no model. `deepseek41flash_tests`
# builds every test binary, so ctest never runs a stale one.
if(BUILD_TESTING AND NOT TARGET deepseek41flash_tests)
  add_custom_target(deepseek41flash_tests)
endif()

function(_deepseek41flash_test name labels)
  cmake_parse_arguments(ARG "" "" "LINK_LIBRARIES" ${ARGN})
  set(target deepseek41flash_${name}_test)
  gufo_extension_target(
    NAME ${target}
    SOURCES tests/${name}_test.cpp
    LINK_LIBRARIES ${ARG_LINK_LIBRARIES} gufo_ext_runtime
  )
  if(NOT TARGET ${target})
    return()
  endif()
  target_compile_definitions(${target} PRIVATE
    GUFO_DS41_CHAT_GOLDENS="${GUFO_EXTENSION_BUILD_ROOT}/tests/fixtures/encoding-1017.json")
  add_test(NAME deepseek41flash.${name} COMMAND ${target})
  set_tests_properties(deepseek41flash.${name} PROPERTIES
    LABELS "deepseek41flash;${labels}" SKIP_RETURN_CODE 77)
  add_dependencies(deepseek41flash_tests ${target})
endfunction()

_deepseek41flash_test(chat_template "cpu;tokenization"
  LINK_LIBRARIES gufo_deepseek41flash)
_deepseek41flash_test(text_contract "cpu;tools"
  LINK_LIBRARIES gufo_deepseek41flash)
_deepseek41flash_test(direction_adapter "cpu;gguf"
  LINK_LIBRARIES gufo_deepseek41flash)
_deepseek41flash_test(lora_adapter "cpu;gguf"
  LINK_LIBRARIES gufo_deepseek41flash)

if(ENGINE_ENABLE_HIP)
  _deepseek41flash_test(vision_prompt "cpu;vision"
    LINK_LIBRARIES gufo_deepseek41flash_hip gufo_deepseek41flash)
  _deepseek41flash_test(gguf_loader "cpu;gguf"
    LINK_LIBRARIES gufo_deepseek41flash_hip gufo_deepseek41flash gufo_ext_hip)
  _deepseek41flash_test(checkpoint_size "cpu;cache"
    LINK_LIBRARIES gufo_deepseek41flash_hip gufo_deepseek41flash gufo_ext_hip)
  foreach(name IN ITEMS attention candidate_mask draft_sampling expert_store
                        hc_state_pass mmq_typed steer)
    _deepseek41flash_test(${name} "hip;gpu;gfx1151;quality"
      LINK_LIBRARIES gufo_deepseek41flash_hip gufo_deepseek41flash gufo_ext_hip)
  endforeach()
  if(TARGET deepseek41flash_mmq_typed_test)
    target_link_libraries(deepseek41flash_mmq_typed_test PRIVATE
      gufo_deepseek41flash_mmq)
  endif()

  # Probes and kernel benchmarks: quality gates and measurements against a
  # model (docs/models/deepseek-v4.1-flash/QUALITY.md). Built by
  # `deepseek41flash_probes`, not by the default or test targets.
  if(BUILD_TESTING AND NOT TARGET deepseek41flash_probes)
    add_custom_target(deepseek41flash_probes)
  endif()
  foreach(name IN ITEMS load_probe mmq_bench gemv_bench vision_probe)
    gufo_extension_target(
      NAME deepseek41flash_${name}
      SOURCES tools/probe/${name}.cpp
      LINK_LIBRARIES gufo_deepseek41flash_hip gufo_deepseek41flash
                     gufo_deepseek41flash_mmq gufo_ext_hip gufo_ext_runtime
    )
    if(TARGET deepseek41flash_${name})
      set_target_properties(deepseek41flash_${name} PROPERTIES
        EXCLUDE_FROM_ALL TRUE OUTPUT_NAME ds41_${name})
      add_dependencies(deepseek41flash_probes deepseek41flash_${name})
    endif()
  endforeach()
endif()
