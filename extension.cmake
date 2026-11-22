# Gemma 4 (31B, 31B QAT, 26B-A4B) external model extension: dense and
# mixture-of-experts `gemma4` targets, the `gemma4-assistant` MTP drafter and
# the vision sidecar.

# Host side: configuration, tokenizer, chat template, tool syntax, draft
# policy, vision prompt and weight maps, artifact probe and registration.
gufo_extension_target(
  NAME gufo_gemma4
  SOURCES
    chat_template.cpp
    config.cpp
    cycle_trace.cpp
    draft_policy.cpp
    draft_vocab.cpp
    extension.cpp
    probe.cpp
    text_contract.cpp
    tokenizer.cpp
    tool_syntax.cpp
    vision/prompt.cpp
    vision/resize.cpp
    vision/weights.cpp
    weights.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
    ICU::uc
    Threads::Threads
)

if(TARGET gufo_gemma4)
  target_include_directories(gufo_gemma4 PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  # Registration leaves the provider to gufo_gemma4_text in HIP builds.
  if(ENGINE_ENABLE_HIP)
    set_property(SOURCE "${GUFO_EXTENSION_BUILD_ROOT}/extension.cpp"
      APPEND PROPERTY COMPILE_DEFINITIONS ENGINE_ENABLE_HIP=1)
  endif()
endif()

# Device side: Gemma's kernels, executor, engine, vision encoder and
# continuation adapter. hip/half_gemm and hip/ggml are excerpts of Gufo
# kernels (each file names its source).
gufo_extension_target(
  NAME gufo_gemma4_hip
  SOURCES
    continuation_adapter.cpp
    engine.cpp
    hip/device_model.cpp
    hip/executor.cpp
  LINK_LIBRARIES
    gufo_gemma4
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_cache
    gufo_ext_hip
    roc::hipblas
    Threads::Threads
)

if(TARGET gufo_gemma4_hip)
  set(_gemma4_hip_options -O3 -ffast-math -fno-finite-math-only
    -Wno-unused-command-line-argument)
  gufo_extension_hip_sources(gufo_gemma4_hip
    SOURCES
      hip/candidates.hip.cpp
      hip/kernels.hip.cpp
      hip/moe_prefill.hip.cpp
      hip/half_gemm.hip.cpp
      vision/encoder.hip
    OPTIONS ${_gemma4_hip_options}
  )
  # Projections must round identically at every width (verification rows
  # equal single-token decode bit for bit), so they keep strict
  # floating-point semantics. So does prefill attention: under Clang 24, fast
  # math makes a row's result depend on the rest of its forward, and a prefix
  # evaluated with a lookahead then differs from the same prefix on its own.
  gufo_extension_hip_sources(gufo_gemma4_hip
    SOURCES
      hip/attention_wmma.hip.cpp
      hip/gemv.hip.cpp
      hip/moe.hip.cpp
      hip/wmma_gemv.hip.cpp
    OPTIONS ${_gemma4_hip_options} -fno-fast-math -ffp-contract=off
  )
  # The GGML-format units keep their upstream options: the configuration's
  # HIP flags without fast math (decode GEMV and the small-batch passes agree
  # bit for bit), the wave64 prefill tiles built for their wave size, and no
  # SLP vectorization in the blocked K-quant prefill tiles, which otherwise
  # spill.
  gufo_extension_hip_sources(gufo_gemma4_hip
    SOURCES
      hip/ggml/argmax.hip.cpp
      hip/ggml/gemv.hip.cpp
      hip/ggml/gemv_quant.hip.cpp
      hip/ggml/prefill_embed.hip.cpp
  )
  gufo_extension_hip_sources(gufo_gemma4_hip
    SOURCES
      hip/ggml/prefill_quant_gemm.hip.cpp
    OPTIONS -fno-slp-vectorize
  )
  gufo_extension_hip_sources(gufo_gemma4_hip
    SOURCES
      hip/ggml/prefill_quant_wave64.hip.cpp
    OPTIONS -mwavefrontsize64 -fno-slp-vectorize
  )
  target_compile_options(gufo_gemma4_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
  target_include_directories(gufo_gemma4_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# The serving runner, the prompt, chat and bench drivers and the provider
# registration.
gufo_extension_target(
  NAME gufo_gemma4_text
  SOURCES
    drivers.cpp
    provider.cpp
    runner.cpp
  LINK_LIBRARIES
    gufo_gemma4_hip
    gufo_gemma4
    gufo_ext_runtime
    gufo_ext_hip
)
if(TARGET gufo_gemma4_text)
  target_compile_options(gufo_gemma4_text PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
endif()

# The scalar oracle decodes every weight on the fly; unoptimized it is
# unusable even as an oracle, so it keeps -O2 in every configuration. Only
# validation targets build it.
gufo_extension_target(
  NAME gufo_gemma4_reference
  SOURCES
    reference.cpp
    vision/reference.cpp
  LINK_LIBRARIES
    gufo_gemma4
    gufo_ext_runtime
)

if(TARGET gufo_gemma4_reference)
  target_compile_options(gufo_gemma4_reference PRIVATE -O2)
  # Without OpenMP its parallel loops silently run on one thread.
  find_package(OpenMP REQUIRED)
  target_link_libraries(gufo_gemma4_reference PRIVATE OpenMP::OpenMP_CXX)
endif()

# Tests. Host contracts need no model; model-backed checks read the GGUF named
# by GUFO_GEMMA4_MODEL (with GUFO_GEMMA4_MTP_MODEL, GUFO_GEMMA4_MMPROJ) and
# skip (77) without it. `gemma4_tests` builds every test binary, so ctest
# never runs a stale one.
if(BUILD_TESTING AND NOT TARGET gemma4_tests)
  add_custom_target(gemma4_tests)
endif()

set(_gemma4_fixtures "${GUFO_EXTENSION_BUILD_ROOT}/tests/fixtures")

function(_gemma4_test name labels)
  cmake_parse_arguments(ARG "MODEL" "" "SOURCES;LINK_LIBRARIES" ${ARGN})
  set(target gemma4_${name}_test)
  gufo_extension_target(
    NAME ${target}
    SOURCES ${ARG_SOURCES}
    LINK_LIBRARIES ${ARG_LINK_LIBRARIES} gufo_ext_runtime
  )
  if(NOT TARGET ${target})
    return()
  endif()
  target_compile_definitions(${target} PRIVATE
    GUFO_GEMMA4_FIXTURES="${_gemma4_fixtures}"
    GUFO_CHAT_TEMPLATE_HF_GOLDENS="${_gemma4_fixtures}/chat_template_hf_goldens.json")
  add_test(NAME gemma4.${name} COMMAND ${target})
  set_tests_properties(gemma4.${name} PROPERTIES
    LABELS "gemma4;${labels}" SKIP_RETURN_CODE 77)
  if(ARG_MODEL)
    set_tests_properties(gemma4.${name} PROPERTIES RESOURCE_LOCK gemma4)
  endif()
  add_dependencies(gemma4_tests ${target})
endfunction()

_gemma4_test(config "cpu"
  SOURCES tests/config_test.cpp LINK_LIBRARIES gufo_gemma4)
_gemma4_test(cycle_trace "cpu;speculative"
  SOURCES tests/cycle_trace_test.cpp LINK_LIBRARIES gufo_gemma4)
_gemma4_test(draft_policy "cpu;speculative"
  SOURCES tests/draft_policy_test.cpp LINK_LIBRARIES gufo_gemma4)
_gemma4_test(draft_vocab "cpu;speculative"
  SOURCES tests/draft_vocab_test.cpp LINK_LIBRARIES gufo_gemma4)
_gemma4_test(tokenizer_contract "cpu;tokenization"
  SOURCES tests/tokenizer_contract_test.cpp LINK_LIBRARIES gufo_gemma4)
_gemma4_test(template "cpu;chat_template"
  SOURCES tests/template_test.cpp LINK_LIBRARIES gufo_gemma4)
_gemma4_test(tool_syntax "cpu;tools"
  SOURCES tests/tool_syntax_test.cpp LINK_LIBRARIES gufo_gemma4)
_gemma4_test(vision_prompt "cpu;vision"
  SOURCES tests/vision_prompt_test.cpp LINK_LIBRARIES gufo_gemma4)
_gemma4_test(text_contract "cpu;tools;reasoning;vision"
  SOURCES tests/text_contract_test.cpp LINK_LIBRARIES gufo_gemma4)
_gemma4_test(artifact "cpu;tokenization;chat_template;external-model"
  SOURCES tests/artifact_test.cpp LINK_LIBRARIES gufo_gemma4)

foreach(name IN ITEMS attention_ops kernel_ops moe_ops projection_ops)
  _gemma4_test(${name} "hip;gpu;gfx1151"
    SOURCES tests/${name}_test.cpp
    LINK_LIBRARIES gufo_gemma4_hip gufo_gemma4 gufo_ext_hip)
endforeach()
_gemma4_test(target "hip;gpu;gfx1151;quality;external-model;slow" MODEL
  SOURCES tests/target_test.cpp tests/device_headroom_fault.cpp
  LINK_LIBRARIES gufo_gemma4_hip gufo_gemma4_reference gufo_gemma4 gufo_ext_hip
    ${CMAKE_DL_LIBS})
_gemma4_test(continuation "hip;gpu;gfx1151;cache;external-model" MODEL
  SOURCES tests/continuation_test.cpp
  LINK_LIBRARIES gufo_gemma4_hip gufo_gemma4 gufo_ext_hip)
_gemma4_test(vision_encoder "hip;gpu;gfx1151;vision;external-model" MODEL
  SOURCES tests/vision_encoder_test.cpp
  LINK_LIBRARIES gufo_gemma4_hip gufo_gemma4_reference gufo_gemma4 gufo_ext_hip)
_gemma4_test(vision_session "hip;gpu;gfx1151;vision;external-model;slow" MODEL
  SOURCES tests/vision_session_test.cpp
  LINK_LIBRARIES gufo_gemma4_hip gufo_gemma4 gufo_ext_hip)
# The oracle-backed checks run unoptimized references; -O2 as for the oracle.
set_source_files_properties(
  "${GUFO_EXTENSION_BUILD_ROOT}/tests/target_test.cpp"
  "${GUFO_EXTENSION_BUILD_ROOT}/tests/vision_encoder_test.cpp"
  "${GUFO_EXTENSION_BUILD_ROOT}/tests/reference_probe.cpp"
  PROPERTIES COMPILE_OPTIONS "-O2")
if(TARGET gemma4_continuation_test)
  # The same run with freed memory overwritten: waiting on a transfer after
  # its slot is destroyed passes only by reading the settled result.
  add_test(NAME gemma4.continuation_perturbed COMMAND gemma4_continuation_test)
  set_tests_properties(gemma4.continuation_perturbed PROPERTIES
    LABELS "gemma4;hip;gpu;gfx1151;cache;external-model"
    SKIP_RETURN_CODE 77 RESOURCE_LOCK gemma4
    ENVIRONMENT "MALLOC_PERTURB_=165")
endif()

# Probes and tools, built on request.
function(_gemma4_tool target)
  cmake_parse_arguments(ARG "" "" "SOURCES;LINK_LIBRARIES" ${ARGN})
  gufo_extension_target(
    NAME ${target}
    SOURCES ${ARG_SOURCES}
    LINK_LIBRARIES ${ARG_LINK_LIBRARIES} gufo_ext_runtime
  )
  if(TARGET ${target})
    set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
endfunction()

# Offline replay of --draft-trace records through the production draft
# policy.
_gemma4_tool(gemma4_draft_replay
  SOURCES tools/draft_replay.cpp LINK_LIBRARIES gufo_gemma4)
_gemma4_tool(gemma4_reference_probe
  SOURCES tests/reference_probe.cpp
  LINK_LIBRARIES gufo_gemma4_reference gufo_gemma4)
if(ENGINE_ENABLE_HIP)
  _gemma4_tool(gemma4_vision_probe
    SOURCES tests/vision_probe.cpp
    LINK_LIBRARIES gufo_gemma4_hip gufo_gemma4_reference gufo_gemma4
      gufo_ext_hip)
  if(TARGET gemma4_vision_probe)
    target_compile_definitions(gemma4_vision_probe PRIVATE
      GUFO_GEMMA4_VISION_GPU)
  endif()
else()
  _gemma4_tool(gemma4_vision_probe
    SOURCES tests/vision_probe.cpp
    LINK_LIBRARIES gufo_gemma4_reference gufo_gemma4)
endif()
_gemma4_tool(gemma4_vision_generate_probe
  SOURCES tests/vision_generate_probe.cpp
  LINK_LIBRARIES gufo_gemma4_hip gufo_gemma4 gufo_ext_hip)
_gemma4_tool(gemma4_gpu_probe
  SOURCES tests/gpu_probe.cpp
  LINK_LIBRARIES gufo_gemma4_hip gufo_gemma4 gufo_ext_hip)

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/gemma4)
