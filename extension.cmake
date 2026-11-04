# Gemma 4 (31B, 31B QAT, 26B-A4B) external model extension.

gufo_extension_target(
  NAME gufo_gemma4
  SOURCES
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
)

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/gemma4)
