#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_PROVIDER_OPTIONS_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_PROVIDER_OPTIONS_HPP_

#include <cmath>
#include <cstdint>
#include <gufo/ext/options.hpp>
#include <stdexcept>
#include <string>

#include "engine.hpp"

namespace gufo::models::deepseek_v41_flash {

/// Applies the options `extension.json` declares to `options`; an option that
/// was not given keeps the default. Throws std::invalid_argument for values
/// out of range.
inline void ApplyProviderOptions(const ext::ParsedOptions& given,
                                 ModelOptions& options) {
  using ext::FindOption;
  if (const auto* exact = FindOption<bool>(given, "--exact-prefill")) {
    options.exact_prefill = *exact;
  }
  if (const auto* stream = FindOption<bool>(given, "--stream-experts")) {
    options.stream_experts = *stream;
  }
  if (const auto* bytes =
          FindOption<std::int64_t>(given, "--expert-cache-bytes")) {
    if (*bytes <= 0) {
      throw std::invalid_argument("--expert-cache-bytes must be positive");
    }
    options.expert_cache_bytes = static_cast<std::uint64_t>(*bytes);
  }
  if (const auto* path = FindOption<std::string>(given, "--adapter")) {
    options.adapter_path = *path;
  }
  if (const auto* scale = FindOption<double>(given, "--adapter-scale")) {
    if (!std::isfinite(*scale)) {
      throw std::invalid_argument("--adapter-scale must be finite");
    }
    options.adapter_scale = static_cast<float>(*scale);
  }
}

}  // namespace gufo::models::deepseek_v41_flash

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_PROVIDER_OPTIONS_HPP_
