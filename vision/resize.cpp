// Copied from the Gufo fork's feat/gemma src/core/image.cpp (83b764b4).
#include "vision/resize.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gufo::models::gemma4::vision {
namespace {

double Cubic(double x) {
  x = std::abs(x);
  if (x < 1)
    return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2)
    return (((-0.5 * x + 2.5) * x - 4) * x) + 2;
  return 0;
}

struct Filter {
  std::uint32_t begin;
  std::vector<std::int16_t> weights;
};

struct FilterBank {
  std::vector<Filter> rows;
  unsigned precision{0};
};

FilterBank Filters(std::uint32_t input, std::uint32_t output) {
  const double scale = static_cast<double>(input) / output;
  const double antialias = std::max(scale, 1.0);
  const double support = antialias * 2;
  FilterBank filters;
  filters.rows.reserve(output);
  std::vector<std::vector<double>> coefficients;
  coefficients.reserve(output);
  double maximum = 0;
  for (std::uint32_t i = 0; i < output; ++i) {
    const double center = (i + 0.5) * scale;
    const auto begin = std::max(0, static_cast<int>(center - support + 0.5));
    const auto end = std::min(static_cast<int>(input),
                              static_cast<int>(center + support + 0.5));
    Filter filter{static_cast<std::uint32_t>(begin), {}};
    auto& weights = coefficients.emplace_back();
    double sum = 0;
    for (int j = begin; j < end; ++j) {
      const double weight = Cubic((j + 0.5 - center) / antialias);
      weights.push_back(weight);
      sum += weight;
    }
    for (auto& weight : weights) {
      weight /= sum;
      maximum = std::max(maximum, weight);
    }
    filters.rows.push_back(std::move(filter));
  }
  // PyTorch's uint8 antialiased resize quantizes each axis' coefficients
  // to int16, choosing one shared precision for that axis.
  for (; filters.precision < 22; ++filters.precision) {
    if (static_cast<int>(0.5 + maximum * (1U << (filters.precision + 1))) >=
        (1 << 15))
      break;
  }
  for (std::size_t i = 0; i < coefficients.size(); ++i) {
    for (const auto weight : coefficients[i]) {
      const double scaled = weight * (1U << filters.precision);
      filters.rows[i].weights.push_back(
          static_cast<std::int16_t>(scaled < 0 ? scaled - 0.5 : scaled + 0.5));
    }
  }
  return filters;
}

std::uint8_t Pixel(std::int64_t value, unsigned precision) {
  return static_cast<std::uint8_t>(
      std::clamp<std::int64_t>(value >> precision, 0, 255));
}

}  // namespace

core::Image ResizeBicubic(const core::Image& image, std::uint32_t width,
                          std::uint32_t height) {
  if (image.width == 0 || image.height == 0 || width == 0 || height == 0 ||
      image.pixels.size() != std::size_t{image.width} * image.height * 3) {
    throw std::invalid_argument("invalid image resize");
  }
  if (width == image.width && height == image.height)
    return image;
  const auto horizontal = Filters(image.width, width);
  const auto vertical = Filters(image.height, height);
  std::vector<std::uint8_t> intermediate(std::size_t{width} * image.height * 3);
  for (std::uint32_t y = 0; y < image.height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      for (std::size_t c = 0; c < 3; ++c) {
        std::int64_t value = std::int64_t{1} << (horizontal.precision - 1);
        const auto& filter = horizontal.rows[x];
        for (std::size_t j = 0; j < filter.weights.size(); ++j) {
          value +=
              image.pixels[(std::size_t{y} * image.width + filter.begin + j) *
                               3 +
                           c] *
              filter.weights[j];
        }
        intermediate[(std::size_t{y} * width + x) * 3 + c] =
            Pixel(value, horizontal.precision);
      }
    }
  }
  core::Image resized{
      width, height,
      std::vector<std::uint8_t>(std::size_t{width} * height * 3)};
  for (std::uint32_t y = 0; y < height; ++y) {
    const auto& filter = vertical.rows[y];
    for (std::uint32_t x = 0; x < width; ++x) {
      for (std::size_t c = 0; c < 3; ++c) {
        std::int64_t value = std::int64_t{1} << (vertical.precision - 1);
        for (std::size_t j = 0; j < filter.weights.size(); ++j) {
          value += intermediate[((filter.begin + j) * width + x) * 3 + c] *
                   filter.weights[j];
        }
        resized.pixels[(std::size_t{y} * width + x) * 3 + c] =
            Pixel(value, vertical.precision);
      }
    }
  }
  return resized;
}

}  // namespace gufo::models::gemma4::vision
