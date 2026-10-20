#include "src/models/gemma4/cycle_trace.hpp"

#include <chrono>
#include <stdexcept>

#include "src/core/json.hpp"

namespace gufo::models::gemma4 {
namespace {

json::Value Floats(const std::vector<float>& values) {
  json::Value array = json::Value::array();
  for (const float v : values) {
    array.push_back(static_cast<double>(v));
  }
  return array;
}

}  // namespace

std::string FormatCycleRecord(const CycleRecord& r) {
  json::Value v = json::Value::object();
  v["forward"] = static_cast<unsigned long long>(r.forward);
  v["sessions"] = static_cast<std::size_t>(r.sessions);
  v["position"] = static_cast<std::size_t>(r.position);
  v["sampled"] = r.sampled;
  v["rows"] = static_cast<std::size_t>(r.rows);
  v["copied"] = static_cast<std::size_t>(r.copied);
  v["siblings"] = static_cast<std::size_t>(r.siblings);
  v["drafter_steps"] = static_cast<std::size_t>(r.drafter_steps);
  v["small_head"] = r.small_head;
  v["accepted"] = static_cast<std::size_t>(r.accepted);
  v["sibling_accepted"] = r.sibling_accepted;
  v["emitted"] = static_cast<std::size_t>(r.emitted);
  v["priced_draft_ms"] = static_cast<double>(r.priced_draft_ms);
  v["priced_verify_ms"] = static_cast<double>(r.priced_verify_ms);
  v["expected"] = static_cast<double>(r.expected);
  v["signals"] = Floats(r.signals);
  v["estimates"] = Floats(r.estimates);
  v["margins"] = Floats(r.margins);
  v["sibling_estimates"] = Floats(r.sibling_estimates);
  if (r.stop) {
    json::Value stop = json::Value::object();
    stop["signal"] = static_cast<double>(r.stop->signal);
    stop["estimate"] = static_cast<double>(r.stop->estimate);
    stop["gain"] = static_cast<double>(r.stop->gain);
    stop["cost"] = static_cast<double>(r.stop->cost);
    v["stop"] = std::move(stop);
  } else {
    v["stop"] = nullptr;
  }
  v["begin_ns"] = static_cast<long long>(r.begin_ns);
  v["drafted_ns"] = static_cast<long long>(r.drafted_ns);
  v["verified_ns"] = static_cast<long long>(r.verified_ns);
  v["end_ns"] = static_cast<long long>(r.end_ns);
  return v.dump();
}

CycleTrace::CycleTrace(const std::string& path) : out_(path, std::ios::app) {
  if (!out_) {
    throw std::runtime_error("cannot open draft trace " + path);
  }
}

void CycleTrace::Write(const CycleRecord& record) {
  const std::string line = FormatCycleRecord(record);
  std::lock_guard lock(mutex_);
  out_ << line << '\n';
  out_.flush();
}

std::int64_t CycleTrace::Now() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace gufo::models::gemma4
