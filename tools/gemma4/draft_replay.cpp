// Offline replay of the Gemma 4 calibrated draft policy over --draft-trace
// records, through the production DraftCalibration and CalibratedChain.
//
//   gemma4_draft_replay validate [--experts] [--min-drafts N] TRACE.jsonl
//     Rebuilds the shared calibration from the recorded observations and
//     checks every recorded estimate, decision margin, stop decision, sibling
//     estimate and expected token count of single-session cycles bit for
//     bit. Exit 1 on any mismatch.
//
//   gemma4_draft_replay simulate [--experts] [--replicas N] [--seed S]
//                                TRACE.jsonl...
//     Evaluates stopping rules on complete chains: cycles of a run with
//     --draft-policy fixed --draft-tokens 15 whose 15 drafter proposals were
//     all verified (no copies). Each row's acceptance label is conditional on
//     the rows before it, so a chain stopped after k drafts is expected to
//     emit 1 + sum_{j<k} prod_{i<=j} a_i tokens (siblings are not modelled).
//     Costs come from the policy's price tables. Policies: the calibrated
//     rule learning online (sampled outcomes drawn from the labels, averaged
//     over replicas), its frozen prior, fixed lengths 0..15 and a per-cycle
//     oracle. Also reports how well the learned estimates predict the labels.
//     Prints one JSON object.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/gemma4/draft_policy.hpp"

namespace g4 = gufo::models::gemma4;
using gufo::json::Value;

namespace {

std::vector<float> Floats(const Value& v, const char* key) {
  std::vector<float> out;
  if (const Value* a = v.find(key); a != nullptr && a->is_array()) {
    for (const Value& x : a->items()) {
      out.push_back(static_cast<float>(x.as_double()));
    }
  }
  return out;
}

std::vector<bool> Bools(const Value& v, const char* key) {
  std::vector<bool> out;
  if (const Value* a = v.find(key); a != nullptr && a->is_array()) {
    for (const Value& x : a->items()) {
      out.push_back(x.as_bool());
    }
  }
  return out;
}

std::vector<Value> Read(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot read " + path);
  }
  std::vector<Value> records;
  for (std::string line; std::getline(in, line);) {
    if (!line.empty()) {
      records.push_back(gufo::json::parse(line));
    }
  }
  return records;
}

struct Options {
  bool experts{false};
  std::uint32_t min_drafts{1};
  int replicas{8};
  std::uint64_t seed{1};
  std::vector<std::string> files;
};

// ---------------------------------------------------------------- validate

int Validate(const Options& opt) {
  std::array<g4::DraftCalibration, 4> tables;  // as Session::Calibration
  std::size_t cycles = 0;
  std::size_t checked = 0;
  std::size_t mismatches = 0;
  const auto mismatch = [&](std::size_t line, const std::string& what) {
    if (++mismatches <= 20) {
      std::cerr << "cycle " << line << ": " << what << '\n';
    }
  };
  for (const std::string& file : opt.files) {
    for (const Value& r : Read(file)) {
      ++cycles;
      const bool sampled = r.find("sampled")->as_bool();
      g4::DraftCalibration& drafts = tables[sampled ? 1 : 0];
      g4::DraftCalibration& siblings = tables[sampled ? 3 : 2];
      const std::vector<float> signals = Floats(r, "signals");
      const std::vector<bool> observed = Bools(r, "observed");
      const Value* stop = r.find("stop");
      const bool alone = r.member_size("sessions") == 1;
      if (alone && (!signals.empty() || (stop && !stop->is_null()))) {
        ++checked;
        const g4::DraftCosts costs = g4::DraftCostsAt(
            static_cast<std::uint32_t>(r.member_size("position")), opt.experts);
        g4::CalibratedChain chain(
            drafts, costs, opt.min_drafts,
            static_cast<std::uint32_t>(r.member_size("step_cap")));
        const std::vector<float> estimates = Floats(r, "estimates");
        const std::vector<float> margins = Floats(r, "margins");
        for (std::size_t i = 0; i < signals.size(); ++i) {
          if (drafts.Estimate(signals[i]) != estimates[i]) {
            mismatch(cycles, "estimate " + std::to_string(i));
          }
          if (!chain.Include(signals[i])) {
            mismatch(cycles, "draft " + std::to_string(i) + " not included");
          }
          const auto d = chain.LastDecision();
          if ((d.compared ? d.gain - d.cost : 0.0F) != margins[i]) {
            mismatch(cycles, "margin " + std::to_string(i));
          }
        }
        if (stop && !stop->is_null()) {
          const auto signal = static_cast<float>(stop->member_double("signal"));
          if (chain.Include(signal)) {
            mismatch(cycles, "stop not reproduced");
          }
          const auto d = chain.LastDecision();
          if (d.gain != static_cast<float>(stop->member_double("gain")) ||
              d.cost != static_cast<float>(stop->member_double("cost"))) {
            mismatch(cycles, "stop gain/cost");
          }
        }
        const std::vector<float> sibling_estimates =
            Floats(r, "sibling_estimates");
        const std::vector<float> up_signals = Floats(r, "runner_up_signals");
        auto rows = static_cast<std::uint32_t>(r.member_size("rows") -
                                               sibling_estimates.size());
        for (std::size_t i = 0; i < sibling_estimates.size(); ++i) {
          const float hit = siblings.Estimate(up_signals[i]);
          if (hit != sibling_estimates[i]) {
            mismatch(cycles, "sibling estimate " + std::to_string(i));
          }
          if (!chain.IncludeSibling(static_cast<std::uint32_t>(i), rows++,
                                    hit)) {
            mismatch(cycles, "sibling " + std::to_string(i) + " not included");
          }
        }
        if (chain.Expected() !=
            static_cast<float>(r.member_double("expected"))) {
          mismatch(cycles, "expected tokens");
        }
      }
      for (std::size_t j = 0; j < observed.size(); ++j) {
        drafts.Observe(signals[j], observed[j]);
      }
      if (const Value* s = r.find("sibling_observed"); s && !s->is_null()) {
        const std::vector<float> up_signals = Floats(r, "runner_up_signals");
        siblings.Observe(up_signals[r.member_size("accepted")], s->as_bool());
      }
    }
  }
  Value out = Value::object();
  out["cycles"] = cycles;
  out["checked"] = checked;
  out["mismatches"] = mismatches;
  std::cout << out.dump() << '\n';
  return mismatches == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- simulate

constexpr std::size_t kChain = 15;

struct Chain {
  std::uint32_t position{0};
  bool sampled{false};
  std::array<float, kChain> signal{};
  std::array<float, kChain> accept{};  // conditional acceptance labels
};

/// Tokens a cycle verifying k drafts is expected to emit.
double Expected(const Chain& c, std::size_t k) {
  double survival = 1.0;
  double tokens = 1.0;
  for (std::size_t j = 0; j < k; ++j) {
    survival *= c.accept[j];
    tokens += survival;
  }
  return tokens;
}

/// Priced time of a cycle that ran `steps` drafter steps and verifies k.
double Cost(const g4::DraftCosts& costs, std::size_t steps, std::size_t k) {
  return costs.draft[steps] + costs.verify[k + 1];
}

struct Tally {
  double tokens{0.0};
  double ms{0.0};
  double drafts{0.0};
  double cycles{0.0};
  void Add(double t, double m, double d) {
    tokens += t;
    ms += m;
    drafts += d;
    cycles += 1.0;
  }
  [[nodiscard]] Value Json() const {
    Value v = Value::object();
    v["tokens_per_s"] = ms > 0.0 ? 1000.0 * tokens / ms : 0.0;
    v["tokens_per_cycle"] = cycles > 0.0 ? tokens / cycles : 0.0;
    v["drafts_per_cycle"] = cycles > 0.0 ? drafts / cycles : 0.0;
    return v;
  }
};

struct Calibration {
  // Per bin: estimates against labels; per position: the same.
  std::array<double, 8> est{}, label{}, count{};
  std::array<double, kChain> pos_est{}, pos_label{}, pos_count{};
  double brier{0.0}, log_loss{0.0}, n{0.0};
  double lag_xy{0.0}, lag_xx{0.0}, lag_yy{0.0}, lag_n{0.0};
  void Add(std::size_t position, float signal, double e, double a) {
    const std::size_t bin = g4::DraftCalibration::Bin(signal);
    est[bin] += e;
    label[bin] += a;
    count[bin] += 1.0;
    pos_est[position] += e;
    pos_label[position] += a;
    pos_count[position] += 1.0;
    const double p = std::clamp(e, 1e-6, 1.0 - 1e-6);
    brier += (e - a) * (e - a);
    log_loss -= a * std::log(p) + (1.0 - a) * std::log(1.0 - p);
    n += 1.0;
  }
  void AddLag(double r0, double r1) {
    lag_xy += r0 * r1;
    lag_xx += r0 * r0;
    lag_yy += r1 * r1;
    lag_n += 1.0;
  }
  [[nodiscard]] Value Json() const {
    Value v = Value::object();
    v["labels"] = n;
    v["brier_vs_label"] = n > 0 ? brier / n : 0.0;
    v["log_loss"] = n > 0 ? log_loss / n : 0.0;
    Value bins = Value::array();
    for (std::size_t b = 0; b < est.size(); ++b) {
      Value x = Value::object();
      x["bin"] = b;
      x["count"] = count[b];
      x["mean_estimate"] = count[b] > 0 ? est[b] / count[b] : 0.0;
      x["mean_label"] = count[b] > 0 ? label[b] / count[b] : 0.0;
      bins.push_back(std::move(x));
    }
    v["bins"] = std::move(bins);
    Value pos = Value::array();
    for (std::size_t j = 0; j < kChain; ++j) {
      Value x = Value::object();
      x["position"] = j;
      x["count"] = pos_count[j];
      x["mean_estimate"] = pos_count[j] > 0 ? pos_est[j] / pos_count[j] : 0.0;
      x["mean_label"] = pos_count[j] > 0 ? pos_label[j] / pos_count[j] : 0.0;
      pos.push_back(std::move(x));
    }
    v["positions"] = std::move(pos);
    v["residual_lag1_correlation"] =
        lag_xx > 0 && lag_yy > 0 ? lag_xy / std::sqrt(lag_xx * lag_yy) : 0.0;
    return v;
  }
};

int Simulate(const Options& opt) {
  std::vector<Chain> chains;
  std::size_t skipped = 0;
  for (const std::string& file : opt.files) {
    for (const Value& r : Read(file)) {
      const std::vector<float> s = Floats(r, "proposal_signals");
      const std::vector<float> a = Floats(r, "row_acceptance");
      if (r.member_size("copied") != 0 || s.size() != kChain ||
          a.size() != kChain) {
        ++skipped;
        continue;
      }
      Chain c;
      c.position = static_cast<std::uint32_t>(r.member_size("position"));
      c.sampled = r.find("sampled")->as_bool();
      std::copy(s.begin(), s.end(), c.signal.begin());
      std::copy(a.begin(), a.end(), c.accept.begin());
      chains.push_back(c);
    }
  }
  std::vector<Tally> fixed(kChain + 1);
  Tally oracle;
  Tally learned;
  Tally prior;
  Calibration fit;
  for (const Chain& c : chains) {
    const g4::DraftCosts costs = g4::DraftCostsAt(c.position, opt.experts);
    for (std::size_t k = 0; k <= kChain; ++k) {
      fixed[k].Add(Expected(c, k), Cost(costs, k, k), static_cast<double>(k));
    }
    // The cycle's best length for its expected rate.
    std::size_t best = 0;
    double rate = 0.0;
    for (std::size_t k = 0; k <= kChain; ++k) {
      const double r = Expected(c, k) / Cost(costs, k, k);
      if (r > rate) {
        rate = r;
        best = k;
      }
    }
    oracle.Add(Expected(c, best), Cost(costs, best, best),
               static_cast<double>(best));
  }
  const auto run_policy = [&](const g4::DraftCalibration& table,
                              const Chain& c, const g4::DraftCosts& costs,
                              Tally* tally) {
    g4::CalibratedChain chain(table, costs, opt.min_drafts, kChain);
    std::size_t k = 0;
    while (k < kChain && chain.Include(c.signal[k])) {
      ++k;
    }
    // A declined proposal still ran its drafter step.
    const std::size_t steps = std::min(k + 1, kChain);
    tally->Add(Expected(c, k), Cost(costs, steps, k), static_cast<double>(k));
    return k;
  };
  const g4::DraftCalibration frozen;
  std::mt19937_64 rng(opt.seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (int replica = 0; replica < opt.replicas; ++replica) {
    std::array<g4::DraftCalibration, 2> tables;  // greedy, sampled
    for (const Chain& c : chains) {
      const g4::DraftCosts costs = g4::DraftCostsAt(c.position, opt.experts);
      g4::DraftCalibration& table = tables[c.sampled ? 1 : 0];
      if (replica == 0) {
        run_policy(frozen, c, costs, &prior);
        // How the current table predicts every labelled row.
        double previous = 0.0;
        for (std::size_t j = 0; j < kChain; ++j) {
          const double e = table.Estimate(c.signal[j]);
          fit.Add(j, c.signal[j], e, c.accept[j]);
          const double residual = c.accept[j] - e;
          if (j > 0) {
            fit.AddLag(previous, residual);
          }
          previous = residual;
        }
      }
      const std::size_t k = run_policy(table, c, costs, &learned);
      // Learn as FinishCycle does: verified drafts up to the first
      // rejection, drawn from the labels.
      for (std::size_t j = 0; j < k; ++j) {
        const bool accepted = uniform(rng) < c.accept[j];
        table.Observe(c.signal[j], accepted);
        if (!accepted) {
          break;
        }
      }
    }
  }
  Value out = Value::object();
  out["chains"] = chains.size();
  out["skipped_cycles"] = skipped;
  out["replicas"] = opt.replicas;
  out["calibrated_learning"] = learned.Json();
  out["calibrated_frozen_prior"] = prior.Json();
  out["oracle_per_cycle"] = oracle.Json();
  Value fixed_json = Value::array();
  for (std::size_t k = 0; k <= kChain; ++k) {
    Value x = fixed[k].Json();
    x["drafts"] = k;
    fixed_json.push_back(std::move(x));
  }
  out["fixed"] = std::move(fixed_json);
  out["calibration_fit"] = fit.Json();
  std::cout << out.dump() << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::cerr << "usage: gemma4_draft_replay validate|simulate [--experts] "
                   "[--min-drafts N] [--replicas N] [--seed S] TRACE...\n";
      return 2;
    }
    const std::string mode = argv[1];
    Options opt;
    for (int i = 2; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--experts") {
        opt.experts = true;
      } else if (arg == "--min-drafts" && i + 1 < argc) {
        opt.min_drafts = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      } else if (arg == "--replicas" && i + 1 < argc) {
        opt.replicas = std::stoi(argv[++i]);
      } else if (arg == "--seed" && i + 1 < argc) {
        opt.seed = std::stoull(argv[++i]);
      } else {
        opt.files.push_back(arg);
      }
    }
    if (mode == "validate") {
      return Validate(opt);
    }
    if (mode == "simulate") {
      return Simulate(opt);
    }
    std::cerr << "unknown mode " << mode << '\n';
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
