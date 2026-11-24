// DeepSeek V4.1 Flash GGUF loader: header counts are bounded by the file
// before they size the metadata and tensor tables.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "runtime/native_internal.h"

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

void PutU32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i)
    out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

void PutU64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i)
    out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

void PutString(std::vector<std::uint8_t>& out, const std::string& text) {
  PutU64(out, text.size());
  out.insert(out.end(), text.begin(), text.end());
}

// A GGUF v3 header followed by `n_kv` string pairs and `entries` F32 tensors
// of four elements, though the header claims `n_tensors`.
std::vector<std::uint8_t> Gguf(std::uint64_t n_tensors, std::uint64_t n_kv,
                               std::uint64_t entries) {
  std::vector<std::uint8_t> out;
  PutU32(out, 0x46554747);  // "GGUF"
  PutU32(out, 3);
  PutU64(out, n_tensors);
  PutU64(out, n_kv);
  if (n_kv == 1) {
    PutString(out, "general.name");
    PutU32(out, 8);  // string
    PutString(out, "test");
  }
  for (std::uint64_t i = 0; i < entries; ++i) {
    PutString(out, "t" + std::to_string(i));
    PutU32(out, 1);  // ndim
    PutU64(out, 4);
    PutU32(out, 0);  // F32
    PutU64(out, i * 16);
  }
  out.resize((out.size() + 31) / 32 * 32 + entries * 16, 0);
  return out;
}

std::string WriteTemp(const std::vector<std::uint8_t>& bytes) {
  char path[] = "/tmp/ds41_gguf_loader_XXXXXX";
  const int fd = mkstemp(path);
  if (fd < 0 || write(fd, bytes.data(), bytes.size()) !=
                    static_cast<ssize_t>(bytes.size())) {
    std::perror("ds41 gguf_loader_test: temp file");
    std::exit(1);
  }
  close(fd);
  return path;
}

// Opens the bytes as a model; returns the loader's error, or "" on success
// after checking the tensor count.
std::string Open(const std::vector<std::uint8_t>& bytes,
                 std::uint64_t expect_tensors = 0) {
  const std::string path = WriteTemp(bytes);
  ds41_model model{};
  std::string error;
  try {
    ds41_model_open(&model, path.c_str());
    Expect(model.n_tensors == expect_tensors, "tensor count of a valid file");
    ds41_model_close(&model);
  } catch (const std::runtime_error& e) {
    error = e.what();
  }
  unlink(path.c_str());
  return error;
}

}  // namespace

int main() {
  Expect(Open(Gguf(2, 1, 2), 2).empty(), "a valid file opens");

  // (2^57 + 1) * sizeof(ds41_tensor) wraps to one entry; the count is
  // rejected before the table is allocated, not by the walk past it.
  static_assert(sizeof(ds41_tensor) == 128);
  const std::uint64_t wraps = (1ull << 57) + 1;
  Expect(Open(Gguf(wraps, 0, 2)).find("record count exceeds") !=
             std::string::npos,
         "a tensor count that wraps the table size is rejected");
  Expect(Open(Gguf(UINT64_MAX, 0, 1)).find("record count exceeds") !=
             std::string::npos,
         "the largest tensor count is rejected");
  Expect(!Open(Gguf(3, 0, 2)).empty(),
         "a tensor count one past the entries present is rejected");
  Expect(Open(Gguf(0, 1ull << 40, 0)).find("record count exceeds") !=
             std::string::npos,
         "a metadata count beyond the file is rejected");

  if (failures != 0)
    return 1;
  std::printf("gguf_loader: ok\n");
  return 0;
}
