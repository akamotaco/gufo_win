# Windows 포팅 작업 노트 (win/gemma 브랜치)

이 저장소의 `win/gemma` 브랜치를 고치거나 업데이트할 사람(또는 AI)을 위한 기록이다.
사용자용 운영 가이드는 저장소 밖 `..\gufo-windows.md`(업데이트 스크립트 `..\gufo_update.bat`)에 있다.
포팅 설계 자체는 pixmaate가 쓴 [WINDOWS.md](WINDOWS.md)에 있다. 이 문서는 그 위에 이 장비에서
2026-10-18에 겪은 문제와 결정을 적는다.

## 1. 브랜치 구조와 원칙

- `win/gemma` = Windows 커밋들 + `origin/feat/gemma`(franzmoca/gufo)를 **merge**로 받아들인 브랜치.
  [akamotaco/gufo_win](https://github.com/akamotaco/gufo_win) `main`으로 게시되어 있으므로 **rebase하지 않는다**(게시된 기록을 다시 쓰게 된다).
  업데이트: `git merge origin/feat/gemma`. 본가(gufo-org) 커밋이 feat/gemma보다 먼저 필요하면 `git merge upstream/main`도 된다(2026-10-18 `1f970923` 사례).
  게시: `git push publish win/gemma:main`(fast-forward만, 강제 푸시 금지).
- Windows 포팅 원본은 `win/main`(pixmaate/gufo). 2026-10-03 시점 `d9a84f13`(gufo-org main)에서 갈라졌고,
  그 뒤 본가가 187커밋 진행했다. 그래서 pixmaate 커밋은 거의 모두 충돌한다.
- 원칙:
  - **Linux 동작은 바꾸지 않는다.** Windows 차이는 `compat/win32/`(POSIX 대체 헤더와 구현),
    `src/core/platform/`(소켓, 장치 메모리, 튜닝 스위치), `#ifdef _WIN32` 분기에만 둔다.
  - 본가 코드를 고치기보다 `compat/win32`에 POSIX 함수를 추가하는 쪽을 먼저 택한다. rebase 충돌이 줄어든다.
  - 충돌에서 본가가 같은 기능을 다르게 구현했다면 본가 쪽을 택한다(아래 4절 사례).

## 2. 툴체인 (이 장비)

| 항목 | 값 | 주의 |
| --- | --- | --- |
| ROCm | TheRock **10.2.0** (`rulith-inference\toolchain\rocm-venv\Lib\site-packages\_rocm_sdk_devel`) | pixmaate 검증은 **10.0.0**. 10.2 차이는 3절 |
| C/C++/HIP 컴파일러 | TheRock clang (`lib\llvm\bin\clang++.exe`), 링커 lld | MSVC 컴파일러는 쓰지 않고 STL·SDK만 쓴다 |
| MSVC | VS 2022 Build Tools 14.44, Windows SDK 10.0.26100 | `compat/win32/hip_include/cmath`는 14.5x용 우회. 14.44에서도 문제없었다 |
| vcpkg | `..\tool\vcpkg`, triplet x64-windows | icu curl openssl libpng libjpeg-turbo **libwebp pkgconf** (뒤 둘은 본가가 2026-10에 추가한 의존성) |
| CMake, Ninja | VS 내장 CMake, rulith의 ninja.exe | `build.ps1`이 vcvars64를 불러 PATH를 만든다 |

`build.ps1`은 `-Rocm -Vcpkg -Ninja`로 경로를 받는다. `..\tool\gufo_update.ps1`이 이 장비 경로를 넘긴다.

## 3. 겪은 문제와 해결

### 3.1 본가의 새 POSIX 코드 (포팅 이후 추가된 것)

pixmaate 포팅 이후 본가가 추가한 코드가 Windows 호환 계층에 없는 함수를 썼다.

| 위치 | 쓰는 것 | 해결 |
| --- | --- | --- |
| `src/cache/disk.cpp` (새 디스크 캐시) | `mkdirat`, `fdopendir/readdir/closedir`, `syscall(SYS_renameat2, ..., RENAME_NOREPLACE)`, `<linux/fs.h>`, `dev_t/ino_t` | `compat/win32`에 구현 추가. `renameat2`는 `FileRenameInfoEx`에서 REPLACE 플래그를 빼서 구현, `readdir`는 `FindFirstFileExW` |
| `src/cli/serve/component_text_cache.cpp` | `getrandom` (`<sys/random.h>`) | `BCryptGenRandom`. `gufo_posix_compat`에 `bcrypt` 링크 추가 |
| `src/cli/serve/serve.cpp` | `SIGBUS` (치명 시그널 리포터) | `signal.h`에 `SIGBUS 7` 정의, `sigaction`에서 "전달되지 않는 시그널"로 처리. UCRT `signal()`에 모르는 번호를 넘기면 invalid parameter handler가 돌아 프로세스가 죽을 수 있다 |
| `src/cli/serve/http_server.cpp` `LoadingResponder` | 생 `::setsockopt`, `::close`, `::recv`, `pollfd{.fd = int}`, `::accept` | `gufo::platform::SetSocketOption / CloseSocket / SocketRecv / CloseConnection`, `static_cast<decltype(pollfd::fd)>(fd)`, `static_cast<int>(::accept(...))` |
| `src/cli/serve/websocket.cpp` | `getsockopt(SO_RCVTIMEO)`를 `timeval`로 읽음 | `platform::GetSocketTimeout` 추가 (Winsock은 DWORD 밀리초) |
| `src/core/hip/committed_backing.cpp` (본가 `1f970923`, #633) | io_uring(`<linux/io_uring.h>`, `SYS_io_uring_setup/register`)으로 캐시 페이지를 long-term pin | Windows에선 `PinPages`가 -1(고정 안 함)을 반환하도록 `#ifdef _WIN32`. 원래 실패 시 기존 등록 경로로 돌아가는 선택 기능이고, Windows에선 `hipHostRegister`가 페이지를 고정한다 |
| `src/core/diagnostics/linux_sysfs.cpp` | `proc_root_`를 `std::filesystem::absolute("/proc")`로 바꿈(본가 `76661e30`) | Windows에선 `C:\proc`가 되어 포팅의 `proc_root_ == "/proc"` 대체 경로가 죽었다 → `gufo diagnose`가 CPU·메모리 0 표시. 비교 대상을 `std::filesystem::absolute("/proc")`로 바꿔 해결 |

**업데이트 때마다 할 점검** (새 POSIX 헤더가 들어왔는지):

```sh
grep -rhoE '#include <(sys/[a-z_/]+|linux/[a-z_/]+|[a-z]+)\.h>' src | sort -u   # 결과를 compat/win32/include 목록과 비교
grep -rn '::setsockopt\|::getsockopt\|::close(\|::recv(\|SO_RCVTIMEO\|timeval' src/cli/serve   # 소켓 코드
grep -rn 'syscall(SYS_\|/proc/\|/sys/\|io_uring' src                           # Linux 전용 경로·기능
```

`immintrin.h`, `cpuid.h`, `png.h`, `jpeglib.h`, `windows.h`는 목록에 없어도 괜찮다(clang 내장이거나 vcpkg 제공).

### 3.2 Winsock 차이 (소켓 코드를 고칠 때)

- `setsockopt`의 값 인자가 `const char*`다. `platform::SetSocketOption(fd, level, name, value)`를 쓴다.
- `SO_RCVTIMEO`/`SO_SNDTIMEO`는 `timeval`이 아니라 **DWORD 밀리초**다. timeval을 넘기면 조용히 tv_sec 밀리초가 된다.
- 소켓은 `int`로 다루지만 `pollfd::fd`는 `SOCKET`(64비트 unsigned)이라 중괄호 초기화에서 narrowing 오류가 난다.
- 소켓은 `close()`가 아니라 `closesocket()`. 받은 연결은 `CloseConnection`(Windows는 미전송 데이터를 리셋으로 버리므로 drain 후 닫는다).
- **listener에 `O_NONBLOCK`을 걸지 않는다.** Winsock에서는 accept된 소켓이 non-blocking을 상속하고, 연결 처리 코드는 blocking 읽기를 전제로 한다.
  본가는 device-loss 감시를 위해 listener를 항상 non-blocking으로 바꾸므로, 그 블록은 `#ifndef _WIN32`로 감쌌다. Windows는 100ms `poll` 조각으로 대신한다.
- `SO_REUSEADDR`은 Windows에서 다른 프로세스가 포트를 가로챌 수 있게 한다 → `SO_EXCLUSIVEADDRUSE`.
- Windows의 `IPV6_V6ONLY` 기본값은 1이다. 본가의 `::` dual-stack 설정이 Windows에서도 필요하다.

### 3.3 TheRock 10.2 관련

- **장치 코드 LTO 크래시**: `src/models/deepseek_v4_flash/kernels/rocm/backend.hip.cpp` 링크 중 ld.lld가
  "Live DEBUG_VALUE analysis" 패스에서 접근 위반(0xC0000005)으로 죽는다(`moe_down_q2K_hotlist_wmma_wide_kernel`).
  RelWithDebInfo의 `-g` 때문. Windows에서만 이 파일에 `-g0`을 준다(`src/models/deepseek_v4_flash/CMakeLists.txt`).
  같은 증상이 다른 HIP 파일에 생기면 같은 방법을 쓴다. 연산 결과는 바뀌지 않는다.
- **새 런타임 DLL**: hipBLASLt가 `libtensilelite-host.dll`에 의존한다(10.0에는 없음). 빠지면 `gufo.exe`가
  `0xC0000135`(STATUS_DLL_NOT_FOUND, exit -1073741515)로 아무 출력 없이 끝난다. `build.ps1`의 `$rocmDlls`에 추가했다.
  누락 DLL은 `llvm-readobj --coff-imports`로 재귀 확인한다(`..\tool\gufo_update.ps1`의 Check 단계).
- `build.ps1`의 "validated on 10.0.0" 경고는 정상이다. `qwen_ssm_ops_test` 등 bit-exact 테스트는 새 clang에서 다르게 반올림할 수 있다(WINDOWS.md).

### 3.4 git·셸 함정 (자동화할 때 특히)

- **pixmaate 커밋 제목 끝에 CR(`\r`)이 있다.** cherry-pick 충돌 표시가 `>>>>>>> sha (subject\r)`가 된다.
  Python `splitlines()`나 일부 에디터는 CR을 줄바꿈으로 보고 `)`만 있는 줄을 코드에 남긴다. 충돌 해결 도구는 `\n`으로만 나눠야 한다.
  해결 후 `grep -n '^)\r\?$'`로 확인한다.
- `core.autocrlf=true`(전역 설정)라 작업 트리 파일이 CRLF다. 문자열 정확 일치로 패치하는 스크립트는 CRLF를 고려해야 한다.
- Windows PowerShell 5.1:
  - `*>`로 리다이렉트하면 UTF-16과 UTF-8이 섞인 로그가 나온다. `2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 파일`을 쓴다.
  - BOM 없는 `.ps1`은 ANSI로 읽힌다. 스크립트는 ASCII로만 쓴다.
  - 이름은 대소문자를 구분하지 않는다. `function Git { git ... }`은 자기 자신을 무한 재귀한다.
  - 네이티브 인자 `-msse4.2`는 `.2`가 분리된다. 따옴표로 감싼다.
  - 스크립트 안의 `git log`는 페이저에서 멈춘다. `$env:GIT_PAGER = "cat"`.
- 이 장비에서 `python`은 PATH에 없다(Store 별칭). `C:\ProgramData\miniforge3\python.exe`를 쓴다.

## 4. 충돌 해결에서 내린 결정 (rebase 때 다시 만날 수 있음)

| 파일 | 결정 |
| --- | --- |
| `CMakeLists.txt` | 본가 ccache 블록 + Windows 블록 둘 다. kernel-bench는 본가 `GUFO_REVISION` + Windows의 roctx 선택 링크 |
| `src/models/qwen38_flash_next/ngram.cpp` | 본가가 이미 모든 플랫폼에서 128 readers로 올렸으므로 본가 쪽(포팅의 `kWindowsWorkers` 분기 불필요) |
| `src/cli/serve/inference_backend.cpp` 프롬프트 체크포인트 | 본가 `prompt->stable_prefix_tokens`(#281) 유지. 포팅의 `prompt_checkpoint` 스위치는 남아 있지만 쓰이지 않는다 |
| `src/models/qwen/chat_template.hpp` | 본가 `RenderedPromptBoundBytes`(#288) 유지. 포팅의 8MiB 고정값과 `require_tool_call`은 버림 |
| `src/cli/serve/openai_chat.cpp` | `top_logprobs: 0` 허용만 가져옴. `response_format: text`는 본가가 구조화 출력으로 처리하므로 버림 |
| `docs/WINDOWS.md` "Other text models" | 해당 커밋(27B, A3B)을 넣지 않았으므로 제외 |

## 5. 열린 항목

- [x] `gufo diagnose`의 CPU·메모리 0 표시 (3.1 마지막 줄). `proc_root_ == std::filesystem::absolute("/proc")`로 비교하도록 고쳤다(2026-10-18).
  서버 메모리 예산은 `HostAvailableBytes()`(`sysconf(_SC_AVPHYS_PAGES)`)를 쓰므로 영향 없음(로그 `host_available_mib` 정상).
- [ ] Flash-Next 디코드 Windows 튜닝 이식: `win/main`의 `ae594013`, `b0b5d8a9`, `a8b45a0b`, `aeee89a5`.
  `ae594013`만 56곳 충돌(engine.cpp 14, executor.cpp 13, prompt_lookup.hpp 11, …). `prompt_lookup.hpp`는 본가 #613 구현이 대체하므로
  `--prompt-lookup` 부분은 버리고 `tuning.hpp` 스위치(copy_kernels, recorded_rollback, keep_rollback_rows, flush_before_wait,
  flag_waits, fast_sampling, verify_graph_candidates, fused_hc_down, hot_first_upload)만 옮기는 방향을 권한다.
  검증: `bench --logit-eval`로 스위치 on/off bit-identical 확인(WINDOWS.md "Windows-validated switches").
- [ ] **GPU 메모리 넘침 방지**: `src/core/platform/device_memory.cpp`의 Windows `DeviceMemoryInfo`는 여유 공간을
  "전용(LOCAL) 여유 + 공유(NON_LOCAL) 여유"로 돌려준다. 그래서 gufo는 공유 메모리(시스템 RAM)까지 용량으로 보고
  carve-out을 넘겨 할당한다. 2026-10-18 측정: Flash-Next 2세션×262K + MTP + 비전이 전용 95.3GB(한계)에 붙었고,
  요청이 시작되자 페이지 파일에 약 20MB를 썼다(Pages Output 1,396/s). 1세션×262K는 91.4GB에서 쓰기 0.
  Rulith는 `manager.py` `host_expert_layers()`에서 "carve×0.96 − 다른 프로그램 사용량"을 넘는 부분을 고정 호스트
  메모리로 직접 옮겨 이 문제를 피했다. 지금은 `..	ool\gufo_select.py`의 사전 점검(합계 93.4GB)으로만 막는다.
  후보: LOCAL 여유만 보고하도록 바꾸기(체크포인트 용량·세션 판단에 미치는 영향 확인 필요).
- [ ] Qwen3.8-27B Windows 튜닝 14커밋, Qwen3.6-A3B 4커밋 (목록은 `..\gufo-windows.md`).
- [ ] 테스트 미포팅: `BUILD_TESTING=OFF`로만 빌드했다. 본가가 새로 쓴 테스트 코드(`tests/cli/http_server_test.cpp` 2170행 이후 등)에
  생 `::close`/`timeval` 소켓 호출이 남아 있어 `gpu-test` 프리셋은 아직 빌드되지 않을 것이다.
- [ ] **런타임 미검증 경로**: 3.1의 디스크 캐시 함수(`readdir`, `renameat2`, `mkdirat`)는 컴파일만 확인했다.
  `--cache-disk DIR`를 켜고 체크포인트 저장·복원을 실제로 시험해야 한다(기본은 `disk_cache=off`).
- [ ] Qwen3.8-Flash-Next를 이 Windows 빌드로 아직 돌려보지 않았다(GPU 약 89GiB, rulith 서버를 내려야 함 — 사용자 동의 필요).

## 6. 업데이트 후 확인 순서

1. `..\gufo_update.bat` (merge → 빌드 → DLL 확인 → `gufo diagnose` PASS). 확인이 끝나면 `..\gufo_update.bat publish`
2. 3.1의 grep 점검으로 새 POSIX 사용 확인
3. Gemma급 모델 하나로 기동·요청 확인. 예:
   `gufo serve --host 127.0.0.1 --port 8093 --sessions 1 llm --model <gguf> --context 32768 --speculative mtp --mtp-model <mtp gguf> --cache-ram-bytes 2147483648`
   요청 본문에 `model`이 필수이며 값은 `/v1/models`의 `id`와 같아야 한다.
4. `--cache-ram-bytes`를 항상 지정한다. 지정하지 않으면 "남은 RAM의 절반"(이 장비 측정 8.1GB)을 pin한다.
   사용자의 1차 목표는 Windows RAM 100% 사용과 그로 인한 SSD 페이징 쓰기를 막는 것이다.
5. 모델 로딩 규칙(사용자 합의): Gemma 26B/31B급 하나는 테스트로 올려도 되지만, Qwen3.8-Flash-Next처럼 큰 모델은 먼저 묻는다.
