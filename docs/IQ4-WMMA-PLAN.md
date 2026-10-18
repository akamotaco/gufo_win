# IQ4_NL / IQ4_XS 전문가를 routed F16 WMMA prefill 경로에 넣기 — 작업 명세

> **완료(2026-10-18).** 구현은 4절 권장안과 달리 더 단순한 방법을 썼다: IQ4 블록을 가중치 읽기 단계(`fetch_stage`)에서
> `kvalues_iq4nl`(int8) 32개 + 스케일(d, IQ4_XS는 d×(ls−32))로 펼쳐 **Q8_0 경로(2바이트 정렬 읽기, LDS·풀기·WMMA)를 그대로 재사용**
> (`Iq4Lookup`: `__builtin_amdgcn_perm` 2회 + 비트3 선택). 재포장·메모리 증가 없음. 넓은 배치(≥1024 토큰)에서 IQ4_XS는
> 묶음(paired) 커널 대신 64토큰 타일 지도로 gate/up을 따로 처리.
> 결과: 테스트 IQ4_XS/IQ4_NL 6경우 통과(|F16−F64|가 MMQ보다 약 10배 작음). AtomicChat prefill 2.1K/8.5K/31.8K/160K =
> 1,327/1,316/1,338/1,241 tok/s(이전 876/940/1,008/962, rulith 1,048/1,151/1,222/1,166). UD-Q4_K_XL 1,477/1,469/1,496(회귀 없음).
> GPU 86.3 GiB 그대로. 아래는 작업 전 작성한 명세 원문.

작성 2026-10-18. 대상 브랜치 `win/gemma`(akamotaco/gufo_win). 이 문서만 읽고 구현을 시작할 수 있도록 쓴다.
포팅 전반은 [WINDOWS-PORTING-NOTES.md](WINDOWS-PORTING-NOTES.md) 참고.

## 1. 목표와 배경

AtomicChat abliterated Qwen3.8-Flash-Next(`AD-4.85bpw-Q5_K_M`, 42 분할, 99.9 GiB)의 prefill을 빠른 경로로 올린다.

| 측정(2026-10-18, 사고 끔) | 2.1K | 8.5K | 31.8K | 160K |
| --- | ---: | ---: | ---: | ---: |
| gufo, UD-Q4_K_XL(전부 WMMA) | 1,491 | 1,466 | 1,497 | 1,387 |
| gufo, AtomicChat 현재(12층만 WMMA) | 973 | 1,045 | 1,094 | – |
| rulith(llama.cpp), AtomicChat | 1,048 | 1,151 | 1,222 | 1,166 |

AtomicChat 전문가 형식: 36층 gate/up **IQ4_XS** + down **IQ4_NL**, 12층 gate/up Q5_K + down Q5_0.
12층은 `35aaee06`에서 Q5_0을 업로드 시 Q5_1로 바꿔 이미 WMMA 경로. **이 작업은 나머지 36층.**
목표: AtomicChat prefill ≥ 1,300 tok/s(31.8K), UD-Q4_K_XL 회귀 없음(±3%).

## 2. 현재 구조 (파일·심볼)

- `src/models/qwen38_flash_next/kernels/rocm/executor.cpp`
  - `Executor::WmmaExperts()`: gate/up ∈ {Q4_K, Q5_K} 이고 down ∈ {Q5_1, Q8_0} 일 때만 true. **여기에 IQ4_XS gate/up, IQ4_NL down을 추가해야 36층이 들어온다.**
  - `Executor::MoeExperts()`: WMMA 경로면 `RoutedCompact`(전문가별 16행 버킷) → 활성값 F16 변환 → `RoutedF16Gemm`(gate, up; up은 SwiGLU를 적용해 F16으로 down 입력을 씀) → `RoutedF16Gemm`(down). `gate_type/down_type`을 `WeightType`으로 바꾸는 부분(현재 Q5_1 기본값)도 확장 필요.
  - 아니면 `GatedExperts()`/`Experts()` → `qfn_mmq_*_moe_raw`(llama.cpp 정수 MMQ, `mmq/`) — AtomicChat 36층이 지금 타는 경로.
- `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`
  - `RoutedF16GEMMKernel<WeightType, BM, BN, BK, kPair>`(약 4960행): 256 스레드, 8 wave, `BK == 2`(한 stage = K 64 = 32원소 블록 2개).
    가중치 코드는 LDS에 **패킹된 채** 두고, wave가 자기 16행만 F16으로 풀어 WMMA(F16, F32 누적)로 K 전체를 누적.
  - 푸는 공식: 코드 바이트를 `__builtin_amdgcn_perm`으로 `0x64xx`(1024+q)에 넣고 1024를 빼서 F16 q를 만든 뒤 `w = q*scale + bias`
    (`CodesToHalves`, magic 1024, 부호 있는 바이트는 q+128로 실어 magic 1152). scale/bias는 (행, K블록)마다 half2로 staging.
  - 형식별 staging: Q4_K/Q5_K는 32바이트 니블 그룹 + 헤더, **Q5_1은 "한 행의 BK블록(24B×2=48B)이 16바이트 경계에 맞는다"는 전제로 블록을 통째로 16B 청크 로드**, Q8_0은 블록 2개 = 4청크.
  - `RoutedF16RowBytes<kType>`: 행 바이트 수. `RoutedF16Gemm()`(약 6188행): 형식/타일 분기, `k % block_elems` 검사(K-quant 256, 나머지 64).
  - `kernels.hpp` `enum class WeightType`: F32, F16, Q5_1(7), Q8_0(8), Q4_K(12), Q5_K(13), BF16(30). **IQ4_NL(20), IQ4_XS(23) 추가.**
- 업로드: `kernels/rocm/device_model.cpp` `Uploader` — `Copy()`, `CopyPleConv()`(BF16→F32), `CopyQ5_0AsQ5_1()`(Q5_0→Q5_1). 재포장이 필요하면 여기에.

## 3. 형식 (ggml, `mmq/ggml-common.h`)

```
kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113}

block_iq4_nl (18 B, 32원소):  half d; uint8 qs[16];
  x[j]    = d * kvalues[qs[j] & 15]      j = 0..15
  x[j+16] = d * kvalues[qs[j] >> 4]

block_iq4_xs (136 B, 256원소 = 32원소 하위블록 8개):
  half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128];
  하위블록 ib(0..7): ls = ((scales_l[ib/2] >> 4*(ib%2)) & 15) | (((scales_h >> 2*ib) & 3) << 4)
                     dl = d * (ls - 32)
                     qs[16*ib .. 16*ib+15]가 IQ4_NL과 같은 니블 배치: x[32*ib + j] = dl*kvalues[lo], x[32*ib+j+16] = dl*kvalues[hi]
```
CPU 참조: `src/core/quant/ggml_dequant.cpp` `DequantizeIQ4_NL`, `DequantizeIQ4_XS`.

## 4. 설계 방향 (권장안과 제약)

**값 풀기(두 형식 공통):** 니블 → `kvalues[q] + 128`(0..240, 부호 없는 바이트)을 레지스터의 16바이트 표에서 `__builtin_amdgcn_perm` 두 번
(하위 8바이트/상위 8바이트) + 비트3 선택으로 만들고, 기존 `CodesToHalves(magic=1152, scale=dl, bias=0)` 재사용.
`dl`은 IQ4_NL은 d, IQ4_XS는 d*(ls−32)(half로 반올림 — **이 반올림이 CPU 참조와 어긋나지 않는지 테스트로 확인**; 어긋나면 F32로 scale 적용 검토).

**정렬 제약(핵심 난점):**
- IQ4_NL 18B 블록 2개 = 36B → 16B 경계 전제가 깨진다. 선택지:
  (a) 커널에서 dword/바이트 단위 로드로 18B 블록을 직접 읽기(메모리 증가 없음, 코드 복잡).
  (b) 업로드 시 20B `{d, pad, qs16}`로 재포장(+11%, AtomicChat 추정 +1.9 GiB) 후 Q5_1식 staging 변형.
  24B 재포장(Q5_1 레이아웃 재사용)은 +33%(추정 +5.7 GiB)로 **GPU 약 92 GiB → 안전선(어댑터 전용 93.4 GB, 데스크톱 약 1.3 GB 포함) 초과 위험, 금지.**
- IQ4_XS 136B: `qs`가 오프셋 8 → 하위블록 쌍(32B)이 8바이트 정렬. 8B 로드로 처리하거나 업로드 시 144B로 패딩(+5.9%, 추정 +1.3 GiB).
- **메모리 상한:** AtomicChat + MTP + 비전 현재 86.3 GiB(gufo 보고값). 변경 후 **gufo 사용량 ≤ 90 GiB**를 지킨다
  (Flash-Next 1세션 262K 기준, `gufo_select` 사전 점검이 모델 87 GiB 추정 + 기타로 막는다 — 추정치도 함께 갱신).

**LDS/레지스터:** 커널은 stage 11–12 KB, WGP당 5블록 상주를 노린다. 새 형식이 이 예산을 넘기면 기존 Q4_K 속도도 떨어질 수 있으니
`kType`별 `constexpr`로 분리하고 **기존 형식의 코드 경로는 바꾸지 않는다.**

**gate/up 묶음(kPair):** 지금은 Q4_K만. IQ4_XS는 우선 묶지 않고(separate) 정확성과 속도를 먼저 확인, 효과가 있으면 다음 단계.

## 5. 단계와 완료 기준

| 단계 | 내용 | 완료 기준 |
| --- | --- | --- |
| S0 | 검증 장치: `routed_wmma_ops_test`, `iq_experts_ops_test`를 Windows에서 빌드·실행 | 기존 형식 전부 통과(상태는 아래 6절) |
| S1 | 테스트에 IQ4_NL/IQ4_XS 생성기(`MakeIQ4NL/MakeIQ4XS`) + WMMA 결과를 ① `qfn_mmq_iq4_*_moe_raw`(정수 MMQ) ② CPU 배정밀도 참조와 비교 | 새 형식은 아직 실패(미구현)로 나오되 장치는 동작 |
| S2 | `WeightType::kIQ4_NL` + 커널 staging/풀기 + `RoutedF16RowBytes` + `RoutedF16Gemm` 분기 (down) | S1 테스트의 IQ4_NL 통과(기존 형식과 같은 허용 오차) |
| S3 | `WeightType::kIQ4_XS` (gate/up, separate) | IQ4_XS 통과 |
| S4 | `WmmaExperts()`/`MoeExperts()` 분기에 연결 | AtomicChat 36층 WMMA 진입(로그·속도로 확인) |
| S5 | 종단 확인 | 아래 |

S5 종단 기준:
- AtomicChat: 계산(17×23=391, 391/17=23), 한국어(서울·경복궁), 코드, 이미지 설명이 정상. `bench.py` prefill 31.8K ≥ 1,300.
- UD-Q4_K_XL 회귀: prefill 2.1K/8.5K/31.8K가 1,491/1,466/1,497 대비 ±3% 이내, 출력 정상.
- 메모리: gufo 보고 GPU ≤ 90 GiB, 감시 시 페이지 파일 사용량 증가 없음.

## 6. 빌드·테스트 방법 (이 장비)

```powershell
# 테스트 대상만 (build\gpu-test, 서버용 build\release와 별개)
$llm = "C:\Users\AMD-MAX395-128G-2T\Desktop\ai\llm"
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test `
  -Target qwen38_flash_next_routed_wmma_ops_test `
  -Rocm "$llm\rulith-inference\toolchain\rocm-venv\Lib\site-packages\_rocm_sdk_devel" `
  -Vcpkg "$llm\tool\vcpkg" -Ninja "$llm\rulith-inference\toolchain\ninja.exe" -Jobs 8 `
  -CMakeArgs "-DPython3_EXECUTABLE=C:/ProgramData/miniforge3/python.exe"   # 테스트 설정은 Python3 필요(PATH에 없음)
build\gpu-test\tests\models\qwen38_flash_next\qwen38_flash_next_routed_wmma_ops_test.exe
# 서버 빌드·기동: ..\gufo_update.bat nopull, ..\gufo_select.bat (8080: 2=AtomicChat, 1=UD-Q4_K_XL)
```
테스트 실행에 필요한 DLL은 `build\release`에 복사된 것과 같다(없으면 `build\gpu-test`에 같은 DLL을 복사).
GPU는 서버와 공유하므로 테스트는 작은 크기로 돌리고, 큰 측정은 서버를 내린 뒤 한다.

**S0 상태: 완료(2026-10-18).** 두 테스트를 위 명령(gpu-test, `-CMakeArgs` Python 지정)으로 빌드하고 실행:
`PATH`에 `build\gpu-test`와 `build
elease`를 앞에 붙여 실행(DLL은 빌드 폴더 최상위에 복사됨).
- `qwen38_flash_next_iq_experts_ops_test`: IQ3_S/IQ4_XS/IQ4_NL 정수 MMQ 12개 경우 PASS(배정밀도 대비 rel_error 0.0032–0.0043).
- `qwen38_flash_next_routed_wmma_ops_test`: Q4_K/Q5_K/Q5_1/Q8_0의 F16 WMMA를 MMQ·F64 참조와 비교, 모두 통과.
  **F16 WMMA가 MMQ보다 F64에 약 10배 가깝다**(예: Q4_K m=640 k=2560: |F16−F64| 1.33, |MMQ−F64| 12.46).
  S1은 이 테스트의 형식 목록(약 232행 생성기, 278행 MMQ 함수 선택, 536행 형식 목록)에 IQ4_NL/IQ4_XS를 더하는 것으로 시작한다.

## 7. 하지 말 것

- 기존 형식(Q4_K/Q5_K/Q5_1/Q8_0)의 커널 경로·타일 크기 변경.
- 24B 재포장처럼 GPU 메모리를 크게 늘리는 방식.
- 테스트 없이 커밋. 빌드만 통과한 커널 변경은 커밋하지 않는다(`ad680fc3`은 Q5_0 누락이 실행으로만 드러났다).
- `win/gemma`는 게시된 브랜치: rebase·강제 푸시 금지(merge만).

---

## 다음 단계(선택): IQ4_XS gate/up 묶음(paired) 처리 — 명세

상태: **미착수.** 측정상 기대 효과 약 +5~6%(AtomicChat prefill 약 1,330 → 약 1,400 tok/s 추정).

### 근거 측정 (2026-10-18, UD-Q4_K_XL, 같은 빌드에서 묶음만 끔/켬)

| 프롬프트 | 묶음 끔(gate/up 따로, 48토큰 타일) | 묶음 켬(`RoutedGatedF16Gemm`, 128토큰) | 차이 |
| --- | ---: | ---: | ---: |
| 2.1K | 1,383 | 1,430 | +3.4%(편차 수준) |
| 8.5K | 1,367 | 1,455 | +6.4% |
| 31.8K | 1,406 | 1,494 | +6.3% |

측정용 임시 코드(`GUFO_EXP_NO_PAIR`, 48토큰 지도 추가)는 되돌렸다. 재측정이 필요하면 같은 방식으로 넣는다.

### 현재 동작
- 토큰 ≥ 1024이고 `routed_tile_rows_ == 48`이면 타일 지도는 64토큰(down)과 128토큰(묶음)만 만든다(`Executor::RouteHints` 근처의 `append`).
- `Executor::MoeExperts()`: `paired = wide_map && gate_type != kIQ4_XS`. IQ4_XS는 64토큰 지도로 gate → up(SwiGLU, F16 출력) 두 번.
- 묶음 커널 = `RoutedF16GEMMKernel<..., kPair=true>`(BM=128: gate 64행 + up 64행, BN=128). `RoutedGatedF16Gemm`은 Q4_K/Q5_K만 분기.

### 묶음 커널의 구조(바꿔야 할 부분)
- 가중치 코드가 **LDS를 거치지 않고 레지스터**에 있다: wave 반쪽(`half_id`)마다 행의 16바이트 청크 하나(`f_c = half_id`).
- Q4_K 배치에서 32바이트 니블 그룹 = K32 블록 2개. 청크 c(16B)의 **하위 니블 = 블록 A의 원소 16c..16c+15, 상위 니블 = 블록 B의 같은 원소** →
  `compute_stage`의 kPair 분기는 `nib = (words[i] >> (4*kb)) & 0x0F0F0F0F`로 두 K32 블록의 K16 조각을 같은 레지스터에서 꺼낸다.
- 슈퍼블록 첫 stage(`code_group == 0`)에 4개 니블 그룹을 `code_cache[u][4]`로 읽어 두고 4 stage에 걸쳐 쓴다. 스케일은 `own_scale`(half2 scale,bias)을 `XorLane(…,16)`으로 반대쪽 반과 교환.

### IQ4_XS용 설계(권장)
1. **fetch(슈퍼블록 첫 stage):** 136B 슈퍼블록에서 하위블록 쌍 (2s, 2s+1)의 qs 32바이트를 읽고, **Q4_K 니블 배치로 레지스터에서 재배열**한다:
   반쪽 h의 청크 바이트 j = `lo/hi(qs[2s][j'])` | `lo/hi(qs[2s+1][j']) << 4` 형태로, "하위 니블 = 블록 2s의 원소 16h+j, 상위 니블 = 블록 2s+1의 같은 원소"가 되게 만든다
   (IQ4_XS 하위블록의 원소 0..15는 qs 하위 니블, 16..31은 상위 니블). 그러면 compute의 `nib` 추출이 그대로 맞는다.
2. **스케일:** (행, 하위블록)마다 `half2(d*(ls-32), 0)`. Q4_K처럼 `own_scale`에 넣고 교환. bias는 0.
3. **compute:** kPair 분기에서 IQ4_XS면 `nib`을 `Iq4Lookup(nib) ^ 0x80808080`으로 바꾸고 magic을 1152로(Q8_0과 같은 부호 처리).
4. **실행 연결:** `RoutedGatedF16Gemm`에 `kIQ4_XS` 분기, `MoeExperts()`의 `paired` 조건에서 IQ4_XS 제외를 없앤다(64토큰 분리 경로는 남겨 둬도 됨).
5. **레지스터 예산:** 묶음 커널은 이미 빠듯하다(`code_cache`). 재배열은 fetch 단계에서 끝내 compute 레지스터를 늘리지 않는다. 늘면 Q4_K 묶음 속도 회귀를 반드시 재측정.

### 완료 기준
- `routed_wmma_ops_test`: 458행 근처의 묶음 검증(`tile_rows == 48 && (Q4_K || Q5_K)`)을 IQ4_XS까지 넓혀, 묶음 결과가 분리 결과와 같은지 확인
  (기존 Q4_K/Q5_K 묶음 검사는 바이트 비교 — IQ4_XS도 같은 기준을 목표로, 안 되면 허용 오차를 근거와 함께 명시).
- AtomicChat prefill 8.5K/31.8K가 현재(1,316/1,338) 대비 +4% 이상, 출력 정상.
- UD-Q4_K_XL 회귀 없음(1,477/1,469/1,496 ±3%), GPU 사용량 변화 없음.
