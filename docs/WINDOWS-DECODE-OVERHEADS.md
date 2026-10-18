# Flash-Next 디코드 on Windows — 어디서 시간이 가는가 (2026-10-18)

AtomicChat abliterated(Qwen3.8-Flash-Next `AD-4.85bpw-Q5_K_M`)를 기준으로, gufo 디코드가 rulith(llama.cpp 포크)보다
어디서 느리고 어디를 고쳐야 하는지 잰 기록이다. gfx1151(Strix Halo), TheRock 10.2, Windows 네이티브, 대화 1개.

## 1. 결론

| 프롬프트 | gufo MTP | rulith MTP | gufo MTP 끔 | rulith MTP 끔 |
| --- | ---: | ---: | ---: | ---: |
| 짧음(41) | 36.1 (적중 59%) | 40.7 (46%) | 29.8 | 28.4 |
| 2.1K | 36.6 (62%) | 43.1 (54%) | 29.1 | 27.8 |
| 8.5K | 47.4 (71%) | 50.4 (70%) | 29.3 | 27.7 |
| 31.8K | 38.5 (60%) | 45.5 (63%) | 29.0 | 27.6 |
| 159.9K | **42.1** (62%) | 34.2 (51%) | 28.1 | – |

디코드 tok/s. 같은 고정 문구, 사고 끔, temperature 0, 캐시 없음, 서버 하나씩 직렬(gufo `10ee14ac`).
측정 스크립트: `llm	ool\measure.py PORT LABEL`(저장소 밖). 출력 정확도는 `llm	ool\copytest.py PORT TAG [filler_k]`(식별자 40개 따라 쓰기).
같은 실행의 prefill은 gufo 1,381/1,396/1,441/1,338, rulith 1,041/1,163/1,218/1,155 tok/s.
MTP 켬 수치는 생성 내용에 따라 ±10% 흔들린다(문구를 매번 바꾸면 적중률이 42~76%로 움직인다).

- **커널은 gufo가 빠르다.** MTP 끔 디코드 +5%.
- **긴 컨텍스트는 gufo가 이미 앞선다**(160K 42.1 대 34.2).
- **짧은~중간 컨텍스트의 차이는 추측 사이클 비용이다.** 적중률은 gufo가 같거나 높지만 사이클당 추측 수가
  gufo 약 1.2개(340/290), rulith 약 3개(678/약 227, mean len 2.25). gufo의 길이 정책(`mtp_policy.hpp`)은
  비용 표(`mtp_costs.hpp`)와 실측 적중률로 기대 토큰 / 시간이 최대인 길이를 고르므로 정상이다 —
  **추측 한 개가 비싸서** 짧게 고른다.
- 이전에 "디코드가 9~17% 뒤진다"던 수치는 HC inject 버그(`92f088c6`) 수정 전 것이다. 그 버그는 MTP 적중률도 깎았다.

## 2. 추측 사이클 비용 (`qwen38_flash_next_gpu_probe --cost-audit 1`, 32K 깊이)

| 폭 | proposal | target(검증) | 합계 | `mtp_costs.hpp`(리눅스 9월) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 34.4 | 35.6 ms | 42.5 |
| 2 | 3.1 | 39.9 | 44.3 | 51.9 |
| 3 | 7.2 | 44.5 | 53.0 | 61.5 |
| 4 | 11.2 | 48.5 | 61.0 | 71.1 |
| 8 | 28.0 | 64.8 | 94.6 | 112.6 |

4K 깊이도 거의 같다. 표보다 전체가 약 7ms 싸지만 비율이 같아 표를 다시 맞춰도 길이 선택은 거의 안 바뀐다.

- 추측 1개의 한계 비용 ≈ **proposal 4.1ms + 검증 4.5ms**(rulith는 합쳐 약 6.6ms).
- 이 도구의 proposal은 **전체 어휘 헤드**(248,320행, 약 675MB ≈ 3ms) 기준이다(`SetDraftVocab`을 부르지 않음).
  서버의 축소 헤드로는 추측 한 단계가 약 2.3ms다(3절).
- 검증 한계 비용(4.5ms/토큰)은 대부분 토큰마다 더 읽는 전문가 가중치라 줄이기 어렵다.

재현(서버를 내린 상태 — 모델을 통째로 올린다, PATH에 `build\release`):

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test -Target qwen38_flash_next_gpu_probe `
  -Rocm <rocm> -Vcpkg <vcpkg> -Ninja <ninja> -Jobs 8          # 인자는 IQ4-WMMA-PLAN.md 6절과 같다
build\gpu-test\tests\models\qwen38_flash_next\qwen38_flash_next_gpu_probe.exe `
  --model <AtomicChat 00001-of-00042.gguf> --mtp-model <mtp-...-shared-Q8_0.gguf> --cost-audit 1
```

`proposal_ms`를 직접 보여 주므로 **디코드 최적화의 A/B 도구**로 쓴다. `--depth N`으로 깊이 하나만 잴 수 있다.

## 3. 사이클 계측 (서버 경로, `GUFO_DECODE_TRACE=1`)

`Session::DecodeStep`이 64사이클마다 구간 평균을 로그에 찍는다. 짧은 컨텍스트, 사이클당 1.5~1.6토큰:

| 구간 | ms/사이클 |
| --- | ---: |
| 추측(`PrepareDecode`: MTP 따라잡기 + 축소 헤드 추측) | 4.1~4.6 |
| 검증(`Forward`, GPU 완료까지 동기) | 37~38 |
| 수락·롤백(`FinishDecode`) | 0.6~0.9 |
| 서버의 사이클 사이 | 0.02 |

- **2절의 proposal 4.1ms/추측은 전체 어휘 헤드 기준이다.** `mtp_audit`는 `SetDraftVocab`을 부르지 않는다.
  서버(98,304행 축소 헤드)에서 추측 한 단계는 약 2.3ms로, 바이트 하한(약 1.8ms)과 0.5ms 차이뿐이다.
  → 드래프트 루프의 왕복을 줄여 얻을 몫은 디코드 1~2% 수준.
- 호스트·서버 쪽도 원인이 아니다(0.02 + 0.7ms).
- 사이클의 대부분은 검증이고, 추측 토큰 하나를 더 검증하면 층마다 전문가 10개를 더 읽는다(약 1GB, 약 4.5ms, 대역폭 한계).
- **길이 강제 실험**(정책의 최소 길이를 임시 코드로 올림, 같은 고정 문구): 최소 2개 35.7/39.2/47.0/39.7/41.9,
  최소 3개 32.0/36.7/46.0/36.7/41.5 tok/s(1절 기본 36.1/36.6/47.4/38.5/42.1). **정책은 맞게 고른다.**
- rulith는 3개씩 추측해 사이클당 약 1.25개가 맞고, gufo는 최소 3개 강제 시 약 1.05개다. 깊은 추측의 품질 차이인지,
  생성 내용·집계 방식 차이인지는 이 자료로 가릴 수 없다(엔진마다 생성되는 글이 다르다).

## 4. 다음 작업 (우선순위, 3절 반영)

1. **proposal 단계의 왕복 줄이기** — 3절대로 몫이 1~2%라 우선순위를 낮춘다. 드래프트 단계마다 토큰을 pin 메모리로 받고 sync한다
   (`Executor::MtpHead`/`MtpBody`, 엔진의 드래프트 루프). 후보: pixmaate `a8b45a0b`("fewer host round trips per
   MTP cycle")의 해당 부분, 드래프트 루프의 `flush_before_wait`(5절), 드래프트 단계를 한 번의 제출로 묶기.
   2ms 줄이면 한계 비용이 rulith 수준이 되고, 정책이 스스로 더 길게 추측한다. 효과는 `proposal_ms`로 판정.
2. **low-rank 헤드**(rulith `apply_mtp_lowrank_head`: 2560→512 사전 점수로 후보 512행만 헤드 곱). 헤드 약 1.3ms → 수백 µs.
   rulith 드래프터(`models\shared\mtp-...-Q4_K_M-head-lr512.gguf`)에 텐서가 있지만 gufo는 그 파일의 Q4_K hc 텐서를 거부한다.
3. 작은 배치(2~8토큰) IQ4 전문가 커널의 효율 확인 — 검증 한계 비용 쪽. 검증이 사이클의 90%라 레버는 여기다.
   rulith에는 이 구간 패치가 여러 개 있다(`apply_small_batch_decode`, `apply_verify_small_products`,
   `apply_mmvq_rows4_058`, `apply_topk_moe_spec_058` — 라우터 커널 8.45 → 6.63µs/층).
4. 같은 글로 깊은 추측의 적중률 비교: 한 엔진의 출력을 고정하고 다른 엔진에 강제로 먹이며 MTP 제안을 비교해야 한다.

Flash-Next MTP는 `--min-draft-tokens`/`--draft-policy`를 막아 둔다(`src/cli/serve/inference_backend.cpp`의
"Flash-Next MTP requires a positive draft limit"). 길이를 강제하는 실험에는 코드 변경이 필요하다.

## 5. Windows HIP 동작 마이크로 벤치 (참고)

pixmaate/gufo `win/main`의 Flash-Next 디코드 튜닝 3커밋(`ae594013`, `a8b45a0b`, `aeee89a5`, 합쳐 +8~15% 주장, TheRock 10.0
기준)의 전제를 단독 HIP 벤치로 다시 쟀다(pi 에이전트, 벤치 소스는 오류 검사가 없어 저장소에 넣지 않았다).

| 주장 | 실측 | 판정 |
| --- | --- | --- |
| Windows는 flush할 때만 큐를 넘긴다 → `flush_before_wait` | 그래프(약 0.9ms 작업) 실행 후 호스트가 쉬면 sync 대기 0.371ms, 실행 직후 `hipStreamQuery` 한 번이면 0.004ms | **확인.** 이득은 "제출 → 호스트가 다른 일 → 대기" 구간에만 있다 |
| 큰 그래프 뒤 같은 스트림 제출이 막힌다 → 두 스트림 전환(`aeee89a5`) | 1600노드 뒤 같은 스트림 0.062ms, 다른 스트림 0.049ms | 기각(약 13µs) |
| `hipStreamSynchronize`가 늦게 뜬다 | 400노드 작업 대기 sync 0.910 / query+sync 0.903 / query 폴링 0.907ms | 기각 |
| 카피 엔진 전환 비용 → `copy_kernels` | 2KB D2D 24회: memcpy와 커널 모두 0.131ms | 기각. pin 호스트→GPU만 회당 약 9µs 이득 |
| 롤백 카피를 그래프로(`recorded_rollback`) | 작은 D2D 카피 회당 약 1.8µs | 효과 0.1ms 안팎 |
| pin 완료 플래그(`flag_waits`) | flush 없이 폴링하면 커널의 pin 쓰기가 sync 전까지 안 보임. D2H 카피 + `hipStreamQuery`면 0.092ms에 보임 | **미결.** 실패 원인이 flush 부재일 수 있어 flush와 함께 다시 재야 한다 |

- 그래프 launch+sync는 노드당 약 1.0µs(1노드 0.041ms, 1600노드 1.665ms). 대부분 GPU 쪽 커널당 비용이라
  커널을 합쳐야 줄어든다.
- `hipMemGetInfo`는 서버가 88GiB를 써도 107.7/107.9GiB 여유로 보고한다(프로세스별 전망). 전용 메모리 판단에 쓰면 안 된다.
- `std::this_thread::sleep_for(5ms)`는 실제로 약 15.6ms 잔다(Windows 타이머 해상도).
- 기본 `hipHostMalloc` 메모리에 커널이 직접 쓰면 프로세스가 죽는다. `hipHostMallocMapped`는 접근은 된다.

## 6. 하지 말 것

- 두 스트림 전환(`aeee89a5`) 이식 — 실측 약 10µs.
- 롤백을 recorded graph로 바꾸기 — 카피 회당 1.8µs.
- D2D `hipMemcpyAsync`를 막연히 커널로 바꾸기 — 이득 0.
- 효과를 잴 지점 없이 `flush_before_wait`를 그래프 실행 뒤 곳곳에 넣기 — 바로 대기하는 곳에서는 이득이 0이다.
  드래프트 루프처럼 `proposal_ms`로 판정할 수 있는 곳부터.
