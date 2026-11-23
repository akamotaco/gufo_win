#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_EXPERT_STORE_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_EXPERT_STORE_H_

#include <cstdint>

#include <hip/hip_runtime_api.h>

#include <gufo/ext/experimental/expert_stream.hpp>

struct ds41_expert_store;
struct ds41_model;
struct ds41_weights;

/* The statistics and overlap view of Gufo's experimental expert store
 * (gufo/ext/experimental/expert_stream.hpp). Each expert has three pieces,
 * so an overlap's table holds the gate, up and down tables, in that order. */
using ds41_expert_store_stats = gufo::expert_stream::Stats;
using ds41_expert_overlap = gufo::expert_stream::Overlap;

/* Available unified host memory, including reclaimable file cache, as Gufo
 * measures it (gufo::ext::HostAvailableBytes). */
uint64_t ds41_expert_available_memory();

/* The model and its shard descriptors outlive the store. cache_limit_bytes
 * bounds the cache when nonzero. The cache is empty until it is sized and
 * filled: by serving once the sessions exist
 * (gufo::ext::TextRunnerExpertCacheClaim), or by
 * ds41_expert_store_fit_available. It is filled from histogram_path, a
 * DS41_EXPERT_HISTOGRAM_V1 text file, when that exists and parses, otherwise
 * from the built-in V4.1 ranking, then with the experts the ranking leaves
 * out. prefill_batch is the engine's batch size (0: the default): the
 * headroom fit_available leaves grows with it. */
ds41_expert_store* ds41_expert_store_create(
    const ds41_model* model, const ds41_weights* weights,
    uint64_t cache_limit_bytes = 0, uint32_t prefill_batch = 0,
    const char* histogram_path = nullptr);
void ds41_expert_store_destroy(ds41_expert_store* store);

/* Splits every expert read between the model's GGUF and a tails file built by
 * tools/ds41/pack_expert_tails.py, normally on a second drive, so both read
 * at once. Call before the cache is filled. The file must match this model's
 * expert tensors byte for byte; false (with a message) leaves reads
 * unchanged. */
bool ds41_expert_store_attach_tails(ds41_expert_store* store, const char* path);

/* The underlying store, which serving sizes once the sessions exist
 * (gufo::ext::TextRunnerExpertCacheClaim). */
gufo::expert_stream::Store* ds41_expert_store_handle(ds41_expert_store* store);

/* Sizes and fills the cache from the memory available now, leaving headroom
 * for the sessions and their batch scratch: for hosts without serving
 * (gufo prompt, gufo bench, probes), right after load. */
bool ds41_expert_store_fit_available(ds41_expert_store* store);

/* Routes are device [n_tokens][6]. The returned gate, up and down tables map
 * each expert index to its device weights and stay valid until the next call.
 * Engine evaluations are serialized, including joint verification. With
 * routes_ready, an event recorded on the null stream once the routes are
 * written, the routes are read without waiting for work queued after it;
 * that work must not touch the expert stage, slots or tables. */
bool ds41_expert_store_prepare(ds41_expert_store* store, uint32_t layer,
                               const int32_t* routes, uint32_t n_tokens,
                               const void** gate, const void** up,
                               const void** down,
                               bool (*overlap)(void*, const ds41_expert_overlap*) = nullptr,
                               void* overlap_context = nullptr,
                               hipEvent_t routes_ready = nullptr);

/* Queue a background read of the given uncached experts of `layer` into the
 * stage set the layer's next prepare consumes, without displacing anything.
 * The reads run only while no demand phase does; the next prepare for the
 * layer adopts what is still pending, and any other prepare drops it.
 * `wide` marks a prefill-scale expert set, which only a batch-scale prepare
 * adopts. Best effort: returns false when the layer is not the next
 * prepare's target or successor, has no uncached listed experts, or the job
 * was refused. */
bool ds41_expert_store_prefetch(ds41_expert_store* store, uint32_t layer,
                                const int32_t* experts, uint32_t count,
                                bool wide = false);

/* Records a router prediction whose ids `done` signals in decoder-owned
 * pinned memory; a store thread submits the prefetch when the event lands.
 * Returns false when the record was refused (best effort by design). */
bool ds41_expert_store_predict(ds41_expert_store* store, uint32_t layer,
                               hipEvent_t done, const int32_t* ids, uint32_t count,
                               bool wide);

/* Reopens the prediction window at `layer`, the next prepare's target, when
 * an evaluation starts somewhere other than after the last prepared layer
 * (a prompt batch that stopped at the replay boundary). Every kernel of the
 * previous evaluation's layers has been ordered before that prepare's
 * route read, so no running kernel reads the layer's parity half. */
void ds41_expert_store_expect(ds41_expert_store* store, uint32_t layer);

/* Waits until every recorded prediction has been submitted, so the ids and
 * event may be released. */
void ds41_expert_store_predict_flush(ds41_expert_store* store);

/* DS41_EXPERT_TRACE=path writes a binary routing trace (the seeded cache,
 * then each prepare's routes, hits and timing) for
 * tools/ds41/prefetch_sim.py. Decoders add router predictions for later
 * layers with ds41_expert_store_trace_prediction once the target layer's
 * prepare returned. Tracing is a measurement mode: it disables the
 * production prediction and adds router work. */
bool ds41_expert_store_tracing(const ds41_expert_store* store);
void ds41_expert_store_trace_prediction(ds41_expert_store* store, uint32_t layer,
                                        uint32_t lead, const int32_t* ids,
                                        uint32_t n_tokens, uint32_t k);

/* True when no background prefetch work is queued or running. */
bool ds41_expert_store_prefetch_idle(const ds41_expert_store* store);
ds41_expert_store_stats ds41_expert_store_get_stats(
    const ds41_expert_store* store);
void ds41_expert_store_log(const ds41_expert_store* store);

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_EXPERT_STORE_H_
