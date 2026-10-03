/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nixl_ep.hpp"

#include "cuda_warn.hpp"
#include "kernels/api.cuh"

#include <pybind11/functional.h>

#include <ATen/cuda/CUDADataType.h>
#include <torch/python.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <unordered_set>
#include <dlfcn.h>
#include <map>
#include <tuple>
#ifdef NIXL_EP_DPA
#include <epdpa.h>
#endif

#define NIXL_ETCD_WATCH_TIMEOUT std::chrono::microseconds(1000000000) // 1000 seconds

namespace {

void sleep_ms(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

uint64_t milliseconds_to_cycles(uint64_t milliseconds, int device_clock_rate_khz) {
    EP_HOST_ASSERT(device_clock_rate_khz > 0);
    return milliseconds * static_cast<uint64_t>(device_clock_rate_khz);
}

} // namespace

namespace nixl_ep {

void Buffer::update_memory_buffers(int num_ranks, int num_experts_per_rank, int64_t num_rdma_bytes, int64_t num_nvl_bytes)
{
    if (!available) {
        init(num_ranks, num_experts_per_rank, num_nvl_bytes, num_rdma_bytes);
        available = true;
    } else {
        throw std::runtime_error("Multiple calls to update_memory_buffers are not supported");
    }
}

Buffer::Buffer(int rank, bool explicitly_destroy, bool low_latency_mode, int timeout_ms):
        low_latency_mode(low_latency_mode),
        timeout_ms([timeout_ms] {
            EP_HOST_ASSERT(timeout_ms >= 0);
            return static_cast<uint64_t>(timeout_ms);
        }()),
        rank(rank),
        explicitly_destroy(explicitly_destroy),
        comm_stream(at::cuda::getStreamFromPool(true)) {}

bool Buffer::_is_rank_connected(int rank_id) const {
    return rank_id == rank or std::find(remote_ranks.begin(), remote_ranks.end(), rank_id) != remote_ranks.end();
}

void Buffer::set_active_rank_bound(int bound) {
    EP_HOST_ASSERT(bound > 0 && "active_rank_bound must be positive");
    active_rank_bound = bound;
}

void Buffer::_refresh_active_rank_bound() {
    int bound = 0;
    for (int rank_id = max_num_ranks - 1; rank_id >= 0; --rank_id) {
        if (active_ranks[rank_id]) {
            bound = rank_id + 1;
            break;
        }
    }
    set_active_rank_bound(bound);
}

int Buffer::get_rank_bound(std::optional<int> num_experts) const {
    if (!num_experts)
        return active_rank_bound;

    EP_HOST_ASSERT(*num_experts % num_experts_per_rank == 0);
    const int active_expert_bound = active_rank_bound * num_experts_per_rank;
    const int max_num_experts = max_num_ranks * num_experts_per_rank;
    EP_HOST_ASSERT(*num_experts >= active_expert_bound and *num_experts <= max_num_experts);
    return *num_experts / num_experts_per_rank;
}

// ---- DPA offload of the HT dispatch RDMA leg (libepdpa, loaded with dlopen; build with -Dnixl_ep_dpa_inc) ----
#ifdef NIXL_EP_DPA
// One DPA stream per (channel, remote rdma rank) gathers x | scales | SourceMeta | topk of every token the
// dispatch kernel publishes in the stream's mailbox straight into the peer's ring slot, waits for ring credit
// on head.buffer(dst) and adds the tokens posted so far to the peer's tail.buffer(my rdma rank) (atomic fetch-add).
// One QP tag per channel: up to 32 channels when the engine has as many tags.
static constexpr int kDpaChannels = EPDPA_MAX_TAGS < 32 ? EPDPA_MAX_TAGS : 32;
static constexpr int kDpaMaxBlocks = 128, kDpaMaxBlockTokens = 128;
static constexpr int kDpaMaxTopK = 128, kDpaMaxScales = 128;
static constexpr int kDpaMaxRecBytes = kDpaMaxScales * 4 + 8 + kDpaMaxTopK * 8;
static constexpr int kDpaMaxStageBytes = 16384; // per token, NIXL_EP_DPA_X_STAGE=1 (diagnostic)
static constexpr size_t kDpaAlign = 1 << 16;

static int dpa_env(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : dflt;
}

static int dpa_max_tokens() {
    return dpa_env("NIXL_EP_DPA_MAX_TOKENS", 16384);
}

struct DpaScratch {
    size_t mbox, sig, meta, topk, scales, rec, stage, end;
    explicit DpaScratch(int num_rdma_ranks) {
        auto up = [](size_t v) { return (v + kDpaAlign - 1) / kDpaAlign * kDpaAlign; };
        const size_t t = dpa_max_tokens();
        mbox = 0;
        sig = up(mbox + size_t(kDpaChannels) * num_rdma_ranks * kDpaMaxBlocks * 32);
        meta = up(sig + size_t(kDpaChannels) * num_rdma_ranks * EPDPA_SIG_SLOTS * 8);
        topk = up(meta + size_t(num_rdma_ranks) * t * 8);
        scales = up(topk + t * 2 * kDpaMaxTopK * 4);
        rec = up(scales + t * kDpaMaxScales * 4);
        stage = up(rec + size_t(num_rdma_ranks) * t * kDpaMaxRecBytes);
        end = up(stage + (dpa_env("NIXL_EP_DPA_X_STAGE", 0) ? t * kDpaMaxStageBytes : 0));
    }
};

size_t dpa_scratch_offset(int64_t num_rdma_bytes) {
    return (static_cast<size_t>(num_rdma_bytes) + (2u << 20) - 1) / (2u << 20) * (2u << 20);
}

size_t dpa_scratch_size(int num_rdma_ranks) {
    return DpaScratch(num_rdma_ranks).end;
}

struct DpaPeer {
    uint64_t base;
    uint32_t rkey, sig_rkey;
};

struct DpaState {
    decltype(&epdpa_core_create) core_create;
    decltype(&epdpa_reg) reg;
    decltype(&epdpa_reg_cached) reg_cached;
    decltype(&epdpa_window) window;
    decltype(&epdpa_peer_create) peer_create;
    decltype(&epdpa_peer_connect) peer_connect;
    decltype(&epdpa_peer_destroy) peer_destroy;
    decltype(&epdpa_quiesce) quiesce;
    decltype(&epdpa_streams) streams;
    decltype(&epdpa_streams_free) streams_free;
    decltype(&epdpa_launch) launch;
    decltype(&epdpa_allgather) allgather;
    epdpa_core* core = nullptr;
    pybind11::function ag; // Python all_gather_object over the EP group (kept for (re)connects)
    int rank = 0, rdma_rank = 0, nvl_rank = 0, num_rdma_ranks = 0;
    uint8_t* base = nullptr; // rdma buffer, scratch at scratch_off
    size_t scratch_off = 0;
    DpaScratch layout{1};
    ibv_mr *mr = nullptr, *sig_mr = nullptr;
    doca_dpa_dev_mmap_t mm = 0;
    DpaPeer me{};
    std::vector<DpaPeer> peers; // [global rank]
    // Per remote rdma rank (the peer of the same nvl rank): connection, QP / completion per channel (tag)
    std::vector<epdpa_peer*> conn;
    std::vector<std::vector<doca_dpa_dev_verbs_qp_t>> qp;
    std::vector<std::vector<doca_dpa_dev_completion_t>> comp;
    uint32_t sq_depth = 0;
    std::map<std::tuple<int, int, int, int>, std::pair<uint64_t, int>> launch_streams; // (channels, bpt, ring, block|sig)
    uint64_t round = 0;
    bool fused = false; // NIXL_EP_DPA_FUSED=1: one block per channel (see the dispatch kernel)
    int sge = 2;         // NIXL_EP_DPA_SGE: 2 = x + packed [scales | meta | topk], 4 = x, scales, meta, topk
    int inflight = 16;   // NIXL_EP_DPA_INFLIGHT: tail signals in flight per stream (<= EPDPA_SIG_SLOTS)
    int sig_max = 16;    // largest signal step the send queues are sized for (NIXL_EP_DPA_SIG_EVERY, else 16)
    int fence_token = 0; // NIXL_EP_DPA_FENCE=token: system fence per token instead of per (warp, block)
    bool x_stage = false; // NIXL_EP_DPA_X_STAGE=1: copy x into the RDMA MR before the kernel (diagnostic)
    int reconnect_every = 0; // NIXL_EP_DPA_RECONNECT_EVERY=N: drop and re-create all peer connections every N dispatches
    uint64_t dispatches = 0;
    int reconnects = 0;
    double disconnect_ms = 0, connect_ms = 0, last_disconnect_ms = 0;
};

static bool dpa_fused(const void* state) {
    return state != nullptr and static_cast<const DpaState*>(state)->fused;
}

static int dpa_allgather_cb(void* user, void* buf, size_t elem) {
    auto* d = static_cast<DpaState*>(user);
    pybind11::gil_scoped_acquire gil;
    auto* b = static_cast<char*>(buf);
    auto all = d->ag(pybind11::bytes(b + size_t(d->rank) * elem, elem)).cast<pybind11::list>();
    for (size_t i = 0; i < all.size(); ++i) {
        const std::string s = all[i].cast<std::string>();
        EP_HOST_ASSERT(s.size() == elem);
        std::memcpy(b + i * elem, s.data(), elem);
    }
    return 0;
}

static void dpa_clear_streams(DpaState* d) {
    for (auto& kv : d->launch_streams)
        d->streams_free(d->core, kv.second.first);
    d->launch_streams.clear();
}

// What a rank publishes for one remote rdma rank: its QP numbers towards it and its buffer.
struct DpaConnMsg {
    epdpa_conn_info info;
    DpaPeer peer;
};

// Collective: connects to the given remote rdma ranks (peer = r * NVL + nvl_rank), one QP per channel (tag).
static void dpa_connect_peers(DpaState* d, const std::vector<int>& rdma_peers) {
    const int R = d->num_rdma_ranks;
    std::vector<DpaConnMsg> all(size_t(R) * NUM_MAX_NVL_PEERS * R); // [rank][remote rdma rank]
    DpaConnMsg* mine = all.data() + size_t(d->rank) * R;
    for (int r : rdma_peers) {
        EP_HOST_ASSERT(r != d->rdma_rank and d->conn[r] == nullptr);
        EP_HOST_ASSERT(d->peer_create(d->core, kDpaChannels, d->sq_depth, EPDPA_SIG_SLOTS, &d->conn[r], &mine[r].info) == 0);
        mine[r].peer = d->me;
    }
    EP_HOST_ASSERT(dpa_allgather_cb(d, all.data(), sizeof(DpaConnMsg) * R) == 0);
    for (int r : rdma_peers) {
        const int p = r * NUM_MAX_NVL_PEERS + d->nvl_rank;
        const DpaConnMsg& m = all[size_t(p) * R + d->rdma_rank];
        d->peers[p] = m.peer;
        d->qp[r].resize(kDpaChannels);
        d->comp[r].resize(kDpaChannels);
        EP_HOST_ASSERT(d->peer_connect(d->conn[r], &m.info, d->qp[r].data(), d->comp[r].data()) == 0);
    }
    dpa_clear_streams(d);
}

// Collective: drops the connections to the given remote rdma ranks once no WQE is in flight on either side.
static void dpa_disconnect_peers(DpaState* d, const std::vector<int>& rdma_peers) {
    EP_HOST_ASSERT(d->quiesce(d->core) == 0);
    std::vector<char> barrier(size_t(d->num_rdma_ranks) * NUM_MAX_NVL_PEERS); // peers have quiesced as well
    EP_HOST_ASSERT(dpa_allgather_cb(d, barrier.data(), 1) == 0);
    for (int r : rdma_peers) {
        d->peer_destroy(d->conn[r]);
        d->conn[r] = nullptr;
        d->qp[r].clear();
        d->comp[r].clear();
    }
    dpa_clear_streams(d);
}

// Remote rdma ranks among the given global ranks that are DPA peers of this rank (same nvl rank).
static std::vector<int> dpa_rdma_peers(const DpaState* d, const std::vector<int>& ranks) {
    std::vector<int> out;
    for (int x : ranks)
        if (x % NUM_MAX_NVL_PEERS == d->nvl_rank and x / NUM_MAX_NVL_PEERS != d->rdma_rank)
            out.push_back(x / NUM_MAX_NVL_PEERS);
    return out;
}

static std::vector<int> dpa_all_rdma_peers(const DpaState* d) {
    std::vector<int> out;
    for (int r = 0; r < d->num_rdma_ranks; ++r)
        if (r != d->rdma_rank and d->conn[r] != nullptr)
            out.push_back(r);
    return out;
}

void Buffer::dpa_connect_ranks(const std::vector<int>& ranks) {
    if (dpa_state == nullptr)
        return;
    auto* d = static_cast<DpaState*>(dpa_state);
    std::vector<int> peers;
    for (int r : dpa_rdma_peers(d, ranks))
        if (d->conn[r] == nullptr)
            peers.push_back(r);
    dpa_connect_peers(d, peers);
}

void Buffer::dpa_disconnect_ranks(const std::vector<int>& ranks) {
    if (dpa_state == nullptr)
        return;
    auto* d = static_cast<DpaState*>(dpa_state);
    std::vector<int> peers;
    for (int r : dpa_rdma_peers(d, ranks))
        if (d->conn[r] != nullptr)
            peers.push_back(r);
    dpa_disconnect_peers(d, peers);
}

void Buffer::dpa_init(const pybind11::function& all_gather_object) {
    EP_HOST_ASSERT(not low_latency_mode and dpa_scratch_bytes > 0 and dpa_state == nullptr);
    const char* lib = std::getenv("NIXL_EP_DPA_LIB") ? std::getenv("NIXL_EP_DPA_LIB") : "libepdpa.so";
    void* h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr)
        throw std::runtime_error(std::string("NIXL_EP_DPA: dlopen failed: ") + dlerror());
    auto* d = new DpaState();
#define DPA_SYM(field, name) EP_HOST_ASSERT((d->field = reinterpret_cast<decltype(d->field)>(dlsym(h, #name))) != nullptr)
    DPA_SYM(core_create, epdpa_core_create);
    DPA_SYM(reg, epdpa_reg);
    DPA_SYM(reg_cached, epdpa_reg_cached);
    DPA_SYM(window, epdpa_window);
    DPA_SYM(peer_create, epdpa_peer_create);
    DPA_SYM(peer_connect, epdpa_peer_connect);
    DPA_SYM(peer_destroy, epdpa_peer_destroy);
    DPA_SYM(quiesce, epdpa_quiesce);
    DPA_SYM(streams, epdpa_streams);
    DPA_SYM(streams_free, epdpa_streams_free);
    DPA_SYM(launch, epdpa_launch);
    DPA_SYM(allgather, epdpa_allgather);
#undef DPA_SYM
    d->ag = all_gather_object;
    d->rank = rank, d->rdma_rank = rdma_rank, d->nvl_rank = nvl_rank, d->num_rdma_ranks = num_rdma_ranks;
    d->reconnect_every = dpa_env("NIXL_EP_DPA_RECONNECT_EVERY", 0);
    d->fused = dpa_env("NIXL_EP_DPA_FUSED", 0) == 1;
    d->sge = dpa_env("NIXL_EP_DPA_SGE", 2);
    d->inflight = dpa_env("NIXL_EP_DPA_INFLIGHT", EPDPA_SIG_SLOTS);
    d->sig_max = dpa_env("NIXL_EP_DPA_SIG_EVERY", 16);
    d->fence_token = std::getenv("NIXL_EP_DPA_FENCE") and std::string(std::getenv("NIXL_EP_DPA_FENCE")) == "token";
    d->x_stage = dpa_env("NIXL_EP_DPA_X_STAGE", 0) == 1;
    EP_HOST_ASSERT((d->sge == 2 or d->sge == 4) and d->inflight >= 1 and d->inflight <= EPDPA_SIG_SLOTS and
                   d->sig_max >= 1);
    d->base = static_cast<uint8_t*>(rdma_buffer_ptr);
    d->scratch_off = dpa_scratch_offset(num_rdma_bytes);
    d->layout = DpaScratch(num_rdma_ranks);
    const size_t total = d->scratch_off + dpa_scratch_bytes;
    EP_HOST_ASSERT(d->core_create(&d->core, rank, num_rdma_ranks * NUM_MAX_NVL_PEERS, device_id, dpa_allgather_cb, d) == 0);
    d->mr = d->reg(d->core, d->base, total, 1);
    d->sig_mr = d->reg(d->core, d->base, total, 0); // tail signals must land after the data: no relaxed ordering
    EP_HOST_ASSERT(d->mr and d->sig_mr and d->window(d->core, d->base, total, &d->mm) == 0);

    d->me = DpaPeer{reinterpret_cast<uint64_t>(d->base), d->mr->rkey, d->sig_mr->rkey};
    d->peers.resize(num_rdma_ranks * NUM_MAX_NVL_PEERS);
    d->peers[rank] = d->me;
    d->conn.assign(num_rdma_ranks, nullptr);
    d->qp.resize(num_rdma_ranks);
    d->comp.resize(num_rdma_ranks);
    // Outstanding WQEs per QP: up to `inflight` signals, each after at most sig_max data WQEs
    d->sq_depth = static_cast<uint32_t>((d->sig_max + 1) * EPDPA_SIG_SLOTS + 16);
    std::vector<int> all;
    for (int r = 0; r < num_rdma_ranks; ++r)
        if (r != rdma_rank)
            all.push_back(r);
    dpa_connect_peers(d, all);
    const int n = static_cast<int>(all.size()) * kDpaChannels;
    dpa_state = d;
    if (rank == 0)
        printf("[nixl_ep] DPA offload of the HT dispatch RDMA leg: %d QPs, scratch %.1f MB, %d SGE, %d signals in "
               "flight, fence per %s%s%s\n", n, dpa_scratch_bytes / 1e6, d->sge, d->inflight,
               d->fence_token ? "token" : "block", d->x_stage ? ", x staged" : "",
               d->fused ? ", fused (1 SM per channel)" : "");

    const auto& L = d->layout;
    uint8_t* s = d->base + d->scratch_off;
    gpu_ctx.dpa.mbox = reinterpret_cast<uint64_t*>(s + L.mbox);
    gpu_ctx.dpa.meta = reinterpret_cast<uint64_t*>(s + L.meta);
    gpu_ctx.dpa.topk = reinterpret_cast<int*>(s + L.topk);
    gpu_ctx.dpa.scales = reinterpret_cast<float*>(s + L.scales);
    gpu_ctx.dpa.rec = s + L.rec;
    gpu_ctx.dpa.fence_token = d->fence_token;
    gpu_ctx.dpa.max_tokens = dpa_max_tokens();
    CUDA_CHECK(cudaMemset(s, 0, L.sig)); // mailboxes start at round 0
}

// Builds (once per layout) the streams of this dispatch and launches the DPA; returns the kernel's DPA view.
static nixl_ep::gpu_nixl_ctx dpa_dispatch(DpaState* d, nixl_ep::gpu_nixl_ctx ctx, int rdma_rank, int num_rdma_ranks,
                                          int num_channels, int hidden_bytes, int num_scales, int num_topk,
                                          int ring, int chunked_send, int num_tokens, const void* x,
                                          const torch::Tensor& x_tensor, cudaStream_t stream) {
    // Block size (NIXL_EP_DPA_BLOCK, default 64) and tail signal granularity (NIXL_EP_DPA_SIG_EVERY, default
    // chunked_send, the receivers' credit step). The engine flushes the pending tail before waiting for ring
    // credit, so block and ring are independent.
    const char* bs = std::getenv("NIXL_EP_DPA_BLOCK");
    const char* se = std::getenv("NIXL_EP_DPA_SIG_EVERY");
    const int block = bs ? std::atoi(bs) : 64;
    const int sig_every = se ? std::atoi(se) : chunked_send;
    EP_HOST_ASSERT(num_channels <= kDpaChannels and num_topk <= kDpaMaxTopK and num_scales <= kDpaMaxScales and
                   num_tokens <= ctx.dpa.max_tokens and block % 32 == 0 and block <= kDpaMaxBlockTokens and
                   sig_every >= 0 and std::min(sig_every ? sig_every : block, block) <= d->sig_max);
    // Test knob: drop and re-create every peer connection (exercises the elastic teardown / create path)
    double reconnect_t0 = 0;
    if (d->reconnect_every > 0 and d->dispatches > 0 and d->dispatches % d->reconnect_every == 0) {
        using clk = std::chrono::steady_clock;
        const auto t0 = clk::now();
        const auto peers = dpa_all_rdma_peers(d);
        dpa_disconnect_peers(d, peers);
        const auto t1 = clk::now();
        dpa_connect_peers(d, peers);
        d->last_disconnect_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        d->disconnect_ms += d->last_disconnect_ms;
        reconnect_t0 = std::chrono::duration<double, std::milli>(t1.time_since_epoch()).count();
        ++d->reconnects;
    }
    ++d->dispatches;
    const int bpt = (hidden_bytes + num_scales * 4 + 8 + num_topk * 8 + 15) / 16 * 16; // ht::get_num_bytes_per_token
    const int rec_bytes = num_scales * 4 + 8 + num_topk * 8; // 2-SGE record: the tail of the token record
    const auto key = std::make_tuple(num_channels, bpt, ring, block * 1024 + sig_every);
    auto it = d->launch_streams.find(key);
    if (it == d->launch_streams.end()) {
        // RDMA buffer layout of the dispatch kernel (SymBuffer chain): data, meta, head, tail
        const int64_t R = num_rdma_ranks, data_ch = int64_t(ring) * bpt * R;
        const int64_t head_off = data_ch * num_channels * 2 + int64_t(NUM_MAX_NVL_PEERS * 2 + 2) * 4 * R * num_channels * 2;
        const int64_t tail_off = head_off + 8 * R * num_channels;
        const auto& L = d->layout;
        const uint64_t scratch = reinterpret_cast<uint64_t>(d->base) + d->scratch_off;
        std::vector<epdpa_stream> st;
        for (int ch = 0; ch < num_channels; ++ch)
            for (int r = 0, q = 0; r < num_rdma_ranks; ++r) {
                if (r == rdma_rank)
                    continue;
                const int qi = ch * (num_rdma_ranks - 1) + q++; // signal source slots per (channel, remote)
                if (d->conn[r] == nullptr)
                    continue; // not connected: no stream
                const DpaPeer& p = d->peers[r * NUM_MAX_NVL_PEERS + (d->rank % NUM_MAX_NVL_PEERS)];
                epdpa_stream s{};
                s.comp = d->comp[r][ch], s.qp = d->qp[r][ch], s.mm = d->mm;
                s.sig_lkey = d->mr->lkey, s.rkey = p.rkey, s.sig_rkey = p.sig_rkey, s.nsge = 4;
                for (int i = 0; i < EPDPA_MAX_SGE; ++i)
                    s.mul[i] = 1;
                // per-destination piece: 2-SGE records or SourceMeta, both [dst][token]
                s.add[d->sge == 2 ? 1 : 2] = static_cast<uint32_t>(r * ctx.dpa.max_tokens);
                s.mbox = scratch + L.mbox + (uint64_t(ch) * R + r) * kDpaMaxBlocks * 32;
                s.block_tokens = block, s.first_block = 0, s.block_step = 1, s.nblocks = kDpaMaxBlocks;
                s.dst_base = p.base + data_ch * (ch + num_channels) + int64_t(ring) * bpt * rdma_rank;
                s.dst_stride = bpt;
                s.ring = ring, s.max_inflight = d->inflight;
                s.credit = reinterpret_cast<uint64_t>(d->base) + head_off + 8 * (R * ch + r);
                s.sig_src = scratch + L.sig + uint64_t(qi) * EPDPA_SIG_SLOTS * 8;
                s.sig_dst = p.base + tail_off + 8 * (R * ch + rdma_rank);
                s.sig_mode = EPDPA_SIG_TAIL, s.tok_from_mbox = 1, s.sig_every = static_cast<uint32_t>(sig_every);
                st.push_back(s);
            }
        uint64_t h = 0;
        EP_HOST_ASSERT(d->streams(d->core, st.data(), static_cast<int>(st.size()), &h) == 0);
        it = d->launch_streams.emplace(key, std::make_pair(h, static_cast<int>(st.size()))).first;
    }
    const auto& L = d->layout;
    const uint64_t scratch = reinterpret_cast<uint64_t>(d->base) + d->scratch_off;
    const int tokens_per_channel = (num_tokens + num_channels - 1) / num_channels;
    struct epdpa_launch p{};
    p.round = ++d->round;
    if (d->x_stage) {
        EP_HOST_ASSERT(hidden_bytes <= kDpaMaxStageBytes);
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(scratch + L.stage), x, size_t(num_tokens) * hidden_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        p.addr[0] = scratch + L.stage, p.len[0] = hidden_bytes, p.lkey[0] = d->mr->lkey;
    } else {
        p.addr[0] = reinterpret_cast<uint64_t>(x), p.len[0] = hidden_bytes;
        p.lkey[0] = d->reg_cached(d->core, x, x_tensor.numel() * x_tensor.element_size());
    }
    for (int i = 1; i < EPDPA_MAX_SGE; ++i)
        p.lkey[i] = d->mr->lkey;
    if (d->sge == 2) {
        p.addr[1] = scratch + L.rec, p.len[1] = rec_bytes;
    } else {
        p.addr[1] = scratch + L.scales, p.len[1] = num_scales * 4;
        p.addr[2] = scratch + L.meta, p.len[2] = 8;
        p.addr[3] = scratch + L.topk, p.len[3] = num_topk * 8;
    }
    p.nblocks = (tokens_per_channel + block - 1) / block;
    EP_HOST_ASSERT(p.lkey[0] != 0 and p.nblocks <= kDpaMaxBlocks);
    EP_HOST_ASSERT(d->launch(d->core, it->second.first, it->second.second, &p) == 0);
    if (reconnect_t0 > 0) { // connect time includes rebuilding the streams with the new QPs
        const double now = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
        d->connect_ms += now - reconnect_t0;
        if (d->rank == 0 and (d->reconnects <= 2 or d->reconnects % 50 == 0))
            printf("[nixl_ep] DPA reconnect #%d (%d peers x %d QPs): disconnect %.2f ms, connect + streams %.2f ms (avg "
                   "%.2f / %.2f ms)\n", d->reconnects, static_cast<int>(dpa_all_rdma_peers(d).size()), kDpaChannels,
                   d->last_disconnect_ms, now - reconnect_t0, d->disconnect_ms / d->reconnects,
                   d->connect_ms / d->reconnects);
    }
    ctx.dpa.round = p.round;
    ctx.dpa.max_blocks = static_cast<int>(p.nblocks);
    ctx.dpa.block_tokens = block;
    ctx.dpa.rec_bytes = d->sge == 2 ? rec_bytes : 0;
    return ctx;
}
#else
size_t dpa_scratch_offset(int64_t num_rdma_bytes) { return static_cast<size_t>(num_rdma_bytes); }
size_t dpa_scratch_size(int) { return 0; }
struct DpaState {};
static bool dpa_fused(const void*) { return false; }
void Buffer::dpa_connect_ranks(const std::vector<int>&) {}
void Buffer::dpa_disconnect_ranks(const std::vector<int>&) {}
void Buffer::dpa_init(const pybind11::function&) {
    throw std::runtime_error("NIXL_EP_DPA=1 needs nixl_ep built with -Dnixl_ep_dpa_inc=<libepdpa include dir>");
}
static nixl_ep::gpu_nixl_ctx dpa_dispatch(DpaState*, nixl_ep::gpu_nixl_ctx ctx, int, int, int, int, int, int, int, int,
                                          int, const void*, const torch::Tensor&, cudaStream_t) {
    return ctx;
}
#endif

void Buffer::init(int num_ranks, int num_experts_per_rank, int64_t num_nvl_bytes, int64_t num_rdma_bytes)
{
    EP_HOST_ASSERT(num_ranks > 0);
    EP_HOST_ASSERT(num_experts_per_rank > 0);

    // Update buffer attributes
    this->max_num_ranks = num_ranks;
    this->num_experts_per_rank = num_experts_per_rank;
    this->num_nvl_bytes = num_nvl_bytes;
    this->num_rdma_bytes = num_rdma_bytes;

    // Metadata memory
    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

    // Common checks
    EP_STATIC_ASSERT(NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);

    // Get ranks
    CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

    // Get device info
    int device_clock_rate_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_device_sms, cudaDevAttrMultiProcessorCount, device_id));
    CUDA_CHECK(cudaDeviceGetAttribute(&device_clock_rate_khz, cudaDevAttrClockRate, device_id));
    timeout_cycles = milliseconds_to_cycles(timeout_ms, device_clock_rate_khz);
    int denom_sms = std::max(1, num_device_sms / 2);
    auto per_channel_bytes = ceil_div<int64_t>(num_rdma_bytes, denom_sms);
    EP_HOST_ASSERT(per_channel_bytes < std::numeric_limits<int>::max());

    if (num_nvl_bytes > 0) {
        // Local IPC: alloc local memory and set local IPC handles
        CUDA_CHECK(cudaMalloc(&buffer_ptrs[nvl_rank], num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes));
        CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));
        buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

        // Set barrier signals
        barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        barrier_signal_ptrs_gpu = reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
    }

    // Create 32 MiB workspace
    m_workspace_alloc = std::make_unique<vmm_region>(NUM_WORKSPACE_BYTES);
    workspace = m_workspace_alloc->ptr();
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    if (!low_latency_mode) {
        // MoE counter
        CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
        *moe_recv_counter = -1;

        // MoE expert-level counter
        CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
        for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++ i)
            moe_recv_expert_counter[i] = -1;

        // MoE RDMA-level counter
        CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
        *moe_recv_rdma_counter = -1;
    }

    // DPA mode: the DPA scratch (mailboxes, packed token pieces, signal sources) lives after the RDMA buffer so a
    // single Data Direct MR and DPA window cover it together with the head/tail counters.
    dpa_scratch_bytes = (std::getenv("NIXL_EP_DPA") and std::string(std::getenv("NIXL_EP_DPA")) == "1" and
                         not low_latency_mode) ? dpa_scratch_size(num_rdma_ranks) : 0;
    m_rdma_alloc = std::make_unique<vmm_region>(static_cast<size_t>(dpa_scratch_offset(num_rdma_bytes) + dpa_scratch_bytes));
    rdma_buffer_ptr = m_rdma_alloc->ptr();
    CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

    // Allocate and clean shrink buffer
    int num_mask_buffer_bytes = max_num_ranks * sizeof(int);
    m_mask_alloc = std::make_unique<vmm_region>(static_cast<size_t>(num_mask_buffer_bytes));
    mask_buffer_ptr = static_cast<int *>(m_mask_alloc->ptr());
    CUDA_CHECK(cudaMemset(mask_buffer_ptr, 0xff, num_mask_buffer_bytes));
    CUDA_CHECK(cudaMemset(mask_buffer_ptr + rank, 0, sizeof(int)));
    active_ranks.assign(max_num_ranks, false);
    active_ranks[rank] = true;
    set_active_rank_bound(rank + 1);

    int num_sync_buffer_bytes = max_num_ranks * sizeof(int);
    m_sync_alloc = std::make_unique<vmm_region>(static_cast<size_t>(num_sync_buffer_bytes));
    sync_buffer_ptr = static_cast<int *>(m_sync_alloc->ptr());
    m_sync_count_alloc = std::make_unique<vmm_region>(static_cast<size_t>(num_sync_buffer_bytes));
    sync_count_ptr = static_cast<int *>(m_sync_count_alloc->ptr());
    CUDA_CHECK(cudaMemset(sync_buffer_ptr, 0, num_sync_buffer_bytes));
    CUDA_CHECK(cudaMemset(sync_count_ptr, 0, num_sync_buffer_bytes));

    if (!low_latency_mode) {
        CUDA_CHECK(cudaMalloc(&local_ht_barrier_counter, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(local_ht_barrier_counter, 0, sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&last_ht_barrier_counter, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(last_ht_barrier_counter, 0, sizeof(uint64_t)));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    my_peer_info.rdma_buffer_ptr = rdma_buffer_ptr;
    my_peer_info.device_id = get_local_device_id();
    my_peer_info.sync_buffer_ptr = sync_buffer_ptr;
    my_peer_info.ht_barrier_ptr = local_ht_barrier_counter;
    my_peer_info.rank = rank;

    nixl_peer_info.resize(max_num_ranks);
    nixl_peer_info[rank] = my_peer_info;

    _nixl_agent_init();

    _nixl_ep_init();
}

Buffer::~Buffer() noexcept {
    if (not explicitly_destroy) {
        destroy();
    } else if (not destroyed) {
        printf("WARNING: destroy() was not called before NIXL_EP buffer destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

bool Buffer::is_available() const {
    return available;
}

bool Buffer::is_ht_available() const {
    return is_available() and max_num_ranks > NUM_MAX_NVL_PEERS;
}

int Buffer::get_num_rdma_ranks() const {
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
    return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const {
    return device_id;
}

pybind11::bytearray Buffer::get_local_ipc_handle() const {
    return {ipc_handles[nvl_rank].reserved, CUDA_IPC_HANDLE_SIZE};
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) + offset;
    auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
    return torch::from_blob(base_ptr, num_bytes / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));
}

torch::Stream Buffer::get_comm_stream() const {
    return comm_stream;
}

void Buffer::destroy() {
    auto warn_cuda = [](cudaError_t status, const char *operation) noexcept {
        cuda::warn(status, "Buffer::destroy()", operation);
    };

    auto warn_nixl = [](nixl_status_t status, const char* operation) noexcept {
        if (status != NIXL_SUCCESS) {
            std::cerr << "WARNING: destroy() failed to " << operation
                      << ": " << nixlEnumStrings::statusStr(status) << '\n';
        }
    };

    if (destroyed) {
        return;
    }

    // Synchronize
    warn_cuda(cudaDeviceSynchronize(), "synchronize device");

    _nixl_ep_destroy();

    if (num_nvl_bytes > 0) {
        intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, timeout_cycles, comm_stream);
        warn_cuda(cudaDeviceSynchronize(), "synchronize device after intranode barrier");

        // Close remote IPC
        if (is_available()) {
            for (int i = 0; i < num_nvl_ranks; ++ i) if (i != nvl_rank)
                warn_cuda(cudaIpcCloseMemHandle(buffer_ptrs[i]), "close remote IPC handle");
        }

        // Free local buffer
        warn_cuda(cudaFree(buffer_ptrs[nvl_rank]), "free local NVL buffer");
    }

    if (nixl_agent_info and nixl_agent_info->agent != nullptr) {
        if (getenv("NIXL_ETCD_ENDPOINTS")) {
            warn_nixl(nixl_agent_info->agent->invalidateLocalMD(),
                      "invalidate local metadata");
        }
        warn_nixl(nixl_agent_info->agent->deregisterMem(
                      nixl_agent_info->rdma_reg_descs,
                      &nixl_agent_info->extra_params),
                  "deregister RDMA memory");
        warn_nixl(nixl_agent_info->agent->deregisterMem(
                      nixl_agent_info->sync_reg_descs,
                      &nixl_agent_info->extra_params),
                  "deregister sync memory");
        warn_nixl(nixl_agent_info->agent->deregisterMem(
                      nixl_agent_info->sync_count_reg_descs,
                      &nixl_agent_info->extra_params),
                  "deregister sync-count memory");
        if (local_ht_barrier_counter != nullptr) {
            warn_nixl(nixl_agent_info->agent->deregisterMem(
                          nixl_agent_info->ht_barrier_reg_descs),
                      "deregister ht barrier memory");
        }

        nixl_agent_info.reset();
    }

    m_rdma_alloc.reset();
    rdma_buffer_ptr = nullptr;
    m_mask_alloc.reset();
    mask_buffer_ptr = nullptr;
    m_sync_alloc.reset();
    sync_buffer_ptr = nullptr;
    m_sync_count_alloc.reset();
    sync_count_ptr = nullptr;

    if (!low_latency_mode) {
        warn_cuda(cudaFree(local_ht_barrier_counter), "free local ht barrier counter");
        local_ht_barrier_counter = nullptr;
        warn_cuda(cudaFree(last_ht_barrier_counter), "free last ht barrier counter");
        last_ht_barrier_counter = nullptr;
        warn_cuda(cudaFreeHost(const_cast<int*>(moe_recv_counter)), "free moe receive counter");
        moe_recv_counter = nullptr;
        warn_cuda(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)), "free moe receive expert counter");
        moe_recv_expert_counter = nullptr;
        warn_cuda(cudaFreeHost(const_cast<int*>(moe_recv_rdma_counter)), "free moe receive rdma counter");
        moe_recv_rdma_counter = nullptr;
    }

    m_workspace_alloc.reset();
    workspace = nullptr;

    destroyed = true;
    available = false;
}

void Buffer::barrier() {
    ep_kernels::barrier(gpu_ctx_ptr, mask_buffer_ptr, timeout_cycles, cuda_stream::get_current());
}

void Buffer::_nixl_agents_connect(const std::vector<int>& ranks, const std::vector<nixl_blob_t>& remote_mds) {
    EP_HOST_ASSERT(!ranks.empty());
    EP_HOST_ASSERT(remote_mds.empty() || remote_mds.size() == ranks.size());

    // Assuming ranks vector does not include current rank and has only new ranks
    remote_ranks.insert(remote_ranks.end(), ranks.begin(), ranks.end());
    for (int remote_rank : ranks) {
        nixl_agent_info->remote_agent_names[remote_rank] = std::to_string(remote_rank);
    }

    // Fire all get metadata requests in parallel
    for (size_t i = 0; i < ranks.size(); i++) {
        int remote_rank = ranks[i];
        std::string agent_name;

        nixl_status_t status = remote_mds.empty()
            ? nixl_agent_info->agent->fetchRemoteMD(nixl_agent_info->remote_agent_names[remote_rank])
            : nixl_agent_info->agent->loadRemoteMD(remote_mds[i], agent_name);

        if (status != NIXL_SUCCESS) {
            throw std::runtime_error("Failed to get metadata for remote agent " +
                                    std::to_string(remote_rank) + ", status: " + std::to_string(status));
        }
    }

    // Wait for all remote metadata to be available
    std::vector<bool> peer_ready(max_num_ranks, false);
    int peers_remaining = static_cast<int>(ranks.size());

    while (peers_remaining > 0) {
        for (int remote_rank : ranks) {
            if (peer_ready[remote_rank]) continue;

            nixl_xfer_dlist_t empty_descs(VRAM_SEG);
            if (nixl_agent_info->agent->checkRemoteMD(std::to_string(remote_rank), empty_descs) == NIXL_SUCCESS) {
                peer_ready[remote_rank] = true;
                peers_remaining--;
            }
        }
        if (peers_remaining > 0) {
            sleep_ms(10);
        }
    }
}

void Buffer::_nixl_agents_peer_info_gather(std::vector<int>& ranks) {
    for (int remote_rank : ranks) {
        std::string my_peer_info_str(reinterpret_cast<const char*>(&my_peer_info), sizeof(NixlPeerInfo));
        nixl_agent_info->agent->genNotif(std::to_string(remote_rank), my_peer_info_str);
    }

    for (int remote_rank : ranks) {
        do {
            nixl_notifs_t notif_map;
            nixl_agent_info->agent->getNotifs(notif_map);
            for (auto &notif : notif_map) {
                std::string my_peer_info_str = notif.second[0];
                NixlPeerInfo remote_peer_info;
                memcpy(&remote_peer_info, my_peer_info_str.c_str(), sizeof(NixlPeerInfo));
                nixl_peer_info[remote_peer_info.rank] = remote_peer_info;
                nixl_agent_info->wire_up_done[remote_peer_info.rank] = true;
            }
        } while (!nixl_agent_info->wire_up_done[remote_rank]);
    }
}

void Buffer::_ipc_handles_sync(const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles = {}) {
    if (num_nvl_bytes > 0) {
        EP_HOST_ASSERT(all_gathered_handles.size() == max_num_ranks);
        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++ i) {
            EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            EP_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE);
                CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs[i], ipc_handles[i], cudaIpcMemLazyEnablePeerAccess));
                barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
            } else {
                EP_HOST_ASSERT(std::memcmp(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE) == 0);
            }
        }

        // Copy all buffer and barrier signal pointers to GPU
        CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void Buffer::connect_ranks(const std::vector<int>& remote_ranks_list, const std::optional<std::vector<nixl_blob_t>>& remote_mds,
    const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles, bool activate) {
    EP_HOST_ASSERT(!remote_ranks_list.empty());
    EP_HOST_ASSERT(!remote_mds.has_value() || remote_mds->size() == remote_ranks_list.size());

    if (!low_latency_mode && num_nvl_bytes > 0) {
        EP_HOST_ASSERT(remote_ranks.empty() && "connect_ranks called more than once in high-throughput mode; elasticity is not yet supported");
    }
    EP_HOST_ASSERT(low_latency_mode || activate);

    std::vector<int> new_ranks;
    std::vector<nixl_blob_t> new_ranks_mds;

    if (all_gathered_handles.size() > 0)
        _ipc_handles_sync(all_gathered_handles);

    for (size_t i = 0; i < remote_ranks_list.size(); i++) {
        int remote_rank = remote_ranks_list[i];
        EP_HOST_ASSERT(remote_rank >= 0 and remote_rank < max_num_ranks);
        // Skip self and ranks we are already connected to
        if (remote_rank == rank or _is_rank_connected(remote_rank))
            continue;

        new_ranks.push_back(remote_rank);
        CUDA_CHECK(cudaMemset(sync_count_ptr + remote_rank, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(sync_buffer_ptr + remote_rank, 0, sizeof(int)));

        if (remote_mds.has_value())
            new_ranks_mds.push_back((*remote_mds)[i]);
    }

    if (!new_ranks.empty()) {
        pybind11::gil_scoped_release release;
        _nixl_agents_connect(new_ranks, new_ranks_mds);

        _nixl_agents_peer_info_gather(new_ranks);

        _nixl_ep_memory_views_stage();
    }
    dpa_connect_ranks(new_ranks); // DPA QPs to the new peers (no-op before dpa_init)

    if (activate) {
        for (int remote_rank : remote_ranks_list) {
            if (remote_rank != rank)
                update_mask_buffer(remote_rank, false);
        }
    }

    // Ready to use
    available = true;
}

void Buffer::disconnect_ranks(const std::vector<int>& remote_ranks_list) {
    EP_HOST_ASSERT(!remote_ranks_list.empty());
    EP_HOST_ASSERT(remote_ranks_list.size() <= remote_ranks.size());

    CUDA_CHECK(cudaDeviceSynchronize());
    dpa_disconnect_ranks(remote_ranks_list);

    // Update mask buffer to mark ranks as inactive
    for (int removed_rank : remote_ranks_list) {
        EP_HOST_ASSERT(removed_rank != rank);
        EP_HOST_ASSERT(_is_rank_connected(removed_rank));
        update_mask_buffer(removed_rank, true);  // mask=true
        CUDA_CHECK(cudaMemset(gpu_ctx.p2p_ptrs + removed_rank, 0, sizeof(void*)));
    }

    _nixl_agents_peer_info_cleanup(remote_ranks_list);

    _nixl_agents_disconnect(remote_ranks_list);

    // Remove ranks from remote_ranks vector (arbitrary order)
    for (int removed_rank : remote_ranks_list) {
        remote_ranks.erase(
            std::remove(remote_ranks.begin(), remote_ranks.end(), removed_rank),
            remote_ranks.end()
        );
    }

    _nixl_ep_memory_views_stage();

    _nixl_ep_memory_views_commit();
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts,
                            std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        cuda_stream::set_current(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event) {
        previous_event->stream_wait(comm_stream);
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
    const int num_ranks = max_num_ranks;
    auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kCUDA));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA));
    if (is_ht_available())
        num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    layout::get_dispatch_layout(topk_idx.data_ptr<topk_idx_t>(),
                                num_tokens_per_rank.data_ptr<int>(),
                                num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,
                                num_tokens_per_expert.data_ptr<int>(),
                                is_token_in_rank.data_ptr<bool>(),
                                num_tokens, num_topk, num_ranks, num_experts,
                                comm_stream);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {num_tokens_per_rdma_rank}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        cuda_stream::set_current(compute_stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::ht_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank, const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                           const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
                           const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                           const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                           int expert_alignment, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(!low_latency_mode && "ht_dispatch() requires high-throughput mode (low_latency_mode=false)");
    const int num_ranks = max_num_ranks;
    // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
    // If users of DeepEP need to execute other Python code on other threads, such as KV transfer, their code will get stuck due to GIL
    // unless we release GIL here.
    pybind11::gil_scoped_release release;
    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and cached_rdma_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and cached_rdma_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and cached_gbl_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)), hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    topk_idx_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        cuda_stream::set_current(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event) {
        previous_event->stream_wait(comm_stream);
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
    auto rdma_channel_prefix_matrix = torch::Tensor();
    auto recv_rdma_rank_prefix_sum = torch::Tensor();
    auto gbl_channel_prefix_matrix = torch::Tensor();
    auto recv_gbl_rank_prefix_sum = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
        rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
        recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
        gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
        recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

        // Just a barrier and clean flags
        ht::cached_notify(hidden_int4, num_scales, num_topk, num_topk,
                                 num_ranks, num_channels, 0, nullptr,
                                 nullptr, nullptr, nullptr,
                                 rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                                 buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
                                 barrier_signal_ptrs_gpu, rank, comm_stream,
                                 config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                 num_nvl_bytes, timeout_cycles, true, gpu_ctx);
    } else {
        rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

        // Send sizes
        *moe_recv_counter = -1;
        *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++ i)
            moe_recv_expert_counter[i] = -1;
        ht::notify_dispatch(num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
                                   num_tokens_per_rdma_rank->data_ptr<int>(), moe_recv_rdma_counter_mapped,
                                   num_tokens_per_expert->data_ptr<int>(), moe_recv_expert_counter_mapped, num_experts,
                                   is_token_in_rank.data_ptr<bool>(), num_tokens, num_channels,
                                   hidden_int4, num_scales, num_topk, expert_alignment,
                                   rdma_channel_prefix_matrix.data_ptr<int>(), recv_rdma_rank_prefix_sum.data_ptr<int>(),
                                   gbl_channel_prefix_matrix.data_ptr<int>(), recv_gbl_rank_prefix_sum.data_ptr<int>(),
                                   rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                                   buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
                                   barrier_signal_ptrs_gpu, rank, comm_stream,
                                   config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                   num_nvl_bytes, timeout_cycles, gpu_ctx);

        // Synchronize total received tokens and tokens per expert
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            // Read total count
            num_recv_tokens = static_cast<int>(*moe_recv_counter);
            num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

            // Read per-expert count
            bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
            for (int i = 0; i < num_local_experts and ready; ++ i)
                ready &= moe_recv_expert_counter[i] >= 0;

            if (ready)
                break;

            // Timeout check
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() > NUM_CPU_TIMEOUT_SECS) {
                for (int i = 0; i < num_local_experts; ++ i)
                    printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                throw std::runtime_error("NixlEP error: timeout (dispatch CPU)");
            }
        }
        num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(), recv_x_scales = std::optional<torch::Tensor>();
    auto recv_src_meta = std::optional<torch::Tensor>();
    auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto send_rdma_head = std::optional<torch::Tensor>();
    auto send_nvl_head = std::optional<torch::Tensor>();
    void* recv_src_meta_ptr = nullptr;
    int* recv_rdma_channel_prefix_matrix_ptr = nullptr;
    int* recv_gbl_channel_prefix_matrix_ptr = nullptr;
    int* send_rdma_head_ptr = nullptr;
    int* send_nvl_head_ptr = nullptr;
    if (not cached_mode) {
        recv_src_meta = torch::empty({num_recv_tokens, ht::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kCUDA));
        recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_src_meta_ptr = recv_src_meta->data_ptr();
        recv_rdma_channel_prefix_matrix_ptr = recv_rdma_channel_prefix_matrix->data_ptr<int>();
        recv_gbl_channel_prefix_matrix_ptr = recv_gbl_channel_prefix_matrix->data_ptr<int>();
        send_rdma_head_ptr = send_rdma_head->data_ptr<int>();
        send_nvl_head_ptr = send_nvl_head->data_ptr<int>();
    }

    // Assign pointers
    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Launch data dispatch
    // NOTES: the buffer size checks are moved into the `.cu` file
    ht::dispatch(recv_x.data_ptr(), recv_x_scales_ptr, recv_topk_idx_ptr, recv_topk_weights_ptr,
                        recv_src_meta_ptr,
                        x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
                        send_rdma_head_ptr, send_nvl_head_ptr,
                        recv_rdma_channel_prefix_matrix_ptr,
                        recv_gbl_channel_prefix_matrix_ptr,
                        rdma_channel_prefix_matrix.data_ptr<int>(), recv_rdma_rank_prefix_sum.data_ptr<int>(),
                        gbl_channel_prefix_matrix.data_ptr<int>(), recv_gbl_rank_prefix_sum.data_ptr<int>(),
                        is_token_in_rank.data_ptr<bool>(),
                        num_tokens, hidden_int4, num_scales, num_topk, num_experts,
                        scale_token_stride, scale_hidden_stride,
                        rdma_buffer_ptr, config.num_max_rdma_chunked_send_tokens, config.num_max_rdma_chunked_recv_tokens,
                        buffer_ptrs_gpu, config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens,
                        rank, num_ranks, cached_mode,
                        comm_stream, num_channels, timeout_cycles,
                        dpa_state ? dpa_dispatch(static_cast<DpaState*>(dpa_state), gpu_ctx, rdma_rank, num_rdma_ranks,
                                                 num_channels, hidden_int4 * 16, num_scales, num_topk,
                                                 config.num_max_rdma_chunked_recv_tokens,
                                                 config.num_max_rdma_chunked_send_tokens, num_tokens, x.data_ptr(), x,
                                                 comm_stream)
                                  : gpu_ctx,
                        dpa_fused(dpa_state));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, is_token_in_rank, recv_x,
                       rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {x_scales, topk_idx, topk_weights,
                        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert,
                        cached_rdma_channel_prefix_matrix, cached_recv_rdma_rank_prefix_sum,
                        cached_gbl_channel_prefix_matrix, cached_recv_gbl_rank_prefix_sum,
                        recv_topk_idx, recv_topk_weights, recv_x_scales,
                        recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix, send_rdma_head, send_nvl_head,
                        recv_src_meta}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        cuda_stream::set_current(compute_stream);

    // Return values
    return {recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
            recv_src_meta, send_rdma_head, send_nvl_head, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::ht_combine(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                          const std::optional<torch::Tensor>& bias_0, const std::optional<torch::Tensor>& bias_1,
                          const torch::Tensor& src_meta, const torch::Tensor& is_combined_token_in_rank,
                          const torch::Tensor& rdma_channel_prefix_matrix, const torch::Tensor& rdma_rank_prefix_sum, const torch::Tensor& gbl_channel_prefix_matrix,
                          const torch::Tensor& combined_rdma_head, const torch::Tensor& combined_nvl_head,
                          const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(!low_latency_mode && "ht_combine() requires high-throughput mode (low_latency_mode=false)");
    const int num_ranks = max_num_ranks;
    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()) == CUDA_R_16BF);
    EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);
    EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and is_combined_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and combined_rdma_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and combined_nvl_head.scalar_type() == torch::kInt32);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)), hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(src_meta.size(1) == ht::get_source_meta_bytes());
    EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and rdma_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and combined_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        cuda_stream::set_current(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event) {
        previous_event->stream_wait(comm_stream);
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Top-k checks
    int num_topk = 0;
    auto combined_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
        combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
    }

    // Extra check for avoid-dead-lock design
    EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

    // Launch barrier and reset queue head and tail
    ht::cached_notify(hidden_int4, 0, 0, num_topk,
                             num_ranks, num_channels,
                             num_combined_tokens, combined_rdma_head.data_ptr<int>(),
                             rdma_channel_prefix_matrix.data_ptr<int>(), rdma_rank_prefix_sum.data_ptr<int>(), combined_nvl_head.data_ptr<int>(),
                             rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                             buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
                             barrier_signal_ptrs_gpu, rank, comm_stream,
                             config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                             num_nvl_bytes, timeout_cycles, false, gpu_ctx);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++ i) if (bias_opts[i].has_value()) {
        auto bias = bias_opts[i].value();
        EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
        EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
        EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
        bias_ptrs[i] = bias.data_ptr();
    }

    // Launch data combine
    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    ht::combine(combined_x.data_ptr(), combined_topk_weights_ptr,
                       is_combined_token_in_rank.data_ptr<bool>(),
                       x.data_ptr(), topk_weights_ptr, bias_ptrs[0], bias_ptrs[1],
                       combined_rdma_head.data_ptr<int>(), combined_nvl_head.data_ptr<int>(),
                       src_meta.data_ptr(), rdma_channel_prefix_matrix.data_ptr<int>(), rdma_rank_prefix_sum.data_ptr<int>(), gbl_channel_prefix_matrix.data_ptr<int>(),
                       num_tokens, num_combined_tokens, hidden, num_topk,
                       rdma_buffer_ptr, config.num_max_rdma_chunked_send_tokens, config.num_max_rdma_chunked_recv_tokens,
                       buffer_ptrs_gpu, config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens,
                       rank, num_ranks, comm_stream, num_channels, timeout_cycles, gpu_ctx);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, src_meta,
                       is_combined_token_in_rank, rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix,
                       combined_x, combined_rdma_head, combined_nvl_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {topk_weights, combined_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        cuda_stream::set_current(compute_stream);

    // Return values
    return {combined_x, combined_topk_weights, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx,
                             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                             const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                             int num_max_dispatch_tokens_per_rank, std::optional<int> num_experts,
                             bool use_fp8, bool round_scale, bool use_ue8m0,
                             bool async, bool return_recv_hook) {
    EP_HOST_ASSERT(low_latency_mode && "dispatch() requires low-latency mode (low_latency_mode=true)");
    // Tensor checks
    // By default using `ptp128c` FP8 cast
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    const int rank_bound = get_rank_bound(num_experts);
    const int num_max_recv_tokens = rank_bound * num_max_dispatch_tokens_per_rank;
    EP_HOST_ASSERT(num_max_recv_tokens % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

    // Diagnosis tensors
    if (cumulative_local_expert_recv_stats.has_value()) {
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts_per_rank);
    }
    if (dispatch_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == rank_bound);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_topk = static_cast<int>(topk_idx.size(1));

    // Buffer control
    int max_num_experts = max_num_ranks * num_experts_per_rank;
    EPLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, max_num_ranks, max_num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[buffer_idx];
    auto next_buffer = layout.buffers[buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    cudaStream_t compute_stream = cuda_stream::get_current();
    cudaStream_t launch_stream = return_recv_hook ? compute_stream : comm_stream.stream();
    EP_HOST_ASSERT(not (async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate packed tensors
    auto packed_recv_x = torch::empty({num_experts_per_rank, num_max_recv_tokens, hidden},
                                      x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn: torch::kBFloat16));
    auto packed_recv_src_info = torch::empty({num_experts_per_rank, num_max_recv_tokens}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto packed_recv_layout_range = torch::empty({num_experts_per_rank, rank_bound}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto packed_recv_count = torch::empty({num_experts_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // Allocate column-majored scales
    auto packed_recv_x_scales = std::optional<torch::Tensor>();
    void* packed_recv_x_scales_ptr = nullptr;
    if (use_fp8) {
        // TODO: support unaligned cases
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (not use_ue8m0) {
            packed_recv_x_scales = torch::empty({num_experts_per_rank, hidden / 128, num_max_recv_tokens},
                                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
        } else {
            EP_HOST_ASSERT(round_scale);
            packed_recv_x_scales = torch::empty({num_experts_per_rank, hidden / 512, num_max_recv_tokens},
                                                torch::dtype(torch::kInt).device(torch::kCUDA));
        }
        packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=, this](int phases) {
        ep_kernels::dispatch(packed_recv_x.data_ptr(), packed_recv_x_scales_ptr,
                               packed_recv_src_info.data_ptr<int>(), packed_recv_layout_range.data_ptr<int64_t>(),
                               packed_recv_count.data_ptr<int>(),
                               mask_buffer_ptr,
                               cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
                               dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                               buffer.dispatch_rdma_recv_data_buffer, buffer.dispatch_rdma_recv_count_buffer,
                               buffer.dispatch_rdma_send_buffer,
                              x.data_ptr(), topk_idx.data_ptr<topk_idx_t>(),
                               next_clean_meta.first, next_clean_meta.second,
                               num_tokens, hidden, num_max_dispatch_tokens_per_rank,
                               num_topk, rank_bound, num_experts_per_rank, rank,
                               use_fp8, round_scale, use_ue8m0,
                               timeout_cycles,
                               workspace, num_device_sms,
                               launch_stream, phases, gpu_ctx_ptr);
    };
    launcher(return_recv_hook ? EP_SEND_PHASE : (EP_SEND_PHASE | EP_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(EP_RECV_PHASE); };

    // Return values
    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::combine(const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                            const torch::Tensor& src_info, const torch::Tensor& layout_range,
                            const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
                            int num_max_dispatch_tokens_per_rank,
                            bool use_logfmt, bool zero_copy, bool async, bool return_recv_hook,
                            const std::optional<torch::Tensor>& out) {
    EP_HOST_ASSERT(low_latency_mode && "combine() requires low-latency mode (low_latency_mode=true)");
    EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts_per_rank);
    const int rank_bound = layout_range.size(1);
    EP_HOST_ASSERT(rank_bound >= active_rank_bound and rank_bound <= max_num_ranks);

    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts_per_rank);
    EP_HOST_ASSERT(x.size(1) == rank_bound * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(src_info.size(1) == rank_bound * num_max_dispatch_tokens_per_rank);

    if (combine_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and combine_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == rank_bound);
    }

    auto hidden = static_cast<int>(x.size(2));
    auto num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

    // Buffer control
    int max_num_experts = max_num_ranks * num_experts_per_rank;
    EPLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, max_num_ranks, max_num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[buffer_idx];
    auto next_buffer = layout.buffers[buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    cudaStream_t compute_stream = cuda_stream::get_current();
    cudaStream_t launch_stream = return_recv_hook ? compute_stream : comm_stream.stream();
    EP_HOST_ASSERT(not (async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate output tensor
    torch::Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        combined_x = out.value();
    } else {
        combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=, this](int phases) {
        ep_kernels::combine(combined_x.data_ptr(),
                              buffer.combine_rdma_recv_data_buffer, buffer.combine_rdma_recv_flag_buffer,
                              buffer.combine_rdma_send_buffer,
                              x.data_ptr(), topk_idx.data_ptr<topk_idx_t>(), topk_weights.data_ptr<float>(),
                              src_info.data_ptr<int>(), layout_range.data_ptr<int64_t>(),
                              mask_buffer_ptr,
                              combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                              next_clean_meta.first, next_clean_meta.second,
                              num_combined_tokens, hidden, num_max_dispatch_tokens_per_rank,
                              num_topk, rank_bound, num_experts_per_rank, rank,
                             use_logfmt, timeout_cycles,
                              workspace, num_device_sms,
                              launch_stream, phases, zero_copy, gpu_ctx_ptr);
    };
    launcher(return_recv_hook ? EP_SEND_PHASE : (EP_SEND_PHASE | EP_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(EP_RECV_PHASE); };

    // Return values
    return {combined_x, event, recv_hook};
}

torch::Tensor
Buffer::get_next_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden,
                                int rank_bound) const {
    EP_HOST_ASSERT(rank_bound >= active_rank_bound and rank_bound <= max_num_ranks);
    int max_num_experts = max_num_ranks * num_experts_per_rank;
    EPLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, max_num_ranks, max_num_experts);

    auto buffer = layout.buffers[buffer_idx];
    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                            {num_experts_per_rank, rank_bound * num_max_dispatch_tokens_per_rank, hidden},
                            {rank_bound * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                            torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
}

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

void Buffer::update_mask_buffer(int rank_to_mask, bool mask) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < max_num_ranks);
    EP_HOST_ASSERT((rank_to_mask != rank or !mask) && "cannot mask the local rank");
    if (!mask)
        EP_HOST_ASSERT(_is_rank_connected(rank_to_mask) && "cannot unmask an unconnected rank");
    _nixl_ep_memory_views_commit();
    active_ranks[rank_to_mask] = !mask;
    _refresh_active_rank_bound();
    ep_kernels::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, cuda_stream::get_current());
}

void
Buffer::query_mask_buffer(const torch::Tensor &mask_status) const {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == max_num_ranks &&
                   mask_status.scalar_type() == torch::kInt32);

    ep_kernels::query_mask_buffer(mask_buffer_ptr,
                                  max_num_ranks,
                                  mask_status.data_ptr<int>(),
                                  cuda_stream::get_current());
}

void Buffer::clean_mask_buffer() {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    _nixl_ep_memory_views_commit();
    std::vector<int> mask(max_num_ranks, 1);
    for (int rank_id = 0; rank_id < max_num_ranks; ++rank_id) {
        const bool active = _is_rank_connected(rank_id);
        active_ranks[rank_id] = active;
        mask[rank_id] = !active;
    }
    _refresh_active_rank_bound();
    CUDA_CHECK(cudaMemcpyAsync(mask_buffer_ptr, mask.data(),
                               max_num_ranks * sizeof(int), cudaMemcpyHostToDevice,
                               cuda_stream::get_current()));
}

std::string Buffer::get_local_metadata() const {
    EP_HOST_ASSERT(nixl_agent_info != nullptr && nixl_agent_info->agent != nullptr);
    nixl_blob_t metadata_blob;
    nixl_status_t status = nixl_agent_info->agent->getLocalMD(metadata_blob);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to get local metadata, status: " + std::to_string(status));
    }
    return metadata_blob;
}

void Buffer::_nixl_ep_memory_views_stage(void) {
    // TODO: Separate the local memory view so peer updates only rebuild remote views.
    NixlMemoryViews& memory_views = staged_memory_views;
    _nixl_ep_memory_views_destroy(memory_views);
    nixl_remote_dlist_t remote_descs(VRAM_SEG);
    nixl_remote_dlist_t barrier_descs(VRAM_SEG);
    nixl_local_dlist_t local_descs(VRAM_SEG);

    local_descs.addDesc(nixlBlobDesc((uintptr_t)(rdma_buffer_ptr), num_rdma_bytes, get_local_device_id(), ""));
    local_descs.addDesc(nixlBlobDesc((uintptr_t)(sync_count_ptr), max_num_ranks * sizeof(int), get_local_device_id(), ""));

    std::unordered_set<int> remote_set(remote_ranks.begin(), remote_ranks.end());
    for (int r = 0; r < max_num_ranks; r++) {
        std::string remote_agent_name = remote_set.count(r) ? nixl_agent_info->remote_agent_names[r] : nixl_null_agent;
        remote_descs.addDesc(nixlRemoteDesc((uintptr_t)nixl_peer_info[r].rdma_buffer_ptr, num_rdma_bytes, nixl_peer_info[r].device_id, remote_agent_name));
        barrier_descs.addDesc(nixlRemoteDesc((uintptr_t)nixl_peer_info[r].sync_buffer_ptr, max_num_ranks * sizeof(int), nixl_peer_info[r].device_id, remote_agent_name));
    }

    EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(local_descs, memory_views.local, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    if (!remote_ranks.empty()) {
        EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(remote_descs, memory_views.remote, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
        EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(barrier_descs, memory_views.barrier, &nixl_agent_info->extra_params) == NIXL_SUCCESS);

        if (!low_latency_mode && max_num_ranks > NUM_MAX_NVL_PEERS) {
            nixl_remote_dlist_t ht_barrier_descs(VRAM_SEG);
            for (int r = 0; r < max_num_ranks; r++) {
                std::string remote_agent_name = remote_set.count(r) ? nixl_agent_info->remote_agent_names[r] : nixl_null_agent;
                ht_barrier_descs.addDesc(nixlRemoteDesc((uintptr_t)nixl_peer_info[r].ht_barrier_ptr, sizeof(uint64_t), nixl_peer_info[r].device_id, remote_agent_name));
            }
            EP_HOST_ASSERT(nixl_agent_info->agent->prepMemView(ht_barrier_descs, memory_views.ht_barrier, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
        }
    }
}

void Buffer::_nixl_ep_memory_views_destroy(NixlMemoryViews& memory_views) {
    if (memory_views.local) nixl_agent_info->agent->releaseMemView(memory_views.local);
    if (memory_views.remote) nixl_agent_info->agent->releaseMemView(memory_views.remote);
    if (memory_views.barrier) nixl_agent_info->agent->releaseMemView(memory_views.barrier);
    if (memory_views.ht_barrier) nixl_agent_info->agent->releaseMemView(memory_views.ht_barrier);
    memory_views = {};
}

void Buffer::_nixl_ep_memory_views_commit(void) {
    if (!staged_memory_views.local)
        return;
    // Wait for all kernels using the active memory views to finish.
    CUDA_CHECK(cudaDeviceSynchronize());
    std::swap(active_memory_views, staged_memory_views);
    gpu_ctx.local_mvh = active_memory_views.local;
    gpu_ctx.remote_mvh = active_memory_views.remote;
    gpu_ctx.barrier_mvh = active_memory_views.barrier;
    gpu_ctx.ht_barrier_mvh = active_memory_views.ht_barrier;
    CUDA_CHECK(cudaMemcpyAsync(gpu_ctx_ptr, &gpu_ctx, sizeof(gpu_ctx),
                               cudaMemcpyHostToDevice, comm_stream));
    for (int remote_rank : remote_ranks)
        ep_kernels::cache_p2p_ptr(gpu_ctx_ptr, remote_rank, comm_stream);
    // Wait for the new GPU context and cached P2P pointers to be ready.
    CUDA_CHECK(cudaStreamSynchronize(comm_stream));
    _nixl_ep_memory_views_destroy(staged_memory_views);
}

void Buffer::_nixl_ep_init(void) {
    gpu_ctx = {
        .sync_buffer_ptr = sync_buffer_ptr,
        .sync_count_ptr = sync_count_ptr,
        .last_ht_barrier_counter = last_ht_barrier_counter,
        .local_ht_barrier_counter_ptr = local_ht_barrier_counter,
        .rdma_buffer_ptr = rdma_buffer_ptr,
        .p2p_ptrs = nullptr,
        .max_num_ranks = max_num_ranks,
        .num_rdma_ranks = num_rdma_ranks,
        .rank = rank,
    };
    // NIXL_EP_TMA_STAGES=1: one TMA stage per token copy in the HT dispatch (default: 2 when they fit)
    gpu_ctx.tma_stages = std::getenv("NIXL_EP_TMA_STAGES") ? std::atoi(std::getenv("NIXL_EP_TMA_STAGES")) : 2;
    CUDA_CHECK(cudaMalloc(&gpu_ctx.p2p_ptrs, max_num_ranks * sizeof(void*)));
    CUDA_CHECK(cudaMemset(gpu_ctx.p2p_ptrs, 0, max_num_ranks * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&gpu_ctx_ptr, sizeof(gpu_nixl_ctx)));
    CUDA_CHECK(cudaMemcpy(gpu_ctx_ptr, &gpu_ctx, sizeof(gpu_ctx), cudaMemcpyHostToDevice));
}

void Buffer::_nixl_ep_destroy(void) {
    _nixl_ep_memory_views_destroy(staged_memory_views);
    _nixl_ep_memory_views_destroy(active_memory_views);
    if (gpu_ctx.p2p_ptrs != nullptr) {
        cudaFree(gpu_ctx.p2p_ptrs);
        gpu_ctx.p2p_ptrs = nullptr;
    }
    if (gpu_ctx_ptr != nullptr) {
        cudaFree(gpu_ctx_ptr);
        gpu_ctx_ptr = nullptr;
    }
}

void Buffer::_nixl_agent_init() {
    std::string agent_name = std::to_string(rank);
    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    cfg.etcdWatchTimeout = NIXL_ETCD_WATCH_TIMEOUT;
    auto agent = std::make_shared<nixlAgent>(agent_name, cfg);

    // Create UCX backend
    nixl_mem_list_t mems;
    nixl_b_params_t init_params;

    nixl_status_t status = agent->getPluginParams("UCX", mems, init_params);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to get UCX plugin parameters for agent " + agent_name +
                                ", status: " + std::to_string(status));
    }

    const char* num_channels_env = std::getenv("NIXL_EP_NUM_CHANNELS");
    init_params["ucx_num_device_channels"] = num_channels_env ? num_channels_env : "4";
    init_params["ucx_error_handling_mode"] = "none";
    init_params["num_workers"] = std::to_string(1);

    nixlBackendH* ucx_backend = nullptr;
    status = agent->createBackend("UCX", init_params, ucx_backend);
    if (status != NIXL_SUCCESS || !ucx_backend) {
        throw std::runtime_error("Failed to create UCX backend for agent " + agent_name +
                                ", status: " + std::to_string(status));
    }

    nixl_agent_info = std::make_unique<NixlAgentInfo>(agent, ucx_backend, max_num_ranks);
    nixl_agent_info->extra_params.backends.push_back(ucx_backend);
    nixl_agent_info->agent_name = agent_name;

    nixl_agent_info->rdma_reg_descs.clear();
    nixl_agent_info->rdma_reg_descs.addDesc(
        nixlBlobDesc(reinterpret_cast<uintptr_t>(rdma_buffer_ptr), num_rdma_bytes, device_id, ""));

    nixl_agent_info->sync_reg_descs.clear();
    nixl_agent_info->sync_reg_descs.addDesc(
        nixlBlobDesc(reinterpret_cast<uintptr_t>(sync_buffer_ptr), max_num_ranks * sizeof(int), device_id, ""));

    nixl_agent_info->sync_count_reg_descs.clear();
    nixl_agent_info->sync_count_reg_descs.addDesc(
        nixlBlobDesc(reinterpret_cast<uintptr_t>(sync_count_ptr), max_num_ranks * sizeof(int), device_id, ""));
    nixl_agent_info->ht_barrier_reg_descs.clear();

    EP_HOST_ASSERT(agent->registerMem(nixl_agent_info->rdma_reg_descs, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    EP_HOST_ASSERT(agent->registerMem(nixl_agent_info->sync_reg_descs, &nixl_agent_info->extra_params) == NIXL_SUCCESS);
    EP_HOST_ASSERT(agent->registerMem(nixl_agent_info->sync_count_reg_descs, &nixl_agent_info->extra_params) == NIXL_SUCCESS);

    if (local_ht_barrier_counter) {
        nixl_agent_info->ht_barrier_reg_descs.addDesc(
            nixlBlobDesc((uintptr_t)(local_ht_barrier_counter), sizeof(uint64_t), get_local_device_id(), ""));
        EP_HOST_ASSERT(agent->registerMem(nixl_agent_info->ht_barrier_reg_descs) == NIXL_SUCCESS);
    }

    if (getenv("NIXL_ETCD_ENDPOINTS")) {
        status = nixl_agent_info->agent->sendLocalMD();
        if (status != NIXL_SUCCESS) {
            throw std::runtime_error("Failed to send local metadata for agent " +
                                    nixl_agent_info->agent_name + ", status: " + std::to_string(status));
        }
    }
}

void Buffer::_nixl_agents_disconnect(const std::vector<int>& ranks) {
    for (int remote_rank : ranks) {
        EP_HOST_ASSERT(remote_rank != rank);
        EP_HOST_ASSERT(remote_rank >= 0 and remote_rank < max_num_ranks);
        nixl_xfer_dlist_t empty_descs(VRAM_SEG);
        if(nixl_agent_info->agent->checkRemoteMD(nixl_agent_info->remote_agent_names[remote_rank], empty_descs) == NIXL_SUCCESS) {
            nixl_status_t status = nixl_agent_info->agent->invalidateRemoteMD(nixl_agent_info->remote_agent_names[remote_rank]);
            // NIXL watchers might invalidate peer metadata, so we ignore NIXL_ERR_NOT_FOUND errors
            if (status != NIXL_SUCCESS && status != NIXL_ERR_NOT_FOUND) {
                printf("WARNING: rank %d Failed to invalidate remote rank %d metadata for agent %s, status: %d\n",
                    rank, remote_rank, std::to_string(remote_rank).c_str(), status); fflush(stdout);
            }
        }
    }
}

void Buffer::_nixl_agents_peer_info_cleanup(const std::vector<int>& ranks) {
    for (int remote_rank : ranks) {
        nixl_agent_info->wire_up_done[remote_rank] = false;
        // Clear nixl_peer_info for removed ranks (only do this once per rank, on first channel)
        nixl_peer_info[remote_rank] = NixlPeerInfo{};
    }
}

static std::optional<std::vector<nixl_blob_t>> convert_mds(const std::optional<std::vector<pybind11::bytes>>& remote_mds) {
    if (!remote_mds.has_value()) {
        return std::nullopt;
    }
    std::vector<nixl_blob_t> md_blobs;
    md_blobs.reserve(remote_mds->size());
    for (const auto& md_bytes : *remote_mds) {
        md_blobs.push_back(nixl_blob_t(md_bytes));
    }
    return md_blobs;
}

} // namespace nixl_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("NUM_MAX_NVL_PEERS") = NUM_MAX_NVL_PEERS;
    m.doc() = "NIXL_EP: an efficient expert-parallel communication library";
    m.def("get_low_latency_buffer_size_hint", &nixl_ep::get_low_latency_buffer_size_hint);

    pybind11::class_<nixl_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6, py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6, py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &nixl_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &nixl_ep::Config::get_rdma_buffer_size_hint);

    pybind11::class_<nixl_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &nixl_ep::EventHandle::current_stream_wait);

    pybind11::class_<nixl_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, bool, bool, int>())
        .def("update_memory_buffers", &nixl_ep::Buffer::update_memory_buffers)
        .def("dpa_init", &nixl_ep::Buffer::dpa_init)
        .def("barrier", &nixl_ep::Buffer::barrier)
        .def("connect_ranks", [](nixl_ep::Buffer &buffer, const std::vector<int>& remote_ranks, const std::optional<std::vector<pybind11::bytes>>& remote_mds, const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles, bool activate) {
            buffer.connect_ranks(remote_ranks, nixl_ep::convert_mds(remote_mds), all_gathered_handles, activate);
        }, py::arg("remote_ranks"), py::arg("remote_mds") = std::nullopt, py::arg("ipc_handles") = std::vector<std::optional<pybind11::bytearray>>{}, py::arg("activate") = true)
        .def("disconnect_ranks", &nixl_ep::Buffer::disconnect_ranks)
        .def("is_available", &nixl_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &nixl_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &nixl_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &nixl_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &nixl_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &nixl_ep::Buffer::get_local_ipc_handle)
        .def("get_local_buffer_tensor", &nixl_ep::Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &nixl_ep::Buffer::get_comm_stream)
        .def("destroy", &nixl_ep::Buffer::destroy)
        .def("get_dispatch_layout", &nixl_ep::Buffer::get_dispatch_layout)
        .def("dispatch", &nixl_ep::Buffer::dispatch)
        .def("combine", &nixl_ep::Buffer::combine)
        .def("ht_dispatch", &nixl_ep::Buffer::ht_dispatch)
        .def("ht_combine", &nixl_ep::Buffer::ht_combine)
        .def("update_mask_buffer", &nixl_ep::Buffer::update_mask_buffer)
        .def("query_mask_buffer", &nixl_ep::Buffer::query_mask_buffer)
        .def("clean_mask_buffer", &nixl_ep::Buffer::clean_mask_buffer)
        .def("get_next_combine_buffer", &nixl_ep::Buffer::get_next_combine_buffer)
        .def("get_local_metadata", [](const nixl_ep::Buffer &buffer) -> pybind11::bytes {
            return pybind11::bytes(buffer.get_local_metadata());
        });
    m.attr("topk_idx_t") = pybind11::cast(c10::CppTypeToScalarType<nixl_ep::topk_idx_t>::value);
    m.def("is_sm90_compiled", nixl_ep::is_sm90_compiled);
}
