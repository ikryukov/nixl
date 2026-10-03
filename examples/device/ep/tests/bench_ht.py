# nixl_ep HT dispatch/combine bench under srun (1 process per GPU, NVL group = NUM_MAX_NVL_PEERS consecutive ranks).
# Env: NIXL_EP_SRC=<examples/device/ep>, MASTER_ADDR; NIXL_EP_DPA=1 [NIXL_EP_DPA_LIB=<libepdpa.so>] for the DPA path.
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.environ["NIXL_EP_SRC"], "tests"))
sys.path.insert(0, os.path.join(os.environ["NIXL_EP_SRC"], "tests", "elastic"))
import torch, torch.distributed as dist
import nixl_ep, store_group
from utils import bench_kineto, create_grouped_scores, inplace_unique, per_token_cast_to_fp8

p = argparse.ArgumentParser()
p.add_argument("--sms", default="12,16,20,24,32")
p.add_argument("--wt", action="store_true")
p.add_argument("--sanity", type=int, default=0)
p.add_argument("--quick", action="store_true")
p.add_argument("--ring", type=int, default=128)
p.add_argument("--rdma-bytes", type=float, default=1e9)
p.add_argument("--tokens", type=int, default=4096)
p.add_argument("--hidden", type=int, default=7168)
p.add_argument("--topk", type=int, default=8)
p.add_argument("--experts", type=int, default=256)
a = p.parse_args()

rank, world, local = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"]), int(os.environ["SLURM_LOCALID"])
torch.cuda.set_device(local)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
dist.init_process_group("nccl", init_method=f"tcp://{os.environ['MASTER_ADDR']}:8361", world_size=world, rank=rank,
                        device_id=torch.device(f"cuda:{local}"))
group = dist.new_group(list(range(world)))
master = store_group.create_master_store(port=9999) if rank == 0 else None
dist.barrier()
store = store_group.create_client_store(master_addr=os.environ["MASTER_ADDR"], port=9999)
max_sms = max(int(s) for s in a.sms.split(","))
buf = nixl_ep.Buffer(rank=rank, low_latency_mode=False, explicitly_destroy=True, group=group, tcp_store_group=store)
buf.update_memory_buffers(num_ranks=world, num_experts_per_rank=max_sms // 2, num_nvl_bytes=int(2e9), num_rdma_bytes=int(a.rdma_bytes) // 128 * 128)
buf.connect_ranks([i for i in range(world) if i != rank])
log = lambda *x: print(*x, flush=True) if rank == 0 else None

# routing as in tests/test_ht.py
torch.manual_seed(rank)
T, H, K, E, nodes = a.tokens, a.hidden, a.topk, a.experts, world // nixl_ep.NUM_MAX_NVL_PEERS
E = max(E, world)
x = torch.randn((T, H), dtype=torch.bfloat16)
x_fp8 = per_token_cast_to_fp8(x)
x_fp8 = (x_fp8[0], x_fp8[1].T.contiguous().T)
scores = torch.randn((T, E), dtype=torch.float32).abs() + 1
gidx = torch.topk(scores.view(T, nodes, -1).amax(dim=-1), k=min(nodes, 4), dim=-1, sorted=False).indices
topk_idx = torch.topk(create_grouped_scores(scores, gidx, nodes), K, dim=-1, largest=True, sorted=False)[1].to(nixl_ep.topk_idx_t)
topk_w = torch.randn((T, K), dtype=torch.float32)
ntpr, ntprr, ntpe, is_in, _ = buf.get_dispatch_layout(topk_idx, E)
rdma_idx = topk_idx // (E // nodes)
rdma_idx.masked_fill_(topk_idx == -1, -1)
inplace_unique(rdma_idx, nodes)
rdma_tokens = rdma_idx.ne(-1).sum().item()

def fenced(f):
    # ponytail: back-to-back cached dispatches hang on cross-node NVL groups (NVL receiver timeout); fence each call
    def g():
        f(); torch.cuda.synchronize(); dist.barrier()
    return g

def cfg(sms, nvl, rdma):
    return nixl_ep.Config(sms, nvl, 512, rdma, a.ring)

if a.sanity:
    sms = int(a.sms.split(",")[0])
    kw = dict(num_tokens_per_rank=ntpr, num_tokens_per_rdma_rank=ntprr, is_token_in_rank=is_in,
              num_tokens_per_expert=ntpe, topk_idx=topk_idx, topk_weights=topk_w, config=cfg(sms, 8, 16))
    for mode in ("fresh", "cached", "cached_nosync"):
        t0 = time.time()
        for i in range(a.sanity):
            if mode == "fresh" or i == 0:
                r = buf.ht_dispatch(x=x, **kw)
                handle = r[4]
                if i == 0:
                    # validation as in tests/test_ht.py: every rank receives exactly its routed token count
                    gbl = ntpr.clone(); dist.all_reduce(gbl)
                    assert r[0].size(0) == gbl[rank].item(), f"recv {r[0].size(0)} != {gbl[rank].item()}"
                    c = buf.ht_combine(x=r[0], handle=handle, config=cfg(sms, 4, 16))[0].float()
                    ref = x.float() * is_in.sum(dim=1, keepdim=True)
                    err = ((c - ref).abs().max() / ref.abs().max().clamp(min=1e-6)).item()
                    assert err < 2e-2, f"combine rel err {err}"
                    log(f"[sanity] validation ok (recv {r[0].size(0)} tokens, combine rel err {err:.1e})")
            else:
                buf.ht_dispatch(x=x, handle=handle, config=cfg(sms, 8, 16))
            if mode != "cached_nosync":
                torch.cuda.synchronize(); dist.barrier()
            if i % 50 == 0:
                log(f"[sanity] {mode} iter {i} ok")
        torch.cuda.synchronize(); dist.barrier()
        log(f"[sanity] {mode} {a.sanity} iters ok in {time.time()-t0:.1f}s")
    sys.exit(0)

for sms in (int(s) for s in a.sms.split(",")):
    recv_x, _, recv_w, _, handle, _ = buf.ht_dispatch(x=x, num_tokens_per_rank=ntpr, num_tokens_per_rdma_rank=ntprr,
                                                      is_token_in_rank=is_in, num_tokens_per_expert=ntpe,
                                                      topk_idx=topk_idx, topk_weights=topk_w, config=cfg(sms, 8, 16))
    for name, cx, f in (("bf16", x, 1.0), ("fp8", x_fp8, (1 + 4 / 128) / 2)):
        best = None
        for nvl in ((8,) if a.quick else (8, 16, 24)):
            for rdma in ((16,) if a.quick else (8, 16, 24, 32)):
                c = cfg(sms, nvl, rdma)
                t, nt = bench_kineto(fenced(lambda: buf.ht_dispatch(x=cx, handle=handle, config=c)), ("dispatch", "notify"), num_tests=10 if a.quick else 30)
                t = torch.tensor([t], device="cuda")
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                if best is None or t.item() < best[0]:
                    best = (t.item(), nvl, rdma)
        bw = rdma_tokens * H * 2 * f / best[0] / 1e9
        log(f"[dispatch] sms {sms:2d} {name}: {best[0]*1e6:8.1f} us (nvl {best[1]}, rdma {best[2]}) RDMA {bw:6.1f} GB/s/rank")
        if a.wt:
            c = cfg(sms, best[1], best[2])
            for _ in range(5):
                fenced(lambda: buf.ht_dispatch(x=cx, handle=handle, config=c))()
            os.environ["NIXL_EP_WT_DUMP"] = "1"
            buf.ht_dispatch(x=cx, handle=handle, config=c)
            os.environ["NIXL_EP_WT_DUMP"] = "0"
            torch.cuda.synchronize(); dist.barrier()
    best = None
    for nvl in ((4,) if a.quick else (2, 4, 6, 8)):
        for rdma in ((16,) if a.quick else (8, 16, 24, 32)):
            c = cfg(sms, nvl, rdma)
            t, nt = bench_kineto(fenced(lambda: buf.ht_combine(x=recv_x, handle=handle, config=c)), ("combine", "notify"), num_tests=10 if a.quick else 30)
            t = torch.tensor([t], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            if best is None or t.item() < best[0]:
                best = (t.item(), nvl, rdma)
    log(f"[combine]  sms {sms:2d} bf16: {best[0]*1e6:8.1f} us (nvl {best[1]}, rdma {best[2]})")

buf.destroy()
dist.barrier()
dist.destroy_process_group()
