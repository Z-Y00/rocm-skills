---
name: rccl-sdma
description: Run, verify and measure RCCL copy-engine (SDMA) collectives on AMD Instinct GPUs (MI355X / gfx950), mainly FSDP and distributed-optimizer parameter AllGather in Primus (Megatron, TorchTitan, Flux). Covers how RCCL picks the copy-engine path, the switches and prerequisites, multi-node network settings, how to prove the copy engines are really used, measurement method, and the silent fallbacks that make a run look normal while staying on RCCL kernels. Use when enabling, debugging or benchmarking SDMA / CE collectives, or when an SDMA run shows no change.
---

# RCCL copy-engine (SDMA) collectives

Copy-engine (CE) collectives move collective data with the GPU's SDMA engines instead of RCCL
kernels, so communication uses no compute units (CUs). The goal is to overlap FSDP or
distributed-optimizer AllGather with GEMMs without slowing them.

**The main hazard is silent fallback.** Almost every misconfiguration still trains correctly,
with the collective quietly running on RCCL kernels. Never report an SDMA result until you have
proved the copy engines ran (see [Proving SDMA is active](#proving-sdma-is-active)).

## How RCCL chooses the copy-engine path

RCCL uses its CE path only when all of these hold:

| Requirement | How it is met |
|---|---|
| Zero-CTA policy on the communicator | `NCCL_CTA_POLICY=2` (process-wide), or per group: `ProcessGroupNCCL.Options().config.cta_policy = NCCL_CTA_POLICY_ZERO` |
| Buffers in a registered symmetric window | PyTorch symmetric memory: allocate from `symm_mem.get_mem_pool(device)` and `symm_mem.rendezvous(tensor, group=name)`; FSDP2 does this through `SymmMemAllGather` |
| cuMem | `NCCL_CUMEM_ENABLE=1`, `NCCL_LOCAL_REGISTER=0`, `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true` |
| Host kernel | Linux 6.8 or newer (RCCL cuMem) |
| RCCL build with CE support | Check the image's RCCL version and commit; older builds stay on kernels, especially across nodes |

What RCCL then runs:

- **One node:** CE AllGather. Peer copies through `hipMemcpyBatchAsync`, with synchronization
  through `hipStreamBatchMemOp`.
- **Several nodes:** hierarchical CE AllGather (log: `AllGather [Hierarchical CE] ... RMA proxy + CE`).
  Intra-node copies on the copy engines, inter-node puts through RCCL's RMA proxy. The RMA proxy
  is only enabled when `NCCL_CROSS_NIC` is not 0.
- **ReduceScatter:** only with `RCCL_CE_REDUCESCATTER=1` on builds that have CE ReduceScatter, and
  single-node only.

Synchronization runs as the blit kernel `__amd_rocclr_batchMemOp`, which spins on a CU.
`GPU_STREAMOPS_CP_WAIT=1` does **not** remove it. That flag only turns single
`hipStreamWaitValue` calls on HSA-signal memory into CP barrier packets, while
`hipStreamBatchMemOp` always uses the blit kernel (verified with a HIP reproducer and rocprofv3).

## Switching it on in Primus

| Framework | Switch | Notes |
|---|---|---|
| Megatron, distributed optimizer | `MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma` | Dedicated full-rank zero-CTA `nccl-lazy` group; `param_data` allocated from the symmetric pool. Requires the distributed-optimizer group to be every rank (TP/PP/CP = 1). `MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES` reserves the buffer before fragmentation if allocation fails |
| Megatron gradient ReduceScatter | `MEGATRON_GRAD_REDUCE_BACKEND=rccl_sdma` + `RCCL_CE_REDUCESCATTER=1` | Same group and pool. On RCCL without CE ReduceScatter it runs RCCL's symmetric-memory kernel |
| TorchTitan FSDP2 | `FSDP_ALL_GATHER_BACKEND=rccl_sdma` (older alias `SDMA_ALL_GATHER=1`) | Patch `torchtitan.fsdp.sdma_symm_mem_collectives` attaches `SymmMemAllGather`; hook 06 sets the zero-CTA policy and cuMem. Before AMD-AGI/Primus#1511 the patch never attached (`ceRegWindow=0`) and only `SDMA_ALL_GATHER=1` enabled it |
| Flux FSDP2 (diffusion backend) | `FSDP_ALL_GATHER_BACKEND=rccl_sdma`, `FSDP_SDMA_MODULES=single_blocks`, `FSDP_ALL_GATHER_IMPL=symm\|copy` | With AMD-AGI/Primus#1511: `symm` (default) = RCCL on a zero-CTA `nccl-lazy` twin of the FSDP group, any node count; `copy` = HIP-IPC peer copies, one node only. Before it, the recipe defaulted to `copy` and `symm` failed on PyTorch 2.16 mesh subgroups |

The startup hook `runner/helpers/hooks/06_enable_sdma_all_gather.sh` exports the prerequisites
into the training processes. `primus-cli direct` does not pass arbitrary host variables to
torchrun children, and the container launcher only forwards listed names and some prefixes, so
check that each switch actually reached the training process (`env.txt` or `/proc/<pid>/environ`).

## Multi-node settings

| Variable | Set to | Why |
|---|---|---|
| `NCCL_CROSS_NIC` | `2` on fabrics that route between rails | `base_env.sh` defaults to 0, which disables the RMA proxy, so no hierarchical CE. Rail-isolated fabrics (e.g. Crusoe) need 0 for basic transport, so they cannot run multi-node CE at all |
| `NCCL_IB_GID_INDEX` | First routable RoCE v2 GID (1 on DCCS) | Primus defaults to 3, which was not routable there |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `IP_INTERFACE` | The out-of-band Ethernet interface | Otherwise derived from the first `hostname -I` address, which can be an RDMA rail and hangs Gloo |
| `NCCL_IB_HCA` | The RDMA NICs, e.g. `ionic_0,...,ionic_7` | Exclude non-RDMA HCAs the container cannot drive |

On AMD Pensando (ionic) hosts, mounting the host's libionic/libibverbs into the container
segfaulted RCCL in one image while the image's own libraries worked; try both.

## Proving SDMA is active

Check at least two of these on every new setup:

1. **RCCL log** (`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,TUNING`): `Taking CE collective path for AllGather`
   or `AllGather [Hierarchical CE]`. `AG CE-registered disqualified ... ceRegWindow=0` means the
   buffer was not registered: the attach or allocation did not happen.
2. **Profiler trace:** AllGather shows `Memcpy` records plus `__amd_rocclr_batchMemOp` kernels and
   no `ncclDevKernel_*AllGather*`. PyTorch labels the peer copies `Memcpy DtoH`; that is a
   mislabel, they are GPU-to-GPU. Records can also be duplicated per batch, so do not sum their
   durations as copy time.
3. **KFD counters:** `/sys/class/kfd/kfd/proc/<pid>/sdma_<gpu_id>` (per process, microseconds)
   grows in SDMA mode and stays near 0 on kernels. Sum over processes only on an exclusive node.
4. **Framework log:** a patch logging `installed` is not proof (the TorchTitan patch did that while
   never attaching).

For a quick correctness and bandwidth check, run an AllGather over registered buffers
(`symm_mem` pool + `rendezvous`) on a zero-CTA group across the target nodes, compare every slice,
and report bus bandwidth `bytes * (n-1)/n / time`.

## Measuring

- Same nodes for SDMA on and off, one after the other, SDMA first.
- Discard about 10 minutes of warm-up, then average about 10 minutes of steps.
- Exclude profiled steps from throughput.
- Make sure every GPU is idle before each mode: Slurm can report a node as idle while another
  user's stray process holds a GPU.
- Report exposed communication from a trace: collective GPU time not covered by compute kernels.
  With little exposed communication, SDMA cannot help step time, so check this first.
- Inter-node bandwidth on a shared fabric can drop for minutes (we saw 65 vs 360 GB/s on the same
  node pair minutes apart). Run a short kernel AllGather bandwidth check before a long comparison
  and gate on it.

## Debugging hangs

- PyTorch's watchdog timeout (`TORCH_NCCL_*`, TorchTitan `comm.train_timeout_seconds`) only fires
  if the watchdog thread is free. In the SDMA-sync hang it was blocked on the communicator mutex
  held by the stuck launch, so no flight-recorder dump was written.
- Flight-recorder dumps go to `<job.dump_folder>/comm_traces/rank_<N>` in TorchTitan; point
  `job.dump_folder` at a mounted directory or they vanish with the container.
- For a live hang: `pip install py-spy` and `apt-get install gdb` in the running container, then
  `py-spy dump --native --pid` for every rank and `gdb -p <pid> -batch -ex "thread apply all bt"`
  for one rank per node. py-spy only shows threads that run Python; RCCL proxy and HIP threads
  need gdb.

## Known RCCL issues (2026-11)

- **SDMA-gated synchronization** (ROCm/rocm-systems#14697, `RCCL_CE_SDMA_SYNC=1`, needs
  `GPU_STREAMOPS_CP_WAIT=1`): standalone AllGather correct and faster, but TorchTitan FSDP2 hung
  after about 6 steps in a host-side deadlock (`ncclRmaProxyEnqueueDesc` spinning on a full RMA
  proxy queue on every rank).
- **Hierarchical CE Ring** (ROCm/rocm-systems#14987): fixes the Ring hang and auto-selects Ring on
  3+ nodes at 2 MiB or more per rank; on 2 nodes it always picks Direct.

## Building RCCL from a PR into the training image

Use a Dockerfile `FROM` the training image. In TheRock-based images the ROCm SDK is a pip package:
`ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel`, compiler
`$ROCM_PATH/bin/amdclang++` (RCCL refuses hipcc), `libhsakmt.a` in `$ROCM_PATH/lib`. Build only the
target arch (`-DGPU_TARGETS=gfx950 -DBUILD_TESTS=OFF`, about a minute on 256 cores). Then copy
`librccl.so` over every `librccl.so.1` under `_rocm_sdk_*` (PyTorch loads the one in
`_rocm_sdk_libraries/lib`) and check `torch.cuda.nccl.version()` and a PR-specific string in the
library. Save the image with `docker save` to shared storage; `docker load -i` takes about 2.5 minutes for 41 GB.
