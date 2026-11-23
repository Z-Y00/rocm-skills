---
name: shared-gpu-node
description: Practices for a coding agent that runs GPU work on a shared remote node over ssh (benchmarks, tests, training proxies in a docker container) - background runs under gpulock, container mounts, compile checks before queueing, copying over flaky links, snap docker paths, and what never to do to other users' processes. Use together with the gpu-lock skill whenever GPU commands run on a remote multi-user node.
---

# Working on a shared remote GPU node

The locking rules themselves are in the **gpu-lock** skill (`gpulock`, installed from
[lhzhang333/gpulock](https://github.com/lhzhang333/gpulock) by this repo's `install.sh`). These are
the practices around it, learned running kernel campaigns and training proxies over ssh.

- **Background + lock + timeout.** Run
  `ssh node '~/.local/bin/gpulock run -g 0 -n "<what>" -- docker exec <ctr> timeout 900 <cmd> > <log> 2>&1; echo EXIT $?'`
  in the background, then poll the log in windows of at most 15 minutes. Tell the user who holds or
  waits for the GPU while you wait (`gpulock status`). Read every `REMINDER` line in the output.
- **Check the container's mounts first**
  (`docker inspect -f '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}' <ctr>`).
  A host path that is not mounted fails only after the lock is acquired, wasting everyone's queue
  time. Put scripts and data under a mounted directory.
- **Compile-check before queueing.** Kernels that compile without a GPU (e.g. FlyDSL
  `FLYDSL_COMPILE_ONLY=1 FLYDSL_COMPILE_ARCH=<gfx>`) should be checked in the container with
  `HIP_VISIBLE_DEVICES=` first, so the GPU slot is spent on measuring, not on a typo.
- **Keep runs short and separate.** One `gpulock run` per measurement; warm compile caches (a
  second run of a training proxy can take a third of the first's time), so split long series.
- **Flaky links.** Copy with `rsync --partial --append-verify` in a retry loop and compare
  `md5sum` on both ends; plain `scp` of tens of MB can drop repeatedly. A transient
  `Permission denied (publickey)` on a healthy node is worth one retry before assuming lost access.
- **snap docker** cannot read hidden directories or `/tmp` on the host: use non-hidden paths
  under `$HOME`. Watch free space on NFS homes.
- **Never `pkill -f <pattern>`** on a shared node: patterns easily match your own ssh session or
  other users' jobs. Kill by PID, and only your own processes. Never reset a GPU.
- `__pycache__` written by root inside a container cannot be deleted from the host; it is
  harmless (Python recompiles when the source is newer).
- **Results from a shared node are noisy.** Compare A and B in the same session (interleaved
  A B A B), not across sessions; a step-time difference below ~1 % between sessions is noise.
