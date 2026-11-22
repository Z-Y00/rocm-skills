---
name: gpu-lock
description: 在多人共用的 GPU 节点上运行任何 GPU 任务之前使用（benchmark、单测、训练、profile、e2e）：用 gpulock 按卡加锁排队、检测没拿锁就上卡的进程，并查看每张卡谁在用、谁在等、历史等待。Use whenever a command will run on a GPU of a shared node, or to see who holds or waits for a GPU.
---

# gpu-lock：共享 GPU 节点按卡加锁

每张卡一把 flock 锁 `<锁目录>/gpu<N>.lock`，和直接用 `flock <锁文件> <命令>` 的人互斥，可以混用。
工具是本目录下的 `gpulock`（单文件，只用 Python ≥ 3.8 标准库）。节点上可以再放一份大家共用，比如放进锁目录。
锁目录取 `$GPULOCK_DIR`；没设时用节点上唯一的共享锁目录（`/var/tmp/gpu-locks`，或已有的 `/var/tmp/<名字>-gpu-locks`），都没有就新建 `/var/tmp/gpu-locks`；找到多个时报错，要设 `$GPULOCK_DIR`。
卡号就是 `HIP_VISIBLE_DEVICES` 的编号。

## 命令（在宿主机上运行，用它包住 docker exec）

```bash
gpulock status                    # 每张卡：谁持锁、谁在排队、卡上有哪些进程；没拿锁就占卡的标 "<< in use without its lock"
gpulock run -g 3 -n "llama e2e r1" -- docker exec -e HIP_VISIBLE_DEVICES=3 my_ctr timeout 1800 bash run.sh
gpulock hold -g 0-3 -t 2h -n "调试"  # 交互式占卡：Ctrl-C 或到时间释放
gpulock history --since 24h       # 谁跑了什么、等了谁多久、有没有被闯入，最后按人汇总
```

`run` 依次做这几件事：
1. 按卡号升序拿锁，卡忙就排队（`status` 里能看到）。
2. 确认卡上没有别人的进程，有就等；默认最多等 30 分钟，超时退出码 75，命令不运行。
3. 运行命令，每 5 秒检查一次有没有别人上了这些卡；有就打 WARNING，结束时汇总。
4. 命令退出后，等它自己的进程离开 GPU，再释放锁。

退出码就是命令自己的退出码；`-g` 支持 `3`、`0,1`、`0-3`、`all`。

每个人的每次 `run` / `hold` 都自动记进 `<锁目录>/history.jsonl`：拿到锁时写一行开始记录，结束时写一行结果（排队多久、等的是谁、运行多久、退出码、有没有人闯入）。
被 kill -9 或碰上节点卡死、重启的运行没有结束记录，`gpulock history` 会把它标成 `NO END RECORD`，可以用来回溯出事时谁在用哪张卡。

## 规则

1. **每条 GPU 命令都包在 `gpulock run` 里**：`-g` 要和命令实际用的卡（`HIP_VISIBLE_DEVICES`）一致；`-n` 写清楚在干什么，别人在 `status` 里能看到。
2. **按"每次运行"拿锁，不要整个批次一直占着**：批处理脚本里给每次运行各包一层 `gpulock run`，这样别人排队时最多只等你一次运行。
   刚放锁时如果有人在排队，`gpulock` 会先让对方拿锁。
3. **多卡一次写全**（如 `-g 0-3`）：gpulock 按卡号升序拿锁。手写 flock 的人也必须按卡号从小到大拿，否则可能死锁。
4. **超时写在 `docker exec` 里面**（如 `docker exec ctr timeout 1800 ...`）：在宿主机上杀掉 docker 客户端，容器里的进程还会继续占卡。
5. **卡忙就等**：不要换到用户没指定的卡上，不要 kill 或 reset 别人的进程。等待可能很长，agent 应把命令放到后台，并告诉用户在等谁（看 `gpulock status`）。
6. **撞卡的结果作废**：输出里出现 `WARNING ... is on a GPU this run locked`，或结尾出现 `INTRUDERS`，说明这次运行和别人的进程撞了，结果不能用于性能对比；把容器名和时间告诉用户。
7. **怎么认自己的进程**：自己的容器从 `docker exec <容器>` / `docker run --name <容器>` 的命令行里识别。用 `docker exec -d`，或用不带 `--name` 的 `docker run` 时，要加 `--own-container <容器>`，否则会被当成闯入者。
8. **可以嵌套**：内层 `gpulock run` 碰到外层已经锁住的卡，不会再拿一次锁；内层要的卡外层没锁，会直接拒绝（防止死锁）。

## Working on a shared remote node (agents)

- **Background + lock + timeout**: `ssh node '~/gpulock run -g 0 -n "<what>" -- docker exec <ctr> timeout 1800 <cmd> > <log> 2>&1; echo EXIT $?'`
  in the background, then poll the log in windows of at most 15 minutes. Tell the user who holds the GPU while waiting.
- **Check the container's mounts first** (`docker inspect -f '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}' <ctr>`):
  a host path that is not mounted fails only after the lock is acquired. Put scripts and data under a mounted directory.
- **Compile-check before queueing**: kernels that can compile without a GPU (e.g. FlyDSL `FLYDSL_COMPILE_ONLY=1`) should be checked in the container with `HIP_VISIBLE_DEVICES=` first.
- **Flaky links**: copy with `rsync --partial --append-verify` in a retry loop and compare `md5sum` on both ends; plain `scp` of tens of MB can drop repeatedly.
- **snap docker** cannot read hidden directories or `/tmp` on the host: use non-hidden paths under `$HOME`. Watch free space on NFS homes.
- **Never `pkill -f <pattern>`** on a shared node: patterns easily match your own ssh session or other users' jobs. Kill by PID, and only your own processes.
- `__pycache__` written by root inside a container cannot be deleted from the host; it is harmless (Python recompiles when the source is newer).
