# Multi-Agent Q-learning Scheduler v2.0

## 项目名称
基于强化学习持续探索的多机器人任务调度算法研究

## 版本说明
当前版本：`v2.0`

相比 `v1.0`，本版本加入了：

1. **动态任务机制**：初始任务为 a-e，动态任务 f 会在训练过程中出现。
2. **事件触发机制**：动态任务出现时，系统触发事件并更新任务集合。
3. **持续探索机制**：事件触发后会提升/保持探索率 epsilon，避免策略过早固定。
4. **可视化增强**：界面显示激活任务、未激活任务、已完成任务、历史最好方案和事件触发日志。
5. **仍然保持 Q-learning 自主学习任务归属**：不会预先指定 A/B/C 分别完成哪些任务。

## 任务设置
机器人：A、B、C  
起点：`(0, 0)`  
终点：`(5, 5)`  
移动耗时：每移动一格 `2s`

初始任务：

| 任务 | 坐标 | 执行时间 | 优先级 | 可执行机器人 |
|---|---:|---:|---:|---|
| a | (0, 2) | 2s | 2 | A |
| b | (2, 4) | 3s | 3 | A, B |
| c | (3, 4) | 4s | 4 | A, B, C |
| d | (1, 3) | 1s | 2 | B, C |
| e | (5, 3) | 5s | 5 | C |

动态任务：

| 任务 | 坐标 | 执行时间 | 优先级 | 可执行机器人 | 触发方式 |
|---|---:|---:|---:|---|---|
| f | (4, 1) | 3s | 6 | A, C | 第 3 个宏动作后自动触发，也可手动触发 |

## 状态空间
程序使用表格型 Q-learning，状态定义为：

```text
S = (positions, active_mask, completed_mask, done_mask, times)
```

含义：

- `positions`：A/B/C 三个机器人的当前位置；
- `active_mask`：当前已经出现的任务集合；
- `completed_mask`：已经完成的任务集合；
- `done_mask`：已经到达终点的机器人集合；
- `times`：A/B/C 各自累计耗时。

## 动作空间
动作定义为：

```text
U = (robot, target)
```

例如：

```text
A -> 任务a
B -> 任务c
C -> 任务e
A -> 终点
```

注意：动作空间只是候选动作，并不是人工提前分配任务。最终谁做什么任务由 Q-learning 通过奖励学习出来。

## 奖励函数
本版本奖励主要由以下部分组成：

```text
完成任务奖励 = 任务优先级 × reward_scale
时间代价 = -makespan 增量
成功奖励 = 所有激活任务完成且 A/B/C 均到终点后给大额奖励
非法动作 = 大额惩罚
```

目标是让系统学习：

```text
在动态任务出现后，仍能完成全部激活任务，并尽量缩短总完成时间 makespan。
```

## 运行方式
本项目只依赖 Python 标准库，无需额外安装第三方包。

```bash
python src/main.py
```

Windows 可以双击：

```text
run.bat
```

Linux/macOS 可执行：

```bash
bash run.sh
```

## Git 版本建议
本版本建议提交为：

```bash
git add .
git commit -m "v2.0: add dynamic task scheduling and continuous exploration"
git tag -a v2.0 -m "v2.0 dynamic task scheduling"
git push
git push origin v2.0
```

## 后续 v3.0 建议
v3.0 可以继续升级为：

1. DQN / Double DQN 替代表格 Q-learning；
2. 加入更多随机任务场景；
3. 加入任务截止时间、机器人故障、障碍物；
4. 与贪心算法、遗传算法进行对比实验；
5. 输出训练曲线和实验统计图。
