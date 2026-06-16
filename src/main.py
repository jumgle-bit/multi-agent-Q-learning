"""
Multi-Agent Q-learning Task Scheduler v1.0

Static first version:
- 3 agents: A, B, C
- 5 tasks: a, b, c, d, e
- All agents start at (0, 0)
- Goal is (5, 5)
- Each grid movement costs 2 seconds
- Each task has location, duration, and capability constraints
- Q-learning learns which agent should do which task and when to go to goal
- Episode succeeds only when all 5 tasks are completed and all agents reach goal

Run:
    python src/main.py
"""

from __future__ import annotations

import csv
import random
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tkinter import scrolledtext, ttk, messagebox
from typing import Dict, List, Optional, Tuple

Position = Tuple[int, int]
State = Tuple[Tuple[Position, Position, Position], int, int, Tuple[int, int, int]]


@dataclass(frozen=True)
class Task:
    name: str
    pos: Position
    duration: int
    capable: Tuple[str, ...]


@dataclass
class StepRecord:
    state: State
    action_index: int
    next_state: State
    reward: float
    done: bool
    info: dict
    old_q: float
    next_max_q: float
    target: float
    td_error: float
    new_q: float
    choose_reason: str


class MultiAgentTaskEnv:
    """Static multi-agent task scheduling environment."""

    def __init__(self) -> None:
        self.grid_size = 6  # coordinates 0..5
        self.start_pos: Position = (0, 0)
        self.goal_pos: Position = (5, 5)
        self.move_time_per_grid = 2

        self.agents = ("A", "B", "C")
        self.agent_index = {ag: i for i, ag in enumerate(self.agents)}

        self.tasks: Dict[str, Task] = {
            "a": Task("a", (0, 2), 2, ("A",)),
            "b": Task("b", (2, 4), 3, ("A", "B")),
            "c": Task("c", (3, 4), 4, ("A", "B", "C")),
            "d": Task("d", (1, 3), 1, ("B", "C")),
            "e": Task("e", (5, 3), 5, ("C",)),
        }
        self.task_names = tuple(self.tasks.keys())
        self.task_index = {name: i for i, name in enumerate(self.task_names)}

        self.all_tasks_mask = (1 << len(self.task_names)) - 1
        self.all_agents_done_mask = (1 << len(self.agents)) - 1

        # Candidate macro-actions. This is NOT a fixed assignment.
        # Q-learning chooses among these actions according to Q-values.
        self.targets = self.task_names + ("GOAL",)
        self.actions: List[Tuple[str, str]] = []
        for ag in self.agents:
            for target in self.targets:
                self.actions.append((ag, target))

    def start_state(self) -> State:
        positions = tuple(self.start_pos for _ in self.agents)  # type: ignore[assignment]
        completed_mask = 0
        done_mask = 0
        times = tuple(0 for _ in self.agents)  # type: ignore[assignment]
        return positions, completed_mask, done_mask, times

    def manhattan(self, p1: Position, p2: Position) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def travel_time(self, p1: Position, p2: Position) -> int:
        return self.manhattan(p1, p2) * self.move_time_per_grid

    def task_completed(self, completed_mask: int, task_name: str) -> bool:
        return (completed_mask & (1 << self.task_index[task_name])) != 0

    def set_task_completed(self, completed_mask: int, task_name: str) -> int:
        return completed_mask | (1 << self.task_index[task_name])

    def agent_done(self, done_mask: int, agent_name: str) -> bool:
        return (done_mask & (1 << self.agent_index[agent_name])) != 0

    def set_agent_done(self, done_mask: int, agent_name: str) -> int:
        return done_mask | (1 << self.agent_index[agent_name])

    def completed_task_names(self, completed_mask: int) -> List[str]:
        return [name for name in self.task_names if self.task_completed(completed_mask, name)]

    def remaining_task_names(self, completed_mask: int) -> List[str]:
        return [name for name in self.task_names if not self.task_completed(completed_mask, name)]

    def makespan(self, times: Tuple[int, int, int]) -> int:
        return max(times)

    def is_terminal_success(self, state: State) -> bool:
        _positions, completed_mask, done_mask, _times = state
        return completed_mask == self.all_tasks_mask and done_mask == self.all_agents_done_mask

    def remaining_tasks_coverable_after_done(self, completed_mask: int, new_done_mask: int) -> bool:
        """Prevent a robot from going to goal if remaining tasks would become impossible."""
        for task_name in self.remaining_task_names(completed_mask):
            capable_agents = self.tasks[task_name].capable
            if not any(not self.agent_done(new_done_mask, ag) for ag in capable_agents):
                return False
        return True

    def legal_actions(self, state: State) -> List[int]:
        positions, completed_mask, done_mask, _times = state
        if self.is_terminal_success(state):
            return []

        legal: List[int] = []
        for idx, (ag, target) in enumerate(self.actions):
            if self.agent_done(done_mask, ag):
                continue

            if target == "GOAL":
                new_done_mask = self.set_agent_done(done_mask, ag)
                if self.remaining_tasks_coverable_after_done(completed_mask, new_done_mask):
                    legal.append(idx)
                continue

            task = self.tasks[target]
            if self.task_completed(completed_mask, target):
                continue
            if ag not in task.capable:
                continue

            legal.append(idx)

        return legal

    def grid_path(self, start: Position, end: Position) -> List[Position]:
        """Simple Manhattan path for visualization: row direction first, then column."""
        path = [start]
        r, c = start
        er, ec = end
        while r != er:
            r += 1 if er > r else -1
            path.append((r, c))
        while c != ec:
            c += 1 if ec > c else -1
            path.append((r, c))
        return path

    def transition(self, state: State, action_index: int) -> Tuple[State, float, bool, dict]:
        positions, completed_mask, done_mask, times = state
        old_makespan = self.makespan(times)

        if action_index not in self.legal_actions(state):
            info = {
                "valid": False,
                "reason": "非法动作：能力不匹配、任务已完成，或会导致剩余任务无人可做",
                "move_time": 0,
                "task_time": 0,
                "action_time": 0,
                "old_makespan": old_makespan,
                "new_makespan": old_makespan,
                "delta_makespan": 0,
                "path": [],
                "agent": None,
                "target": None,
            }
            return state, -100.0, False, info

        ag, target = self.actions[action_index]
        ag_i = self.agent_index[ag]
        old_pos = positions[ag_i]

        new_positions = list(positions)
        new_times = list(times)
        new_completed_mask = completed_mask
        new_done_mask = done_mask

        if target == "GOAL":
            target_pos = self.goal_pos
            move_t = self.travel_time(old_pos, target_pos)
            task_t = 0
            action_t = move_t
            new_positions[ag_i] = target_pos
            new_times[ag_i] += action_t
            new_done_mask = self.set_agent_done(done_mask, ag)
            reason = f"{ag} 去终点：从 {old_pos} 到 {target_pos}，移动耗时 {move_t}s"
        else:
            task = self.tasks[target]
            target_pos = task.pos
            move_t = self.travel_time(old_pos, target_pos)
            task_t = task.duration
            action_t = move_t + task_t
            new_positions[ag_i] = target_pos
            new_times[ag_i] += action_t
            new_completed_mask = self.set_task_completed(completed_mask, target)
            reason = (
                f"{ag} 选择完成任务 {target}：从 {old_pos} 到 {target_pos}，"
                f"移动 {move_t}s，执行任务 {task_t}s"
            )

        new_state: State = (tuple(new_positions), new_completed_mask, new_done_mask, tuple(new_times))  # type: ignore[arg-type]
        new_makespan = self.makespan(tuple(new_times))  # type: ignore[arg-type]
        delta_makespan = new_makespan - old_makespan

        # Basic objective: minimize total completion time.
        reward = -float(delta_makespan)
        done = False

        if target != "GOAL":
            reward += 2.0  # small shaping reward for completing a task

        if self.is_terminal_success(new_state):
            done = True
            reward += 100.0
            reason += "；五个任务全部完成，且 A/B/C 均到达终点"

        info = {
            "valid": True,
            "reason": reason,
            "move_time": move_t,
            "task_time": task_t,
            "action_time": action_t,
            "old_makespan": old_makespan,
            "new_makespan": new_makespan,
            "delta_makespan": delta_makespan,
            "path": self.grid_path(old_pos, target_pos),
            "agent": ag,
            "target": target,
        }
        return new_state, reward, done, info

    def format_action(self, action_index: int) -> str:
        ag, target = self.actions[action_index]
        return f"{ag} → 终点" if target == "GOAL" else f"{ag} → 任务{target}"

    def format_state_short(self, state: State) -> str:
        positions, completed_mask, done_mask, times = state
        completed = self.completed_task_names(completed_mask)
        remaining = self.remaining_task_names(completed_mask)
        done_agents = [ag for ag in self.agents if self.agent_done(done_mask, ag)]
        return f"pos={positions}, 已完成={completed}, 剩余={remaining}, 到终点={done_agents}, t={times}"


class QLearningScheduler:
    def __init__(self, env: MultiAgentTaskEnv) -> None:
        self.env = env
        self.Q = defaultdict(lambda: [0.0 for _ in range(len(env.actions))])

    def choose_action(self, state: State, epsilon: float) -> Tuple[Optional[int], str]:
        legal = self.env.legal_actions(state)
        if not legal:
            return None, "无合法动作"

        if random.random() < epsilon:
            return random.choice(legal), "探索：随机选择一个合法动作"

        q_values = self.Q[state]
        best_q = max(q_values[i] for i in legal)
        best_actions = [i for i in legal if abs(q_values[i] - best_q) < 1e-12]
        chosen = random.choice(best_actions)
        if len(best_actions) > 1:
            return chosen, "利用：在并列最大Q值动作中随机选择"
        return chosen, "利用：选择当前最大Q值动作"

    def update(
        self,
        state: State,
        action_index: int,
        reward: float,
        next_state: State,
        done: bool,
        alpha: float,
        gamma: float,
    ) -> Tuple[float, float, float, float, float]:
        old_q = self.Q[state][action_index]
        next_legal = self.env.legal_actions(next_state)
        next_max_q = 0.0 if done or not next_legal else max(self.Q[next_state][i] for i in next_legal)
        target = reward + gamma * next_max_q
        td_error = target - old_q
        new_q = old_q + alpha * td_error
        self.Q[state][action_index] = new_q
        return old_q, next_max_q, target, td_error, new_q


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("v1.0 多机器人 Q-learning 任务调度")
        self.root.geometry("1320x850")

        self.env = MultiAgentTaskEnv()
        self.agent = QLearningScheduler(self.env)

        # RL parameters
        self.alpha = tk.DoubleVar(value=0.25)
        self.gamma = tk.DoubleVar(value=0.95)
        self.epsilon = tk.DoubleVar(value=1.00)
        self.target_epsilon = tk.DoubleVar(value=0.05)
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995

        self.state = self.env.start_state()
        self.display_state: Optional[State] = None
        self.done = False
        self.episode = 1
        self.total_steps = 0
        self.episode_steps = 0
        self.max_steps_per_episode = 10

        self.last_transition: Optional[StepRecord] = None
        self.last_move_path: List[Position] = []
        self.current_episode_plan: List[StepRecord] = []
        self.best_solution: Optional[dict] = None
        self.episode_results: List[dict] = []

        self.auto_running = False
        self.demo_running = False
        self.demo_plan: List[StepRecord] = []
        self.demo_index = 0

        self.build_ui()
        self.draw_grid()
        self.update_action_table()
        self.update_status("v1.0 已启动：静态 3机器人-5任务调度，Q-learning 自主学习任务归属。")

    # ---------------------- core train logic ----------------------
    def current_display_state(self) -> State:
        return self.display_state if self.display_state is not None else self.state

    def train_one_step(self) -> None:
        if self.demo_running:
            return
        self.display_state = None

        if self.done or self.episode_steps >= self.max_steps_per_episode:
            self.new_episode(clear_log=False)

        state = self.state
        action_index, choose_reason = self.agent.choose_action(state, self.epsilon.get())
        if action_index is None:
            self.finish_episode("失败：无合法动作")
            return

        next_state, reward, done, info = self.env.transition(state, action_index)
        old_q, next_max_q, target, td_error, new_q = self.agent.update(
            state,
            action_index,
            reward,
            next_state,
            done,
            self.alpha.get(),
            self.gamma.get(),
        )

        record = StepRecord(
            state=state,
            action_index=action_index,
            next_state=next_state,
            reward=reward,
            done=done,
            info=info,
            old_q=old_q,
            next_max_q=next_max_q,
            target=target,
            td_error=td_error,
            new_q=new_q,
            choose_reason=choose_reason,
        )

        self.state = next_state
        self.done = done
        self.total_steps += 1
        self.episode_steps += 1
        self.last_transition = record
        self.last_move_path = info.get("path", [])
        self.current_episode_plan.append(record)

        self.log_transition(record)

        if done:
            final_time = self.env.makespan(next_state[3])
            self.record_success(final_time)
            self.finish_episode(f"成功：五个任务全部完成，总时间 {final_time}s")
        elif self.episode_steps >= self.max_steps_per_episode:
            self.finish_episode("达到本回合最大动作数，未完成则结束")

        self.draw_grid()
        self.update_action_table()
        self.update_status()

    def finish_episode(self, reason: str) -> None:
        old_eps = self.epsilon.get()
        new_eps = max(self.epsilon_min, old_eps * self.epsilon_decay)
        self.epsilon.set(round(new_eps, 5))
        self.done = True

        final_time = self.env.makespan(self.state[3])
        success = self.env.is_terminal_success(self.state)
        self.episode_results.append(
            {"episode": self.episode, "success": success, "time": final_time, "epsilon": new_eps, "reason": reason}
        )

        self.log(f"\n第 {self.episode} 回合结束：{reason}。ε 从 {old_eps:.5f} 衰减到 {new_eps:.5f}\n")

    def new_episode(self, clear_log: bool = True) -> None:
        self.state = self.env.start_state()
        self.display_state = None
        self.done = False
        self.episode += 1
        self.episode_steps = 0
        self.last_transition = None
        self.last_move_path = []
        self.current_episode_plan = []
        if clear_log:
            self.log(f"\n新开第 {self.episode} 回合：A、B、C 回到 (0,0)，Q 表保留。\n")
        self.draw_grid()
        self.update_action_table()
        self.update_status()

    def record_success(self, final_time: int) -> None:
        plan_copy = list(self.current_episode_plan)
        if self.best_solution is None or final_time < self.best_solution["time"]:
            self.best_solution = {"time": final_time, "episode": self.episode, "plan": plan_copy}
            self.log(f"\n发现历史最好方案：总时间 {final_time}s，来自第 {self.episode} 回合。\n")
            self.log(self.solution_summary_text(plan_copy, final_time) + "\n")

    def reset_all(self) -> None:
        self.stop_auto()
        self.agent = QLearningScheduler(self.env)
        self.state = self.env.start_state()
        self.display_state = None
        self.done = False
        self.episode = 1
        self.total_steps = 0
        self.episode_steps = 0
        self.last_transition = None
        self.last_move_path = []
        self.current_episode_plan = []
        self.best_solution = None
        self.episode_results = []
        self.epsilon.set(1.00)
        self.target_epsilon.set(0.05)
        self.log_box.delete("1.0", tk.END)
        self.log("已重置：Q表清空，训练重新开始。\n")
        self.draw_grid()
        self.update_action_table()
        self.update_status()

    # ---------------------- auto train ----------------------
    def auto_train_episodes(self, n: int) -> None:
        if self.demo_running:
            return
        self.auto_running = True
        self.auto_episodes_left = n
        if self.done:
            self.new_episode(clear_log=False)
        self.auto_episode_loop()

    def auto_episode_loop(self) -> None:
        if not self.auto_running:
            return
        if self.auto_episodes_left <= 0:
            self.auto_running = False
            self.update_status("自动训练完成。")
            return

        self.train_one_step()
        if self.done:
            self.auto_episodes_left -= 1
            self.new_episode(clear_log=False)
        self.root.after(2, self.auto_episode_loop)

    def auto_train_until_epsilon(self) -> None:
        if self.demo_running:
            return
        target = self.target_epsilon.get()
        if self.epsilon.get() < target:
            self.log(f"\n当前 ε={self.epsilon.get():.5f}，已经小于目标阈值 {target:.5f}。\n")
            return
        self.auto_running = True
        self.log(f"\n开始自动训练：直到 ε < {target:.5f}。\n")
        if self.done:
            self.new_episode(clear_log=False)
        self.auto_until_epsilon_loop()

    def auto_until_epsilon_loop(self) -> None:
        if not self.auto_running:
            return
        target = self.target_epsilon.get()
        if self.epsilon.get() < target:
            self.auto_running = False
            self.log(f"\n训练结束：当前 ε={self.epsilon.get():.5f}，已经小于 {target:.5f}。\n")
            self.update_status("训练到 ε 阈值已完成。")
            return
        self.train_one_step()
        if self.done:
            self.new_episode(clear_log=False)
        self.root.after(2, self.auto_until_epsilon_loop)

    def stop_auto(self) -> None:
        self.auto_running = False
        self.demo_running = False

    # ---------------------- demo ----------------------
    def greedy_plan_from_q(self, max_steps: int = 20) -> List[StepRecord]:
        state = self.env.start_state()
        plan: List[StepRecord] = []
        visited = set()
        for _ in range(max_steps):
            if self.env.is_terminal_success(state):
                break
            legal = self.env.legal_actions(state)
            if not legal:
                break
            q_values = self.agent.Q[state]
            best_q = max(q_values[i] for i in legal)
            best_actions = [i for i in legal if abs(q_values[i] - best_q) < 1e-12]
            action_index = random.choice(best_actions)
            key = (state, action_index)
            if key in visited:
                break
            visited.add(key)
            next_state, reward, done, info = self.env.transition(state, action_index)
            record = StepRecord(
                state=state,
                action_index=action_index,
                next_state=next_state,
                reward=reward,
                done=done,
                info=info,
                old_q=self.agent.Q[state][action_index],
                next_max_q=0.0,
                target=0.0,
                td_error=0.0,
                new_q=self.agent.Q[state][action_index],
                choose_reason="演示：按当前Q表贪心选择",
            )
            plan.append(record)
            state = next_state
            if done:
                break
        return plan

    def demo_current_policy(self) -> None:
        self.stop_auto()
        plan = self.greedy_plan_from_q()
        if not plan:
            self.log("\n当前策略还没有形成可演示方案，可以先训练几百回合。\n")
            return
        final_time = self.env.makespan(plan[-1].next_state[3])
        self.log("\n演示当前 Q 表策略：\n")
        self.log(self.solution_summary_text(plan, final_time) + "\n")
        self.start_demo(plan)

    def demo_best_solution(self) -> None:
        self.stop_auto()
        if self.best_solution is None:
            self.log("\n目前还没有成功方案。可以先点【训练1000回合】或【训练到ε阈值】。\n")
            return
        self.log(
            f"\n演示历史最好方案：总时间 {self.best_solution['time']}s，"
            f"来自第 {self.best_solution['episode']} 回合。\n"
        )
        self.log(self.solution_summary_text(self.best_solution["plan"], self.best_solution["time"]) + "\n")
        self.start_demo(self.best_solution["plan"])

    def start_demo(self, plan: List[StepRecord]) -> None:
        self.demo_plan = plan
        self.demo_index = 0
        self.demo_running = True
        self.display_state = self.env.start_state()
        self.last_move_path = []
        self.demo_loop()

    def demo_loop(self) -> None:
        if not self.demo_running:
            return
        if self.demo_index >= len(self.demo_plan):
            self.demo_running = False
            self.update_status("方案演示结束。")
            return
        record = self.demo_plan[self.demo_index]
        self.display_state = record.next_state
        self.last_move_path = record.info.get("path", [])
        self.last_transition = record
        self.demo_index += 1
        self.draw_grid()
        self.update_action_table()
        self.update_status("正在演示方案，不更新Q表。")
        self.root.after(900, self.demo_loop)

    # ---------------------- summaries and export ----------------------
    def solution_summary_text(self, plan: List[StepRecord], final_time: int) -> str:
        owner = {task_name: None for task_name in self.env.task_names}
        lines_by_agent = {ag: [] for ag in self.env.agents}
        time_by_agent = {ag: 0 for ag in self.env.agents}

        for item in plan:
            ag, target = self.env.actions[item.action_index]
            if target != "GOAL":
                owner[target] = ag
                lines_by_agent[ag].append(f"任务{target}")
            else:
                lines_by_agent[ag].append("终点")
            time_by_agent[ag] = item.next_state[3][self.env.agent_index[ag]]

        lines = [f"学习得到的任务归属：{owner}"]
        for ag in self.env.agents:
            route = " → ".join(lines_by_agent[ag]) if lines_by_agent[ag] else "未行动"
            lines.append(f"{ag}: {route}，累计耗时 {time_by_agent[ag]}s")
        lines.append(f"总完成时间 makespan = {final_time}s")
        return "\n".join(lines)

    def export_results(self) -> None:
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        csv_path = out_dir / "training_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "success", "time", "epsilon", "reason"])
            writer.writeheader()
            writer.writerows(self.episode_results)

        txt_path = out_dir / "best_solution.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            if self.best_solution is None:
                f.write("暂无成功方案。\n")
            else:
                f.write(f"历史最好方案：第 {self.best_solution['episode']} 回合，总时间 {self.best_solution['time']}s\n")
                f.write(self.solution_summary_text(self.best_solution["plan"], self.best_solution["time"]))
                f.write("\n\n详细步骤：\n")
                for i, item in enumerate(self.best_solution["plan"], start=1):
                    f.write(f"{i}. {self.env.format_action(item.action_index)} | {item.info['reason']}\n")
        messagebox.showinfo("导出完成", f"已导出到：\n{csv_path}\n{txt_path}")

    # ---------------------- UI ----------------------
    def build_ui(self) -> None:
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        right = tk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0))

        self.canvas_size = 600
        self.cell = self.canvas_size // self.env.grid_size
        self.canvas = tk.Canvas(left, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        legend = (
            "v1.0 静态任务版本：A/B/C 初始都在 (0,0)，终点是 (5,5)。\n"
            "任务 a,b,c,d,e 必须全部完成；圆圈内显示：任务名/执行时间/可执行机器人。\n"
            "Q-learning 不预设分配方案，通过试错学习谁去做什么任务。\n"
            "总时间 = max(tA,tB,tC)，因为机器人并行执行。"
        )
        tk.Label(left, text=legend, justify=tk.LEFT, wraplength=600).pack(pady=8)

        param_frame = ttk.LabelFrame(right, text="参数设置")
        param_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(param_frame, text="学习率 α").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(param_frame, textvariable=self.alpha, width=8).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(param_frame, text="折扣因子 γ").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Entry(param_frame, textvariable=self.gamma, width=8).grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(param_frame, text="探索率 ε").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ttk.Entry(param_frame, textvariable=self.epsilon, width=8).grid(row=0, column=5, padx=5, pady=5)
        ttk.Label(param_frame, text="训练到 ε <").grid(row=0, column=6, padx=5, pady=5, sticky="w")
        ttk.Entry(param_frame, textvariable=self.target_epsilon, width=8).grid(row=0, column=7, padx=5, pady=5)

        button_frame = ttk.LabelFrame(right, text="训练控制")
        button_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(button_frame, text="单步训练", command=self.train_one_step).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="训练100回合", command=lambda: self.auto_train_episodes(100)).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(button_frame, text="训练1000回合", command=lambda: self.auto_train_episodes(1000)).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(button_frame, text="训练到ε阈值", command=self.auto_train_until_epsilon).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(button_frame, text="演示当前策略", command=self.demo_current_policy).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="演示历史最好", command=self.demo_best_solution).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(button_frame, text="导出结果", command=self.export_results).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(button_frame, text="停止", command=self.stop_auto).grid(row=1, column=3, padx=5, pady=5)
        ttk.Button(button_frame, text="重置", command=self.reset_all).grid(row=1, column=4, padx=5, pady=5)

        status_frame = ttk.LabelFrame(right, text="当前信息")
        status_frame.pack(fill=tk.X, pady=(0, 8))
        self.status_label = tk.Label(status_frame, text="", justify=tk.LEFT, anchor="w")
        self.status_label.pack(fill=tk.X, padx=8, pady=6)

        table_frame = ttk.LabelFrame(right, text="当前状态下可选择动作的 Q 值")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        columns = ("rank", "action", "q", "cost", "new_time", "reward")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=9)
        for col, title, width in [
            ("rank", "序号", 50),
            ("action", "动作", 130),
            ("q", "Q值", 90),
            ("cost", "动作耗时", 90),
            ("new_time", "动作后总时间", 110),
            ("reward", "奖励", 90),
        ]:
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(right, text="训练日志与公式代入")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_box = scrolledtext.ScrolledText(log_frame, height=13, wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def draw_grid(self) -> None:
        self.canvas.delete("all")
        state = self.current_display_state()
        positions, completed_mask, done_mask, _times = state

        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                x1 = c * self.cell
                y1 = r * self.cell
                x2 = x1 + self.cell
                y2 = y1 + self.cell
                pos = (r, c)
                fill = "#ffffff"
                if pos == self.env.start_pos:
                    fill = "#e8f3ff"
                if pos == self.env.goal_pos:
                    fill = "#d9ffd9"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="#999999")
                self.canvas.create_text(x1 + 18, y1 + 14, text=f"{r},{c}", font=("Arial", 8), fill="#555555")

        # goal
        gx, gy = self.cell_center(self.env.goal_pos)
        self.canvas.create_rectangle(gx - 30, gy - 20, gx + 30, gy + 20, fill="#6ccf6c", outline="")
        self.canvas.create_text(gx, gy, text="终点", font=("Arial", 12, "bold"), fill="white")

        # tasks
        for task_name in self.env.task_names:
            task = self.env.tasks[task_name]
            cx, cy = self.cell_center(task.pos)
            completed = self.env.task_completed(completed_mask, task_name)
            color = "#bdbdbd" if completed else "#ffb74d"
            text_color = "#ffffff" if completed else "#000000"
            cap = "".join(task.capable)
            label = f"{task_name}\n{task.duration}s\n{cap}"
            if completed:
                label = f"{task_name}✓\n{task.duration}s\n{cap}"
            self.canvas.create_oval(cx - 25, cy - 25, cx + 25, cy + 25, fill=color, outline="#7a4a00", width=2)
            self.canvas.create_text(cx, cy, text=label, font=("Arial", 9, "bold"), fill=text_color)

        # last path
        if self.last_move_path and len(self.last_move_path) >= 2:
            pts: List[float] = []
            for p in self.last_move_path:
                pts.extend(self.cell_center(p))
            self.canvas.create_line(*pts, fill="#ff5252", width=4, arrow=tk.LAST)

        offsets = {"A": (-18, -18), "B": (18, -18), "C": (0, 18)}
        colors = {"A": "#e53935", "B": "#1e88e5", "C": "#43a047"}
        for ag in self.env.agents:
            i = self.env.agent_index[ag]
            pos = positions[i]
            cx, cy = self.cell_center(pos)
            ox, oy = offsets[ag]
            color = colors[ag]
            if self.env.agent_done(done_mask, ag):
                color = "#757575"
            self.canvas.create_oval(cx + ox - 17, cy + oy - 17, cx + ox + 17, cy + oy + 17, fill=color, outline="white", width=2)
            self.canvas.create_text(cx + ox, cy + oy, text=ag, fill="white", font=("Arial", 12, "bold"))

    def cell_center(self, pos: Position) -> Tuple[float, float]:
        r, c = pos
        return c * self.cell + self.cell / 2, r * self.cell + self.cell / 2

    def update_action_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        state = self.current_display_state()
        legal = self.env.legal_actions(state)
        rows = []
        for idx in legal:
            next_state, reward, _done, info = self.env.transition(state, idx)
            q = self.agent.Q[state][idx]
            rows.append((idx, q, info["action_time"], info["new_makespan"], reward))
        rows.sort(key=lambda x: x[1], reverse=True)
        for rank, (idx, q, action_time, new_makespan, reward) in enumerate(rows, start=1):
            self.tree.insert("", tk.END, values=(rank, self.env.format_action(idx), f"{q:.3f}", f"{action_time}s", f"{new_makespan}s", f"{reward:.2f}"))

    def update_status(self, extra_msg: Optional[str] = None) -> None:
        state = self.current_display_state()
        positions, completed_mask, done_mask, times = state
        completed = self.env.completed_task_names(completed_mask)
        remaining = self.env.remaining_task_names(completed_mask)
        done_agents = [ag for ag in self.env.agents if self.env.agent_done(done_mask, ag)]
        makespan = self.env.makespan(times)
        legal = self.env.legal_actions(state)
        if legal:
            q_values = self.agent.Q[state]
            best_idx = max(legal, key=lambda i: q_values[i])
            best_action = self.env.format_action(best_idx)
            best_q = q_values[best_idx]
        else:
            best_action = "无"
            best_q = 0.0
        best_solution_text = "暂无" if self.best_solution is None else f"{self.best_solution['time']}s，第{self.best_solution['episode']}回合"
        text = (
            f"回合：{self.episode}，本回合动作数：{self.episode_steps}，总训练步数：{self.total_steps}\n"
            f"A位置={positions[0]}，B位置={positions[1]}，C位置={positions[2]}\n"
            f"A/B/C累计耗时：{times}，当前总时间 makespan={makespan}s\n"
            f"已完成任务：{completed if completed else '无'}\n"
            f"剩余任务：{remaining if remaining else '无'}\n"
            f"已到终点智能体：{done_agents if done_agents else '无'}\n"
            f"α={self.alpha.get():.3f}，γ={self.gamma.get():.3f}，ε={self.epsilon.get():.5f}，目标 ε<{self.target_epsilon.get():.3f}\n"
            f"当前最优动作：{best_action}，Q={best_q:.3f}\n"
            f"历史最好方案：{best_solution_text}"
        )
        if self.last_transition:
            t = self.last_transition
            info = t.info
            text += (
                f"\n\n上一步：{self.env.format_action(t.action_index)}\n"
                f"{info['reason']}\n"
                f"动作耗时={info['action_time']}s，makespan增加={info['delta_makespan']}s\n"
                f"Q旧值={t.old_q:.3f}，目标值={t.target:.3f}，TD误差={t.td_error:.3f}，Q新值={t.new_q:.3f}"
            )
        if extra_msg:
            text += f"\n\n{extra_msg}"
        self.status_label.config(text=text)

    def log_transition(self, t: StepRecord) -> None:
        info = t.info
        msg = (
            f"第 {self.total_steps} 步 | 回合 {self.episode}\n"
            f"状态 S：{self.env.format_state_short(t.state)}\n"
            f"动作 U：{self.env.format_action(t.action_index)}，{t.choose_reason}\n"
            f"环境反馈：{info['reason']}\n"
            f"动作耗时：移动 {info['move_time']}s + 任务 {info['task_time']}s = {info['action_time']}s\n"
            f"旧 makespan={info['old_makespan']}s，新 makespan={info['new_makespan']}s，增加 {info['delta_makespan']}s\n"
            f"奖励 r={t.reward:.2f}\n"
            f"新状态 S'：{self.env.format_state_short(t.next_state)}\n"
            f"更新公式：Q(S,U) ← Q(S,U) + α[r + γ maxQ(S',U') - Q(S,U)]\n"
            f"代入：{t.old_q:.3f} + {self.alpha.get():.3f} * "
            f"[{t.reward:.3f} + {self.gamma.get():.3f} * {t.next_max_q:.3f} - {t.old_q:.3f}]\n"
            f"目标值={t.target:.3f}，TD误差={t.td_error:.3f}，新Q值={t.new_q:.3f}\n"
            + "-" * 88 + "\n"
        )
        self.log(msg)

    def log(self, msg: str) -> None:
        self.log_box.insert(tk.END, msg)
        self.log_box.see(tk.END)


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
