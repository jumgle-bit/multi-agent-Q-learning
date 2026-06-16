"""
Multi-Agent Q-learning Scheduler v2.0

v2.0 features:
- Multi-agent static + dynamic task scheduling.
- Three robots A/B/C start at (0, 0), final goal is (5, 5).
- Initial tasks a-e must be completed; a dynamic task f can appear during scheduling.
- Q-learning learns which robot should do which task; task ownership is not hard-coded.
- Event-triggered exploration: when a dynamic task appears, epsilon is boosted.
- Tkinter visualization for learning process, learned assignment, paths, and Q updates.

Run:
    python src/main.py
"""

from __future__ import annotations

import random
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass
from tkinter import scrolledtext, ttk
from typing import Dict, List, Optional, Sequence, Set, Tuple

Position = Tuple[int, int]
State = Tuple[Tuple[Position, ...], int, int, int, Tuple[int, ...]]
# State = (positions, active_mask, completed_mask, done_mask, times)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    pos: Position
    duration: int
    priority: int
    capable: Set[str]
    initial_active: bool = True
    appear_step: Optional[int] = None
    description: str = ""


class DynamicMultiAgentQLearningGUI:
    """
    v2.0: dynamic multi-robot task scheduling with continuous exploration.

    Difference from v1.0:
    - v1.0: all tasks are known at the beginning.
    - v2.0: task f can appear during an episode as an event-triggered dynamic task.
    - When the event occurs, epsilon is boosted to keep exploration active.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Multi-Agent Q-learning v2.0 - Dynamic Task Scheduling")
        self.root.geometry("1360x860")

        # =========================
        # 1. Environment definition
        # =========================
        self.grid_size = 6  # coordinates 0..5
        self.start_pos: Position = (0, 0)
        self.goal_pos: Position = (5, 5)
        self.move_time_per_grid = 2

        self.agents = ["A", "B", "C"]
        self.agent_index = {ag: idx for idx, ag in enumerate(self.agents)}

        # Initial tasks a-e + dynamic task f.
        # The RL agent does NOT know the final assignment; it only knows legal candidate actions.
        self.tasks: Dict[str, TaskSpec] = {
            "a": TaskSpec("a", (0, 2), duration=2, priority=2, capable={"A"}, initial_active=True,
                          description="initial task, only A can do it"),
            "b": TaskSpec("b", (2, 4), duration=3, priority=3, capable={"A", "B"}, initial_active=True,
                          description="initial task, A/B can do it"),
            "c": TaskSpec("c", (3, 4), duration=4, priority=4, capable={"A", "B", "C"}, initial_active=True,
                          description="initial task, A/B/C can do it"),
            "d": TaskSpec("d", (1, 3), duration=1, priority=2, capable={"B", "C"}, initial_active=True,
                          description="initial task, B/C can do it"),
            "e": TaskSpec("e", (5, 3), duration=5, priority=5, capable={"C"}, initial_active=True,
                          description="initial task, only C can do it"),
            "f": TaskSpec("f", (4, 1), duration=3, priority=6, capable={"A", "C"}, initial_active=False,
                          appear_step=3, description="dynamic task, appears after several decisions"),
        }
        self.task_names = list(self.tasks.keys())
        self.task_index = {name: idx for idx, name in enumerate(self.task_names)}

        self.initial_active_mask = 0
        for t in self.task_names:
            if self.tasks[t].initial_active:
                self.initial_active_mask |= self.bit_task(t)

        self.all_agents_done_mask = (1 << len(self.agents)) - 1

        # Action = choose one robot to do one task or go to GOAL.
        # This is only the candidate action space, not a fixed assignment.
        self.targets = self.task_names + ["GOAL"]
        self.actions: List[Tuple[str, str]] = []
        for ag in self.agents:
            for target in self.targets:
                self.actions.append((ag, target))
        self.action_count = len(self.actions)

        # =========================
        # 2. Q-learning parameters
        # =========================
        self.alpha = tk.DoubleVar(value=0.25)
        self.gamma = tk.DoubleVar(value=0.95)
        self.epsilon = tk.DoubleVar(value=1.00)
        self.target_epsilon = tk.DoubleVar(value=0.05)
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.995
        self.event_epsilon_boost = tk.DoubleVar(value=0.35)
        self.stagnation_boost_after = tk.IntVar(value=200)

        self.invalid_penalty = -120.0
        self.success_bonus = 220.0
        self.task_reward_scale = 9.0
        self.max_steps_per_episode = 12

        # Q table: sparse dictionary, state -> Q values of all macro-actions.
        self.Q = defaultdict(lambda: [0.0 for _ in range(self.action_count)])

        # =========================
        # 3. Runtime state
        # =========================
        self.start_state = self.make_start_state()
        self.state: State = self.start_state
        self.display_state: Optional[State] = None

        self.episode = 1
        self.total_steps = 0
        self.episode_steps = 0
        self.done = False
        self.current_episode_plan = []
        self.last_transition = None
        self.last_move_path: Optional[List[Position]] = None
        self.event_log: List[str] = []

        self.best_solution = None
        self.episodes_since_best = 0

        self.auto_running = False
        self.demo_running = False
        self.demo_plan = []
        self.demo_index = 0

        self.build_ui()
        self.draw_grid()
        self.update_action_table()
        self.update_status("v2.0 已启动：初始任务 a-e 激活，动态任务 f 会在本回合第 3 个宏动作后出现。")

    # ============================================================
    # Basic state helpers
    # ============================================================
    def bit_task(self, task_name: str) -> int:
        return 1 << self.task_index[task_name]

    def bit_agent(self, agent_name: str) -> int:
        return 1 << self.agent_index[agent_name]

    def make_start_state(self) -> State:
        positions = tuple([self.start_pos for _ in self.agents])
        active_mask = self.initial_active_mask
        completed_mask = 0
        done_mask = 0
        times = tuple([0 for _ in self.agents])
        return positions, active_mask, completed_mask, done_mask, times

    def current_display_state(self) -> State:
        return self.display_state if self.display_state is not None else self.state

    def manhattan(self, p1: Position, p2: Position) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def travel_time(self, p1: Position, p2: Position) -> int:
        return self.manhattan(p1, p2) * self.move_time_per_grid

    def makespan(self, times: Sequence[int]) -> int:
        return max(times)

    def task_active(self, active_mask: int, task_name: str) -> bool:
        return (active_mask & self.bit_task(task_name)) != 0

    def activate_task(self, active_mask: int, task_name: str) -> int:
        return active_mask | self.bit_task(task_name)

    def task_completed(self, completed_mask: int, task_name: str) -> bool:
        return (completed_mask & self.bit_task(task_name)) != 0

    def set_task_completed(self, completed_mask: int, task_name: str) -> int:
        return completed_mask | self.bit_task(task_name)

    def agent_done(self, done_mask: int, agent_name: str) -> bool:
        return (done_mask & self.bit_agent(agent_name)) != 0

    def set_agent_done(self, done_mask: int, agent_name: str) -> int:
        return done_mask | self.bit_agent(agent_name)

    def active_task_names(self, active_mask: int) -> List[str]:
        return [t for t in self.task_names if self.task_active(active_mask, t)]

    def inactive_task_names(self, active_mask: int) -> List[str]:
        return [t for t in self.task_names if not self.task_active(active_mask, t)]

    def completed_task_names(self, active_mask: int, completed_mask: int) -> List[str]:
        return [t for t in self.active_task_names(active_mask) if self.task_completed(completed_mask, t)]

    def remaining_task_names(self, active_mask: int, completed_mask: int) -> List[str]:
        return [t for t in self.active_task_names(active_mask) if not self.task_completed(completed_mask, t)]

    def is_terminal_success(self, state: State) -> bool:
        positions, active_mask, completed_mask, done_mask, times = state
        active_done = (completed_mask & active_mask) == active_mask
        return active_done and done_mask == self.all_agents_done_mask

    def format_action(self, action_index: int) -> str:
        ag, target = self.actions[action_index]
        if target == "GOAL":
            return f"{ag} → 终点"
        return f"{ag} → 任务{target}"

    def format_state_short(self, state: State) -> str:
        positions, active_mask, completed_mask, done_mask, times = state
        active = self.active_task_names(active_mask)
        completed = self.completed_task_names(active_mask, completed_mask)
        remaining = self.remaining_task_names(active_mask, completed_mask)
        done_agents = [ag for ag in self.agents if self.agent_done(done_mask, ag)]
        return f"pos={positions}, active={active}, doneTask={completed}, remaining={remaining}, goal={done_agents}, t={times}"

    # ============================================================
    # Dynamic event and continuous exploration
    # ============================================================
    def maybe_trigger_dynamic_event(self) -> None:
        """Automatically activate task f after a fixed number of decisions in each episode."""
        positions, active_mask, completed_mask, done_mask, times = self.state

        task_f = self.tasks["f"]
        if task_f.appear_step is None:
            return
        if self.task_active(active_mask, "f"):
            return
        if self.episode_steps < task_f.appear_step:
            return
        if self.done:
            return

        new_active_mask = self.activate_task(active_mask, "f")
        self.state = (positions, new_active_mask, completed_mask, done_mask, times)
        msg = f"事件触发：动态任务 f 在第 {self.episode} 回合第 {self.episode_steps} 个宏动作后出现。"
        self.event_log.append(msg)
        self.log("\n" + msg + "\n")
        self.boost_exploration("动态任务出现")

    def manual_trigger_dynamic_task(self) -> None:
        positions, active_mask, completed_mask, done_mask, times = self.state
        if self.task_active(active_mask, "f"):
            self.log("\n动态任务 f 已经处于激活状态，无需重复触发。\n")
            return
        self.state = (positions, self.activate_task(active_mask, "f"), completed_mask, done_mask, times)
        msg = f"手动事件：动态任务 f 被激活。"
        self.event_log.append(msg)
        self.log("\n" + msg + "\n")
        self.boost_exploration("手动动态任务")
        self.draw_grid()
        self.update_action_table()
        self.update_status("动态任务 f 已激活，epsilon 已提升以继续探索。")

    def boost_exploration(self, reason: str) -> None:
        old_eps = self.epsilon.get()
        boost = self.event_epsilon_boost.get()
        new_eps = max(old_eps, boost)
        self.epsilon.set(round(new_eps, 5))
        self.log(f"持续探索机制：由于 {reason}，ε 从 {old_eps:.5f} 提升/保持到 {new_eps:.5f}\n")

    def maybe_stagnation_boost(self) -> None:
        threshold = self.stagnation_boost_after.get()
        if threshold <= 0:
            return
        if self.episodes_since_best > 0 and self.episodes_since_best % threshold == 0:
            self.boost_exploration(f"连续 {threshold} 回合没有更优方案")

    # ============================================================
    # Legal actions and transition
    # ============================================================
    def legal_actions(self, state: State) -> List[int]:
        positions, active_mask, completed_mask, done_mask, times = state
        if self.is_terminal_success(state):
            return []

        remaining = self.remaining_task_names(active_mask, completed_mask)
        legal = []
        for idx, (ag, target) in enumerate(self.actions):
            if self.agent_done(done_mask, ag):
                continue

            if target == "GOAL":
                # v2.0 baseline: robots go to the final goal only after all active tasks have been completed.
                if not remaining:
                    legal.append(idx)
                continue

            # Task target: must be active, unfinished, and the robot must be capable.
            if not self.task_active(active_mask, target):
                continue
            if self.task_completed(completed_mask, target):
                continue
            if ag not in self.tasks[target].capable:
                continue
            legal.append(idx)

        return legal

    def transition(self, state: State, action_index: int):
        positions, active_mask, completed_mask, done_mask, times = state
        old_makespan = self.makespan(times)

        if action_index not in self.legal_actions(state):
            info = {
                "valid": False,
                "reason": "非法动作：任务未激活/已完成/能力不匹配，或尚有任务未完成却去终点",
                "move_time": 0,
                "task_time": 0,
                "action_time": 0,
                "old_makespan": old_makespan,
                "new_makespan": old_makespan,
                "delta_makespan": 0,
                "path": [],
                "completed_task": None,
            }
            return state, self.invalid_penalty, False, info

        ag, target = self.actions[action_index]
        ag_i = self.agent_index[ag]
        old_pos = positions[ag_i]

        new_positions = list(positions)
        new_times = list(times)
        new_completed_mask = completed_mask
        new_done_mask = done_mask
        completed_task = None

        if target == "GOAL":
            target_pos = self.goal_pos
            move_t = self.travel_time(old_pos, target_pos)
            task_t = 0
            action_t = move_t
            new_positions[ag_i] = target_pos
            new_times[ag_i] += action_t
            new_done_mask = self.set_agent_done(done_mask, ag)
            reason = f"{ag} 前往终点：{old_pos} → {target_pos}，移动耗时 {move_t}s"
            base_reward = -float(max(new_times) - old_makespan)
        else:
            task = self.tasks[target]
            target_pos = task.pos
            move_t = self.travel_time(old_pos, target_pos)
            task_t = task.duration
            action_t = move_t + task_t
            new_positions[ag_i] = target_pos
            new_times[ag_i] += action_t
            new_completed_mask = self.set_task_completed(completed_mask, target)
            completed_task = target
            reason = (
                f"{ag} 自主选择任务{target}：{old_pos} → {target_pos}，"
                f"移动 {move_t}s，执行 {task_t}s，优先级 {task.priority}"
            )
            # Reward balances task priority and makespan increase.
            base_reward = task.priority * self.task_reward_scale - float(max(new_times) - old_makespan)

        new_state: State = (tuple(new_positions), active_mask, new_completed_mask, new_done_mask, tuple(new_times))
        new_makespan = self.makespan(tuple(new_times))
        delta_makespan = new_makespan - old_makespan

        reward = base_reward
        done = False
        if self.is_terminal_success(new_state):
            done = True
            reward += self.success_bonus
            reason += "；所有已激活任务完成，A/B/C 均到达终点"

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
            "completed_task": completed_task,
        }
        return new_state, reward, done, info

    def grid_path(self, start: Position, end: Position) -> List[Position]:
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

    # ============================================================
    # Q-learning
    # ============================================================
    def choose_action(self, state: State):
        legal = self.legal_actions(state)
        if not legal:
            return None, "无合法动作"

        eps = self.epsilon.get()
        if random.random() < eps:
            return random.choice(legal), "探索：随机选择一个合法宏动作"

        q_values = self.Q[state]
        best_q = max(q_values[i] for i in legal)
        best_actions = [i for i in legal if abs(q_values[i] - best_q) < 1e-12]
        chosen = random.choice(best_actions)
        if len(best_actions) > 1:
            return chosen, "利用：在并列最大 Q 值动作中随机选择"
        return chosen, "利用：选择当前最大 Q 值动作"

    def train_one_step(self):
        if self.demo_running:
            return

        self.display_state = None
        if self.done or self.episode_steps >= self.max_steps_per_episode:
            self.new_episode(clear_log=False)

        # Trigger event before action selection if the condition is already met.
        self.maybe_trigger_dynamic_event()

        state = self.state
        action_index, choose_reason = self.choose_action(state)
        if action_index is None:
            self.finish_episode("失败：无合法动作")
            return

        next_state, reward, done, info = self.transition(state, action_index)

        old_q = self.Q[state][action_index]
        next_legal = self.legal_actions(next_state)
        if done or not next_legal:
            next_max_q = 0.0
        else:
            next_max_q = max(self.Q[next_state][i] for i in next_legal)

        alpha = self.alpha.get()
        gamma = self.gamma.get()
        target = reward + gamma * next_max_q
        td_error = target - old_q
        new_q = old_q + alpha * td_error
        self.Q[state][action_index] = new_q

        self.state = next_state
        self.done = done
        self.total_steps += 1
        self.episode_steps += 1
        self.last_move_path = info["path"]

        self.last_transition = {
            "state": state,
            "action_index": action_index,
            "next_state": next_state,
            "reward": reward,
            "done": done,
            "info": info,
            "old_q": old_q,
            "next_max_q": next_max_q,
            "target": target,
            "td_error": td_error,
            "new_q": new_q,
            "choose_reason": choose_reason,
        }
        self.current_episode_plan.append(self.last_transition.copy())
        self.log_transition()

        # Trigger event immediately after the action if the appearance step has just been reached.
        if not done:
            self.maybe_trigger_dynamic_event()

        if done:
            final_time = self.makespan(next_state[4])
            self.record_success(final_time)
            self.finish_episode(f"成功：全部激活任务完成，总时间 {final_time}s")
        elif self.episode_steps >= self.max_steps_per_episode:
            self.finish_episode("达到最大宏动作数，本回合结束")

        self.draw_grid()
        self.update_action_table()
        self.update_status()

    def record_success(self, final_time: int) -> None:
        plan = [item.copy() for item in self.current_episode_plan]
        if self.best_solution is None or final_time < self.best_solution["time"]:
            self.best_solution = {
                "time": final_time,
                "episode": self.episode,
                "plan": plan,
            }
            self.episodes_since_best = 0
            self.log(f"\n发现历史最好方案：总时间 {final_time}s，来自第 {self.episode} 回合。\n")
            self.log(self.solution_summary_text(plan, final_time) + "\n")
        else:
            self.episodes_since_best += 1

    def finish_episode(self, reason: str) -> None:
        old_eps = self.epsilon.get()
        new_eps = max(self.epsilon_min, old_eps * self.epsilon_decay)
        self.epsilon.set(round(new_eps, 5))
        self.done = True
        self.log(f"\n第 {self.episode} 回合结束：{reason}。ε 从 {old_eps:.5f} 衰减到 {new_eps:.5f}\n")
        self.maybe_stagnation_boost()

    def new_episode(self, clear_log: bool = True) -> None:
        self.state = self.start_state
        self.display_state = None
        self.done = False
        self.episode += 1
        self.episode_steps = 0
        self.last_transition = None
        self.last_move_path = None
        self.current_episode_plan = []
        self.event_log = []
        if clear_log:
            self.log(f"\n新开第 {self.episode} 回合：机器人回到 (0,0)，初始任务 a-e 激活，动态任务 f 等待触发。\n")
        self.draw_grid()
        self.update_action_table()
        self.update_status()

    def reset_all(self) -> None:
        self.stop_auto()
        self.Q = defaultdict(lambda: [0.0 for _ in range(self.action_count)])
        self.state = self.start_state
        self.display_state = None
        self.episode = 1
        self.total_steps = 0
        self.episode_steps = 0
        self.done = False
        self.current_episode_plan = []
        self.last_transition = None
        self.last_move_path = None
        self.event_log = []
        self.best_solution = None
        self.episodes_since_best = 0
        self.epsilon.set(1.00)
        self.target_epsilon.set(0.05)
        self.log_box.delete("1.0", tk.END)
        self.log("已重置：Q 表清空，v2.0 动态任务环境重新开始。\n")
        self.draw_grid()
        self.update_action_table()
        self.update_status()

    # ============================================================
    # Auto training
    # ============================================================
    def auto_train_episodes(self, n: int) -> None:
        if self.demo_running:
            return
        self.auto_running = True
        self.auto_episodes_left = n
        if self.done:
            self.new_episode(clear_log=False)
        self.auto_episode_batch_loop()

    def auto_episode_batch_loop(self) -> None:
        if not self.auto_running:
            return
        if self.auto_episodes_left <= 0:
            self.auto_running = False
            self.update_status("自动训练完成。")
            return

        before_episode = self.episode
        self.train_one_step()
        if self.done and self.episode == before_episode:
            self.auto_episodes_left -= 1
        if self.done:
            self.new_episode(clear_log=False)
        self.root.after(2, self.auto_episode_batch_loop)

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
        self.root.after(1, self.auto_until_epsilon_loop)

    def stop_auto(self) -> None:
        self.auto_running = False
        self.demo_running = False

    # ============================================================
    # Demo and solution summary
    # ============================================================
    def greedy_plan_from_q(self, max_steps: int = 24):
        state = self.start_state
        plan = []
        visited = set()
        simulated_episode_steps = 0

        for _ in range(max_steps):
            # Apply the same dynamic event rule in greedy simulation.
            positions, active_mask, completed_mask, done_mask, times = state
            if simulated_episode_steps >= (self.tasks["f"].appear_step or 999999):
                if not self.task_active(active_mask, "f"):
                    active_mask = self.activate_task(active_mask, "f")
                    state = (positions, active_mask, completed_mask, done_mask, times)

            if self.is_terminal_success(state):
                break
            legal = self.legal_actions(state)
            if not legal:
                break
            q_values = self.Q[state]
            best_q = max(q_values[i] for i in legal)
            best_actions = [i for i in legal if abs(q_values[i] - best_q) < 1e-12]
            action_index = random.choice(best_actions)
            key = (state, action_index)
            if key in visited:
                break
            visited.add(key)
            next_state, reward, done, info = self.transition(state, action_index)
            item = {
                "state": state,
                "action_index": action_index,
                "next_state": next_state,
                "reward": reward,
                "done": done,
                "info": info,
                "old_q": self.Q[state][action_index],
                "next_max_q": 0.0,
                "target": 0.0,
                "td_error": 0.0,
                "new_q": self.Q[state][action_index],
                "choose_reason": "演示：按当前 Q 表贪心选择",
            }
            plan.append(item)
            state = next_state
            simulated_episode_steps += 1
            if done:
                break
        return plan

    def demo_current_policy(self) -> None:
        self.stop_auto()
        self.demo_plan = self.greedy_plan_from_q(max_steps=24)
        if not self.demo_plan:
            self.log("\n当前策略还没有形成有效方案。可以先训练几百或几千回合。\n")
            return
        final_state = self.demo_plan[-1]["next_state"]
        final_time = self.makespan(final_state[4])
        self.log("\n演示当前 Q 表学到的策略：\n")
        self.log(self.solution_summary_text(self.demo_plan, final_time) + "\n")
        self.start_demo(self.demo_plan)

    def demo_best_solution(self) -> None:
        self.stop_auto()
        if self.best_solution is None:
            self.log("\n目前还没有成功方案。建议先点【训练1000回合】或【训练到ε阈值】。\n")
            return
        self.log(f"\n演示历史最好方案：总时间 {self.best_solution['time']}s，来自第 {self.best_solution['episode']} 回合。\n")
        self.log(self.solution_summary_text(self.best_solution["plan"], self.best_solution["time"]) + "\n")
        self.start_demo(self.best_solution["plan"])

    def start_demo(self, plan) -> None:
        self.demo_plan = plan
        self.demo_index = 0
        self.demo_running = True
        self.display_state = self.start_state
        self.last_move_path = None
        self.demo_loop()

    def demo_loop(self) -> None:
        if not self.demo_running:
            return
        if self.demo_index >= len(self.demo_plan):
            self.demo_running = False
            self.update_status("方案演示结束。")
            return
        item = self.demo_plan[self.demo_index]
        self.display_state = item["next_state"]
        self.last_move_path = item["info"].get("path", [])
        self.last_transition = item
        self.demo_index += 1
        self.draw_grid()
        self.update_action_table()
        self.update_status("正在演示方案，不更新 Q 表。")
        self.root.after(850, self.demo_loop)

    def solution_summary_text(self, plan, final_time: int) -> str:
        owner = {}
        lines_by_agent = {ag: [] for ag in self.agents}
        for item in plan:
            ag, target = self.actions[item["action_index"]]
            if target != "GOAL":
                owner[target] = ag
                lines_by_agent[ag].append(f"任务{target}")
            else:
                lines_by_agent[ag].append("终点")
        lines = [f"学习得到的任务归属：{owner}"]
        for ag in self.agents:
            seq = " → ".join(lines_by_agent[ag]) if lines_by_agent[ag] else "无动作"
            lines.append(f"{ag}: {seq}")
        lines.append(f"总完成时间 makespan = {final_time}s")
        return "\n".join(lines)

    # ============================================================
    # UI construction
    # ============================================================
    def build_ui(self) -> None:
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        right = tk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0))

        self.canvas_size = 600
        self.cell = self.canvas_size // self.grid_size
        self.canvas = tk.Canvas(left, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        legend = (
            "v2.0：动态任务 + 持续探索。\n"
            "A/B/C 从 (0,0) 出发，最终都要到 (5,5)。\n"
            "初始任务 a-e 激活；动态任务 f 在第 3 个宏动作后出现，也可手动触发。\n"
            "圆圈内显示：任务名 / 执行时间 / 优先级 / 可执行机器人。灰色表示已完成。"
        )
        tk.Label(left, text=legend, justify=tk.LEFT, wraplength=600).pack(pady=8)

        param_frame = ttk.LabelFrame(right, text="参数设置")
        param_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(param_frame, text="学习率 α").grid(row=0, column=0, padx=4, pady=5, sticky="w")
        ttk.Entry(param_frame, textvariable=self.alpha, width=7).grid(row=0, column=1, padx=4, pady=5)

        ttk.Label(param_frame, text="折扣因子 γ").grid(row=0, column=2, padx=4, pady=5, sticky="w")
        ttk.Entry(param_frame, textvariable=self.gamma, width=7).grid(row=0, column=3, padx=4, pady=5)

        ttk.Label(param_frame, text="探索率 ε").grid(row=0, column=4, padx=4, pady=5, sticky="w")
        ttk.Entry(param_frame, textvariable=self.epsilon, width=8).grid(row=0, column=5, padx=4, pady=5)

        ttk.Label(param_frame, text="训练到 ε <").grid(row=0, column=6, padx=4, pady=5, sticky="w")
        ttk.Entry(param_frame, textvariable=self.target_epsilon, width=8).grid(row=0, column=7, padx=4, pady=5)

        ttk.Label(param_frame, text="事件ε提升至").grid(row=1, column=0, padx=4, pady=5, sticky="w")
        ttk.Entry(param_frame, textvariable=self.event_epsilon_boost, width=7).grid(row=1, column=1, padx=4, pady=5)

        ttk.Label(param_frame, text="无改进提升间隔").grid(row=1, column=2, padx=4, pady=5, sticky="w")
        ttk.Entry(param_frame, textvariable=self.stagnation_boost_after, width=7).grid(row=1, column=3, padx=4, pady=5)

        button_frame = ttk.LabelFrame(right, text="训练控制")
        button_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(button_frame, text="单步训练", command=self.train_one_step).grid(row=0, column=0, padx=4, pady=5)
        ttk.Button(button_frame, text="训练100回合", command=lambda: self.auto_train_episodes(100)).grid(row=0, column=1, padx=4, pady=5)
        ttk.Button(button_frame, text="训练1000回合", command=lambda: self.auto_train_episodes(1000)).grid(row=0, column=2, padx=4, pady=5)
        ttk.Button(button_frame, text="训练到ε阈值", command=self.auto_train_until_epsilon).grid(row=0, column=3, padx=4, pady=5)
        ttk.Button(button_frame, text="手动触发任务f", command=self.manual_trigger_dynamic_task).grid(row=0, column=4, padx=4, pady=5)

        ttk.Button(button_frame, text="演示当前策略", command=self.demo_current_policy).grid(row=1, column=0, padx=4, pady=5)
        ttk.Button(button_frame, text="演示历史最好", command=self.demo_best_solution).grid(row=1, column=1, padx=4, pady=5)
        ttk.Button(button_frame, text="停止", command=self.stop_auto).grid(row=1, column=2, padx=4, pady=5)
        ttk.Button(button_frame, text="重置", command=self.reset_all).grid(row=1, column=3, padx=4, pady=5)

        status_frame = ttk.LabelFrame(right, text="当前信息")
        status_frame.pack(fill=tk.X, pady=(0, 8))
        self.status_label = tk.Label(status_frame, text="", justify=tk.LEFT, anchor="w")
        self.status_label.pack(fill=tk.X, padx=8, pady=6)

        table_frame = ttk.LabelFrame(right, text="当前状态下可选择的合法动作 Q 值")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        columns = ("rank", "action", "q", "cost", "new_time", "reward")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=9)
        self.tree.heading("rank", text="序号")
        self.tree.heading("action", text="动作")
        self.tree.heading("q", text="Q值")
        self.tree.heading("cost", text="动作耗时")
        self.tree.heading("new_time", text="动作后总时间")
        self.tree.heading("reward", text="奖励")
        self.tree.column("rank", width=50, anchor="center")
        self.tree.column("action", width=130, anchor="center")
        self.tree.column("q", width=90, anchor="center")
        self.tree.column("cost", width=90, anchor="center")
        self.tree.column("new_time", width=110, anchor="center")
        self.tree.column("reward", width=90, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(right, text="训练日志与公式代入")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_box = scrolledtext.ScrolledText(log_frame, height=13, wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    # ============================================================
    # Drawing and display
    # ============================================================
    def draw_grid(self) -> None:
        self.canvas.delete("all")
        state = self.current_display_state()
        positions, active_mask, completed_mask, done_mask, times = state

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x1, y1 = c * self.cell, r * self.cell
                x2, y2 = x1 + self.cell, y1 + self.cell
                pos = (r, c)
                fill = "#ffffff"
                if pos == self.start_pos:
                    fill = "#e8f3ff"
                if pos == self.goal_pos:
                    fill = "#d9ffd9"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="#999999")
                self.canvas.create_text(x1 + 18, y1 + 14, text=f"{r},{c}", font=("Arial", 8), fill="#555555")

        # Goal
        gx, gy = self.cell_center(self.goal_pos)
        self.canvas.create_rectangle(gx - 30, gy - 20, gx + 30, gy + 20, fill="#6ccf6c", outline="")
        self.canvas.create_text(gx, gy, text="终点", font=("Arial", 12, "bold"), fill="white")

        # Tasks
        for task_name in self.task_names:
            task = self.tasks[task_name]
            cx, cy = self.cell_center(task.pos)
            active = self.task_active(active_mask, task_name)
            completed = active and self.task_completed(completed_mask, task_name)
            if not active:
                color = "#f0f0f0"
                outline = "#b0b0b0"
                text_color = "#888888"
            elif completed:
                color = "#bdbdbd"
                outline = "#777777"
                text_color = "#ffffff"
            else:
                color = "#ffb74d" if task_name != "f" else "#ff7043"
                outline = "#7a4a00"
                text_color = "#000000"
            cap = "".join(sorted(task.capable))
            title = f"{task_name}" + ("✓" if completed else "")
            if not active:
                title = f"{task_name}?"
            label = f"{title}\n{task.duration}s P{task.priority}\n{cap}"
            self.canvas.create_oval(cx - 27, cy - 27, cx + 27, cy + 27, fill=color, outline=outline, width=2)
            self.canvas.create_text(cx, cy, text=label, font=("Arial", 8, "bold"), fill=text_color)

        # Last path
        if self.last_move_path and len(self.last_move_path) >= 2:
            pts = []
            for p in self.last_move_path:
                pts.extend(self.cell_center(p))
            self.canvas.create_line(*pts, fill="#ff5252", width=4, arrow=tk.LAST)

        # Agents, with offsets for same cell
        offsets = {"A": (-18, -18), "B": (18, -18), "C": (0, 18)}
        colors = {"A": "#e53935", "B": "#1e88e5", "C": "#43a047"}
        for ag in self.agents:
            i = self.agent_index[ag]
            pos = positions[i]
            cx, cy = self.cell_center(pos)
            ox, oy = offsets[ag]
            color = "#757575" if self.agent_done(done_mask, ag) else colors[ag]
            self.canvas.create_oval(cx + ox - 17, cy + oy - 17, cx + ox + 17, cy + oy + 17,
                                    fill=color, outline="white", width=2)
            self.canvas.create_text(cx + ox, cy + oy, text=ag, fill="white", font=("Arial", 12, "bold"))

    def cell_center(self, pos: Position) -> Tuple[float, float]:
        r, c = pos
        return c * self.cell + self.cell / 2, r * self.cell + self.cell / 2

    def update_action_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        state = self.current_display_state()
        legal = self.legal_actions(state)
        rows = []
        for idx in legal:
            next_state, reward, done, info = self.transition(state, idx)
            q = self.Q[state][idx]
            rows.append((idx, q, info["action_time"], info["new_makespan"], reward))
        rows.sort(key=lambda x: x[1], reverse=True)
        for rank, (idx, q, action_time, new_makespan, reward) in enumerate(rows, start=1):
            values = (rank, self.format_action(idx), f"{q:.3f}", f"{action_time}s", f"{new_makespan}s", f"{reward:.2f}")
            self.tree.insert("", tk.END, values=values)

    def update_status(self, extra_msg: Optional[str] = None) -> None:
        state = self.current_display_state()
        positions, active_mask, completed_mask, done_mask, times = state
        active = self.active_task_names(active_mask)
        inactive = self.inactive_task_names(active_mask)
        completed = self.completed_task_names(active_mask, completed_mask)
        remaining = self.remaining_task_names(active_mask, completed_mask)
        done_agents = [ag for ag in self.agents if self.agent_done(done_mask, ag)]
        makespan = self.makespan(times)
        legal = self.legal_actions(state)
        if legal:
            q_values = self.Q[state]
            best_idx = max(legal, key=lambda i: q_values[i])
            best_action = self.format_action(best_idx)
            best_q = q_values[best_idx]
        else:
            best_action = "无"
            best_q = 0.0
        best_solution_text = "暂无" if self.best_solution is None else f"{self.best_solution['time']}s，第{self.best_solution['episode']}回合"
        text = (
            f"回合：{self.episode}，本回合动作数：{self.episode_steps}，总训练步数：{self.total_steps}\n"
            f"A位置={positions[0]}，B位置={positions[1]}，C位置={positions[2]}\n"
            f"A/B/C累计耗时：{times}，当前总时间 makespan={makespan}s\n"
            f"激活任务：{active if active else '无'}；未激活任务：{inactive if inactive else '无'}\n"
            f"已完成任务：{completed if completed else '无'}；剩余任务：{remaining if remaining else '无'}\n"
            f"已到终点智能体：{done_agents if done_agents else '无'}\n"
            f"α={self.alpha.get():.3f}，γ={self.gamma.get():.3f}，ε={self.epsilon.get():.5f}，目标 ε<{self.target_epsilon.get():.3f}\n"
            f"事件探索提升阈值={self.event_epsilon_boost.get():.3f}，当前最优动作：{best_action}，Q={best_q:.3f}\n"
            f"历史最好方案：{best_solution_text}"
        )
        if self.last_transition:
            t = self.last_transition
            info = t["info"]
            text += (
                f"\n\n上一步：{self.format_action(t['action_index'])}\n"
                f"{info['reason']}\n"
                f"动作耗时={info['action_time']}s，makespan增加={info['delta_makespan']}s\n"
                f"Q旧值={t['old_q']:.3f}，目标值={t['target']:.3f}，TD误差={t['td_error']:.3f}，Q新值={t['new_q']:.3f}"
            )
        if extra_msg:
            text += f"\n\n{extra_msg}"
        self.status_label.config(text=text)

    def log_transition(self) -> None:
        t = self.last_transition
        info = t["info"]
        msg = (
            f"第 {self.total_steps} 步 | 回合 {self.episode}\n"
            f"状态 S：{self.format_state_short(t['state'])}\n"
            f"动作 U：{self.format_action(t['action_index'])}，{t['choose_reason']}\n"
            f"环境反馈：{info['reason']}\n"
            f"动作耗时：移动 {info['move_time']}s + 任务 {info['task_time']}s = {info['action_time']}s\n"
            f"旧 makespan={info['old_makespan']}s，新 makespan={info['new_makespan']}s，增加 {info['delta_makespan']}s\n"
            f"奖励 r={t['reward']:.2f}\n"
            f"新状态 S'：{self.format_state_short(t['next_state'])}\n"
            f"更新公式：Q(S,U) ← Q(S,U) + α[r + γ maxQ(S',U') - Q(S,U)]\n"
            f"代入：{t['old_q']:.3f} + {self.alpha.get():.3f} * "
            f"[{t['reward']:.3f} + {self.gamma.get():.3f} * {t['next_max_q']:.3f} - {t['old_q']:.3f}]\n"
            f"目标值={t['target']:.3f}，TD误差={t['td_error']:.3f}，新Q值={t['new_q']:.3f}\n"
            + "-" * 92 + "\n"
        )
        self.log(msg)

    def log(self, msg: str) -> None:
        self.log_box.insert(tk.END, msg)
        self.log_box.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = DynamicMultiAgentQLearningGUI(root)
    root.mainloop()
