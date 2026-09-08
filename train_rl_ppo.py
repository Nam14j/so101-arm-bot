#!/usr/bin/env python3
"""
train_rl_ppo.py — SAC for SO-101 Ball Pick (V11 staged reward)

Updated 2026-09-08: dropped HerReplayBuffer. V11 (see so101_env.py) is a
fixed-height staged reward, not a goal-distance reward, so HER's "relabel
a failed episode as a success toward wherever it actually ended up"
trick had nothing meaningful left to do — it was only useful for the old
goal-distance-based reward this project no longer uses. Now uses SAC's
plain default replay buffer instead. Also now starts every episode at
the FINAL curriculum level's position (level 3 — full workspace, home
pose) by default instead of the easier level-1 starting pose, and no
longer auto-advances through curriculum levels — pass --lock_level to
override.

  • Algorithm  : SAC (off-policy, works great with sparse/staged rewards)
  • Replay     : SAC's default ReplayBuffer (no goal relabeling)
  • Reward     : V11 staged reward — see so101_env.py
  • Observation: Dict {observation, achieved_goal, desired_goal} — kept for
                 interface stability, though achieved/desired_goal are now
                 unused by the reward itself

Run:
  python train_rl_ppo.py
  python train_rl_ppo.py --timesteps 2000000
  python train_rl_ppo.py --device cpu
"""

import os
import sys
import time
import argparse
import warnings
from collections import deque
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_env import SO101PickEnv

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "rl_models", "so101_sac_her")


def make_env():
    return SO101PickEnv(reward_type="v11")


# ── Curriculum Callback ────────────────────────────────────────────────────────
class CurriculumCallback(BaseCallback):
    """
    Auto-advances curriculum level, gated on the agent's OWN recent success
    rate rather than a fixed step count — a run that never solves Level 1
    stays on Level 1 instead of being pushed into a harder level it has no
    chance of solving either (that was the bug: 0% success the whole night,
    yet it still advanced to Level 3 at the 800k-step mark regardless).
      Level 1 : Arm starts at ball — just clamp & lift
      Level 2 : Arm 4cm above — drop, clamp & lift
      Level 3 : Full workspace from home pose
    """
    SUCCESS_THRESHOLD    = 0.30   # advance once recent success rate crosses this
    MIN_EPISODES         = 20     # need at least this many recent episodes to judge
    MIN_STEPS_PER_LEVEL  = 50_000 # don't even check before this many steps in a level
    CHECK_EVERY          = 5_000

    def __init__(self, eval_env, verbose=1):
        super().__init__(verbose)
        self.eval_env         = eval_env
        self.current_level    = 1
        self.level_start_step = 0
        self.last_check_step  = 0

    def _on_training_start(self):
        self._set_level(1)

    def _set_level(self, level):
        self.current_level    = level
        self.level_start_step = self.num_timesteps
        self.training_env.env_method("set_curriculum_level", level)
        self.eval_env.env_method("set_curriculum_level", level)
        labels = {
            1: "LEVEL 1 — The Lift (arm already at ball, just clamp & lift)",
            2: "LEVEL 2 — Drop & Clamp (arm 4 cm above ball)",
            3: "LEVEL 3 — Full Workspace (reach from home pose)",
        }
        print("\n" + "=" * 70)
        print(f"🎓 CURRICULUM: {labels[level]}")
        print("=" * 70 + "\n")

    def _recent_success_rate(self):
        buf = getattr(self.model, "ep_success_buffer", None)
        if not buf or len(buf) < self.MIN_EPISODES:
            return None
        return float(np.mean(list(buf)[-self.MIN_EPISODES:]))

    def _on_step(self) -> bool:
        if self.current_level >= 3:
            return True
        if self.num_timesteps - self.level_start_step < self.MIN_STEPS_PER_LEVEL:
            return True
        if self.num_timesteps - self.last_check_step < self.CHECK_EVERY:
            return True
        self.last_check_step = self.num_timesteps

        rate = self._recent_success_rate()
        if rate is None:
            return True
        if self.verbose:
            print(f"📈 Level {self.current_level} success rate (last {self.MIN_EPISODES} eps): {rate:.0%}")
        if rate >= self.SUCCESS_THRESHOLD:
            self._set_level(self.current_level + 1)
        return True


# ── Grasp Diagnostics Callback ──────────────────────────────────────────────────
class GraspDiagnosticsCallback(BaseCallback):
    """
    Added 2026-09-08 to answer "how close is it actually getting?" once
    success stayed at 0% well past 1M steps on curriculum Level 1.

    Doesn't change training at all — just watches info["is_touching"],
    info["dist"], info["grip_angle"], and info["is_held"] every step and
    prints a periodic summary, so we can tell "never even touches the ball"
    apart from "touches it but the grip/distance is a little off" apart from
    "genuinely satisfies the hold gate sometimes, just not long/high enough."
    """
    CHECK_EVERY = 5_000
    WINDOW      = 20  # episodes

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.last_check_step = 0
        self.touched_ever  = deque(maxlen=self.WINDOW)
        self.held_ever     = deque(maxlen=self.WINDOW)
        self.best_dist     = deque(maxlen=self.WINDOW)   # closest approach all episode
        self.best_grip     = deque(maxlen=self.WINDOW)   # tightest grip seen WHILE touching
        self._ep = None  # per-vec-env running trackers, set on training start

    def _on_training_start(self):
        n = self.training_env.num_envs
        self._ep = [self._fresh_ep() for _ in range(n)]

    @staticmethod
    def _fresh_ep():
        return {"touched": False, "held": False, "min_dist": float("inf"), "min_grip_touching": float("inf")}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            e = self._ep[i]
            d = info.get("dist")
            if d is not None:
                e["min_dist"] = min(e["min_dist"], float(d))
            if info.get("is_touching"):
                e["touched"] = True
                g = info.get("grip_angle")
                if g is not None:
                    e["min_grip_touching"] = min(e["min_grip_touching"], float(g))
            if info.get("is_held"):
                e["held"] = True
            if i < len(dones) and dones[i]:
                self.touched_ever.append(e["touched"])
                self.held_ever.append(e["held"])
                self.best_dist.append(e["min_dist"])
                self.best_grip.append(e["min_grip_touching"])
                self._ep[i] = self._fresh_ep()

        if (self.num_timesteps - self.last_check_step >= self.CHECK_EVERY
                and len(self.touched_ever) >= 5):
            self.last_check_step = self.num_timesteps
            n = len(self.touched_ever)
            touch_pct = 100.0 * sum(self.touched_ever) / n
            held_pct  = 100.0 * sum(self.held_ever) / n
            finite_dist = [x for x in self.best_dist if x != float("inf")]
            finite_grip = [x for x in self.best_grip if x != float("inf")]
            best_dist = min(finite_dist) if finite_dist else float("nan")
            best_grip = min(finite_grip) if finite_grip else float("nan")
            if self.verbose:
                print(
                    f"🔍 Grasp diagnostics (last {n} eps): "
                    f"touched ball {touch_pct:.0f}% of episodes | "
                    f"fully satisfied grip+dist gate {held_pct:.0f}% of episodes | "
                    f"closest approach ever={best_dist:.3f}m | "
                    f"tightest grip while touching={best_grip:.3f}rad (gate needs <0.28)"
                )
        return True


# ── Early-Stop Callback ──────────────────────────────────────────────────────────
class EarlyStopCallback(BaseCallback):
    """
    Added 2026-09-08. Stops training early once EITHER condition is hit:
      - recent success rate crosses `success_threshold`
      - wall-clock time since training started passes `time_limit_seconds`
    Whichever comes first. Either can be left as None to disable that check.
    A callback returning False just ends model.learn() normally (no
    exception), so the existing save-final-model code still runs.
    """
    MIN_EPISODES = 20   # need at least this many recent episodes to judge success rate
    CHECK_EVERY  = 2_000

    def __init__(self, success_threshold=None, time_limit_seconds=None, verbose=1):
        super().__init__(verbose)
        self.success_threshold  = success_threshold
        self.time_limit_seconds = time_limit_seconds
        self.start_time         = None
        self.last_check_step    = 0

    def _on_training_start(self):
        self.start_time = time.time()

    def _recent_success_rate(self):
        buf = getattr(self.model, "ep_success_buffer", None)
        if not buf or len(buf) < self.MIN_EPISODES:
            return None
        return float(np.mean(list(buf)[-self.MIN_EPISODES:]))

    def _on_step(self) -> bool:
        if self.time_limit_seconds is not None:
            elapsed = time.time() - self.start_time
            if elapsed >= self.time_limit_seconds:
                print(f"\n⏰ Time limit reached ({elapsed/60:.1f} min) — stopping training.")
                return False

        if self.success_threshold is not None:
            if self.num_timesteps - self.last_check_step >= self.CHECK_EVERY:
                self.last_check_step = self.num_timesteps
                rate = self._recent_success_rate()
                if rate is not None:
                    if self.verbose:
                        print(f"🎯 Recent success rate (last {self.MIN_EPISODES} eps): {rate:.0%} "
                              f"(stop target: {self.success_threshold:.0%})")
                    if rate >= self.success_threshold:
                        print(f"\n✅ Success rate {rate:.0%} crossed target {self.success_threshold:.0%} — stopping training.")
                        return False
        return True


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int,   default=2_000_000)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="auto")
    parser.add_argument("--lock_level",     type=int,   default=3,
                         help="Pin curriculum to this level for the whole run (1/2/3) "
                              "instead of auto-advancing. Defaults to 3 (final level: "
                              "full workspace, home-pose start) as of 2026-09-08 — "
                              "pass e.g. --lock_level 1 to go back to starting at the "
                              "ball, or --lock_level None-equivalent isn't supported; "
                              "use the CurriculumCallback path by editing this default "
                              "back to None if auto-advance is wanted again.")
    parser.add_argument("--success_stop",   type=float, default=None,
                         help="Stop early once recent success rate (last 20 eps) reaches this (e.g. 0.10 = 10%%).")
    parser.add_argument("--time_limit_min", type=float, default=None,
                         help="Stop early after this many minutes of wall-clock training time, regardless of success rate.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("🤖 SO-101 SAC + HER TRAINER  (OpenAI FetchPickAndPlace method)")
    print("=" * 70)
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"  Reward    : Sparse  (0 = success, -1 = fail)")
    print(f"  Replay    : SAC default ReplayBuffer (no HER)")
    print("-" * 70)

    env      = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    callbacks = []
    if args.lock_level is not None:
        env.env_method("set_curriculum_level", args.lock_level)
        eval_env.env_method("set_curriculum_level", args.lock_level)
        print(f"🔒 Curriculum locked to Level {args.lock_level} — will NOT auto-advance this run.")
    else:
        callbacks.append(CurriculumCallback(eval_env=eval_env))

    diagnostics_cb = GraspDiagnosticsCallback()
    callbacks.append(diagnostics_cb)

    if args.success_stop is not None or args.time_limit_min is not None:
        time_limit_seconds = args.time_limit_min * 60 if args.time_limit_min is not None else None
        callbacks.append(EarlyStopCallback(success_threshold=args.success_stop,
                                            time_limit_seconds=time_limit_seconds))
        if args.success_stop is not None:
            print(f"🎯 Will stop early once success rate reaches {args.success_stop:.0%}.")
        if args.time_limit_min is not None:
            print(f"⏰ Will stop early after {args.time_limit_min:.0f} minutes regardless.")

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=OUTPUT_DIR,
        log_path=OUTPUT_DIR,
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=OUTPUT_DIR,
        name_prefix="so101_sac_her",
    )

    # Plain SAC (no HER) — MultiInputPolicy still handles the Dict obs space fine
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=args.lr,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.98,
        tau=0.05,
        learning_starts=5000,  # ~14 full episodes of pure random data before the critic
                                # and entropy auto-tuner start updating (500 was <2 episodes —
                                # too little variety before exploration started collapsing)
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        device=args.device,
    )

    print("\n🚀 Starting SAC + HER training...\n")
    start = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks + [eval_cb, checkpoint_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted.")
    finally:
        final_path = os.path.join(OUTPUT_DIR, "final_model.zip")
        model.save(final_path)
        elapsed = time.time() - start
        print(f"\n💾 Saved → {final_path}")
        print(f"⏱️  Elapsed: {elapsed / 60:.1f} minutes")
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
