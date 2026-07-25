#!/usr/bin/env python3
"""Framework-core smoke tests (no libtorch / no live RL client).

Mirrors GigaLearnCPP algorithms in pure Python so CI/smoke can validate edge cases:
  - GAE last-step OOB / force-truncate banking / all-trunc TD(0)
  - Advantage normalization modes
  - PPO ratio log clamp
  - RuntimeRewardRegistry NaN / Inf / extreme clamp

Run:
  python tools/run_framework_tests.py
  python tools/run_framework_tests.py -q
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TERM_NONE = 0
TERM_NORMAL = 1
TERM_TRUNCATED = 2


# ---------------------------------------------------------------------------
# GAE (mirrors GigaLearnCPP/src/private/GigaLearnCPP/PPO/GAE.cpp CPU path)
# ---------------------------------------------------------------------------

def gae_compute(
    rews: list[float],
    terminals: list[int],
    val_preds: list[float],
    trunc_val_preds: list[float] | None,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    return_std: float = 0.0,
    clip_range: float = 10.0,
) -> tuple[list[float], list[float], list[float], float]:
    n = len(rews)
    assert len(terminals) == n and len(val_preds) == n
    has_trunc = trunc_val_preds is not None
    num_truncs = len(trunc_val_preds) if has_trunc else 0

    # All-truncated TD(0) fast path
    all_trunc = (
        has_trunc
        and num_truncs == n
        and n > 0
        and all(t == TERM_TRUNCATED for t in terminals)
    )
    advantages = [0.0] * n
    returns = [0.0] * n
    total_rew = 0.0
    total_clipped = 0.0

    if all_trunc:
        assert trunc_val_preds is not None
        for step in range(n):
            if return_std != 0:
                cur = rews[step] / return_std
                total_rew += abs(cur)
                if clip_range > 0:
                    cur = max(-clip_range, min(clip_range, cur))
                total_clipped += abs(cur)
            else:
                cur = rews[step]
                total_rew += abs(cur)
            pred = cur + gamma * trunc_val_preds[step]
            advantages[step] = pred - val_preds[step]
            returns[step] = rews[step]
        targets = [val_preds[i] + advantages[i] for i in range(n)]
        clip_portion = (total_rew - total_clipped) / max(total_rew, 1e-7)
        return advantages, targets, returns, clip_portion

    prev_lambda = 0.0
    prev_ret = 0.0
    trunc_count = 0
    assert trunc_val_preds is not None or not has_trunc

    for step in range(n - 1, -1, -1):
        terminal = terminals[step]
        done = 1.0 if terminal == TERM_NORMAL else 0.0
        trunc = 1.0 if terminal == TERM_TRUNCATED else 0.0

        if return_std != 0:
            cur = rews[step] / return_std
            total_rew += abs(cur)
            if clip_range > 0:
                cur = max(-clip_range, min(clip_range, cur))
            total_clipped += abs(cur)
        else:
            cur = rews[step]
            total_rew += abs(cur)

        if terminal == TERM_TRUNCATED:
            assert has_trunc and trunc_val_preds is not None
            if trunc_count >= num_truncs:
                raise RuntimeError("too many truncations")
            # Consume from end (backward scan), matching GAE.cpp
            next_v = trunc_val_preds[num_truncs - 1 - trunc_count]
            trunc_count += 1
        elif step >= n - 1:
            # Force-truncate edge: last buffer step without terminal — V=0, never OOB
            next_v = 0.0
            trunc = 1.0
        else:
            next_v = val_preds[step + 1]

        pred = cur + gamma * next_v * (1.0 - done)
        delta = pred - val_preds[step]
        cur_ret = rews[step] + prev_ret * gamma * (1.0 - done) * (1.0 - trunc)
        returns[step] = cur_ret
        prev_lambda = delta + gamma * lam * (1.0 - done) * (1.0 - trunc) * prev_lambda
        advantages[step] = prev_lambda
        prev_ret = cur_ret

    if has_trunc and trunc_count != num_truncs:
        raise RuntimeError(f"trunc count mismatch {trunc_count}/{num_truncs}")

    targets = [val_preds[i] + advantages[i] for i in range(n)]
    clip_portion = (total_rew - total_clipped) / max(total_rew, 1e-7)
    return advantages, targets, returns, clip_portion


def test_gae_last_step_no_oob() -> None:
    # No terminal on last step — must bootstrap V=0, not read val_preds[n]
    rews = [1.0, 2.0, 3.0]
    terminals = [TERM_NONE, TERM_NONE, TERM_NONE]
    vals = [0.5, 0.5, 0.5]
    adv, tgt, rets, _ = gae_compute(rews, terminals, vals, None, gamma=0.9, lam=0.9)
    assert len(adv) == 3
    # Last step: δ = 3 + 0.9*0 - 0.5 = 2.5
    assert abs(adv[2] - 2.5) < 1e-5, adv[2]
    print("  gae last-step OOB guard ok")


def test_gae_all_trunc_td0() -> None:
    rews = [1.0, -0.5, 2.0]
    terminals = [TERM_TRUNCATED] * 3
    vals = [0.1, 0.2, 0.3]
    trunc_v = [0.4, 0.5, 0.6]
    adv, tgt, rets, _ = gae_compute(rews, terminals, vals, trunc_v, gamma=0.99, lam=0.95)
    for i in range(3):
        expect = rews[i] + 0.99 * trunc_v[i] - vals[i]
        assert abs(adv[i] - expect) < 1e-5, (i, adv[i], expect)
        assert rets[i] == rews[i]
    print("  gae all-trunc TD0 ok")


def test_gae_force_truncate_banking() -> None:
    # Mixed: mid episode then trunc at end (force-truncate bank style)
    rews = [0.1, 0.2, 1.0]
    terminals = [TERM_NONE, TERM_NONE, TERM_TRUNCATED]
    vals = [0.0, 0.0, 0.0]
    trunc_v = [0.5]  # bootstrap for the truncated step
    adv, _, rets, _ = gae_compute(rews, terminals, vals, trunc_v, gamma=1.0, lam=1.0)
    # trunc step: δ = 1 + 1*0.5 - 0 = 1.5
    assert abs(adv[2] - 1.5) < 1e-5, adv
    assert abs(rets[2] - 1.0) < 1e-5  # trunc stops return chain
    print("  gae force-truncate banking ok")


def test_gae_normal_terminal_zeros_bootstrap() -> None:
    rews = [1.0, 1.0]
    terminals = [TERM_NONE, TERM_NORMAL]
    vals = [0.0, 0.0]
    adv, _, rets, _ = gae_compute(rews, terminals, vals, None, gamma=0.99, lam=0.95)
    assert abs(adv[1] - 1.0) < 1e-5
    assert abs(rets[1] - 1.0) < 1e-5
    print("  gae normal terminal ok")


# ---------------------------------------------------------------------------
# Advantage norm (mirrors PPOLearner NormalizeAdvantages)
# ---------------------------------------------------------------------------

def normalize_advantages(adv: list[float], mode: int, eps: float = 1e-8) -> list[float]:
    if mode <= 0 or not adv:
        return list(adv)
    mean = sum(adv) / len(adv)
    if mode == 2:
        return [a - mean for a in adv]
    if mode == 3:
        scale = max(max(abs(a) for a in adv), eps)
        return [a / scale for a in adv]
    # mode 1: mean/std
    var = sum((a - mean) ** 2 for a in adv) / max(len(adv) - 1, 1)
    std = math.sqrt(var) + eps
    return [(a - mean) / std for a in adv]


def test_advantage_norm() -> None:
    adv = [1.0, 2.0, 3.0, 4.0]
    n0 = normalize_advantages(adv, 0)
    assert n0 == adv
    n1 = normalize_advantages(adv, 1)
    assert abs(sum(n1) / len(n1)) < 1e-6
    n2 = normalize_advantages(adv, 2)
    assert abs(sum(n2)) < 1e-6
    n3 = normalize_advantages(adv, 3)
    assert abs(max(abs(x) for x in n3) - 1.0) < 1e-6
    print("  advantage norm modes ok")


# ---------------------------------------------------------------------------
# Ratio log clamp
# ---------------------------------------------------------------------------

def ratio_from_logprobs(log_p: float, log_old: float, clamp: float = 20.0) -> float:
    log_ratio = max(-clamp, min(clamp, log_p - log_old))
    return math.exp(log_ratio)


def test_ratio_clamp() -> None:
    # Unclamped would explode
    r = ratio_from_logprobs(0.0, -100.0, clamp=20.0)
    assert abs(r - math.exp(20.0)) < 1e-3
    r2 = ratio_from_logprobs(0.0, 100.0, clamp=20.0)
    assert abs(r2 - math.exp(-20.0)) < 1e-9
    r3 = ratio_from_logprobs(-0.1, -0.2, clamp=20.0)
    assert abs(r3 - math.exp(0.1)) < 1e-9
    print("  ratio log clamp ok")


# ---------------------------------------------------------------------------
# RuntimeRewardRegistry clamp (mirrors RuntimeRewardRegistry.cpp)
# ---------------------------------------------------------------------------

def clamp_reward_mult(mult: float) -> float:
    if not math.isfinite(mult):
        return 1.0
    return max(-1e6, min(1e6, mult))


def test_reward_registry_clamp() -> None:
    assert clamp_reward_mult(float("nan")) == 1.0
    assert clamp_reward_mult(float("inf")) == 1.0
    assert clamp_reward_mult(float("-inf")) == 1.0
    assert clamp_reward_mult(2.5) == 2.5
    assert clamp_reward_mult(1e9) == 1e6
    assert clamp_reward_mult(-1e9) == -1e6
    print("  RuntimeRewardRegistry NaN clamp ok")


# ---------------------------------------------------------------------------
# InSimEval gate
# ---------------------------------------------------------------------------

def eval_gate_due(
    *,
    update_interval: int,
    iterations_since: int,
    ts_per_eval: int,
    last_eval_ts: int,
    total_ts: int,
) -> bool:
    by_iter = update_interval > 0 and iterations_since >= update_interval
    by_ts = ts_per_eval > 0 and (
        last_eval_ts == 0 or (total_ts - last_eval_ts) >= ts_per_eval
    )
    if last_eval_ts == 0 and iterations_since == 0:
        return by_iter or (ts_per_eval > 0 and total_ts >= ts_per_eval)
    return by_iter or by_ts


def test_eval_gate() -> None:
    assert not eval_gate_due(
        update_interval=16, iterations_since=0, ts_per_eval=5_000_000,
        last_eval_ts=0, total_ts=1000,
    )
    assert eval_gate_due(
        update_interval=16, iterations_since=0, ts_per_eval=5_000_000,
        last_eval_ts=0, total_ts=5_000_000,
    )
    assert eval_gate_due(
        update_interval=16, iterations_since=16, ts_per_eval=0,
        last_eval_ts=100, total_ts=200,
    )
    assert not eval_gate_due(
        update_interval=16, iterations_since=5, ts_per_eval=5_000_000,
        last_eval_ts=1_000_000, total_ts=2_000_000,
    )
    print("  in-sim eval gate ok")


# ---------------------------------------------------------------------------
# Obs-stat sample count (mirrors Learner.cpp — RS_MIN not RS_MAX)
# ---------------------------------------------------------------------------

def obs_stat_sample_count(num_players: int, max_obs_samples: int) -> int:
    return min(num_players, max_obs_samples)


def test_obs_stat_sample_cap() -> None:
    # Bug was RS_MAX → 16384 samples/step on 8192×2 arenas when standardizeObs on
    assert obs_stat_sample_count(16384, 100) == 100
    assert obs_stat_sample_count(2, 100) == 2
    assert obs_stat_sample_count(0, 100) == 0
    print("  obs-stat sample cap (MIN) ok")


def obs_stat_pool_indices(
    *,
    sparring: bool,
    num_players: int,
    new_player_indices: list[int],
) -> list[int]:
    """Indices eligible for obsStat IncrementRow (learning team only when sparring)."""
    if sparring:
        return list(new_player_indices)
    return list(range(num_players))


def test_obs_stat_sparring_pool() -> None:
    pool = obs_stat_pool_indices(
        sparring=True, num_players=4, new_player_indices=[0, 1]
    )
    assert pool == [0, 1]
    pool2 = obs_stat_pool_indices(
        sparring=False, num_players=4, new_player_indices=[0, 1]
    )
    assert pool2 == [0, 1, 2, 3]
    print("  obs-stat sparring pool ok")


# ---------------------------------------------------------------------------
# Mid-run subsystem enable (mirrors Learner::EnsureRuntimeSubsystems logic)
# ---------------------------------------------------------------------------

def want_version_mgr(
    *,
    save_policy_versions: bool,
    skill_tracker_enabled: bool,
    train_against_old: bool,
) -> bool:
    return save_policy_versions or skill_tracker_enabled or train_against_old


def want_opponent_pool(*, enabled: bool, pool_loaded: bool) -> bool:
    return enabled and not pool_loaded


def test_midrun_subsystem_gates() -> None:
    # From-scratch boot
    assert not want_version_mgr(
        save_policy_versions=False, skill_tracker_enabled=False, train_against_old=False
    )
    # AutoTrainer post-warmup skill-eval
    assert want_version_mgr(
        save_policy_versions=False, skill_tracker_enabled=True, train_against_old=False
    )
    # AutoTrainer league sparring
    assert want_version_mgr(
        save_policy_versions=True, skill_tracker_enabled=False, train_against_old=True
    )
    assert want_opponent_pool(enabled=True, pool_loaded=False)
    assert not want_opponent_pool(enabled=True, pool_loaded=True)
    assert not want_opponent_pool(enabled=False, pool_loaded=False)
    print("  mid-run subsystem gates ok")


# ---------------------------------------------------------------------------
# periodic_eval in-sim blend preference
# ---------------------------------------------------------------------------

def live_skill_blend(*, has_in_sim_elo: bool, override: float | None = None) -> float:
    if override is not None:
        return override
    return 0.70 if has_in_sim_elo else 0.40


def test_periodic_eval_insim_blend() -> None:
    assert live_skill_blend(has_in_sim_elo=True) == 0.70
    assert live_skill_blend(has_in_sim_elo=False) == 0.40
    assert live_skill_blend(has_in_sim_elo=True, override=0.5) == 0.5
    # Blend formula
    harness, live = 1000.0, 1400.0
    b = live_skill_blend(has_in_sim_elo=True)
    blended = (1.0 - b) * harness + b * live
    assert abs(blended - 1280.0) < 1e-6
    print("  periodic_eval in-sim blend ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    tests = [
        test_gae_last_step_no_oob,
        test_gae_all_trunc_td0,
        test_gae_force_truncate_banking,
        test_gae_normal_terminal_zeros_bootstrap,
        test_advantage_norm,
        test_ratio_clamp,
        test_reward_registry_clamp,
        test_eval_gate,
        test_obs_stat_sample_cap,
        test_obs_stat_sparring_pool,
        test_midrun_subsystem_gates,
        test_periodic_eval_insim_blend,
    ]

    failed = 0
    if not args.quiet:
        print("Framework core tests:")
    for t in tests:
        try:
            t()
        except Exception as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")

    if failed:
        print(f"FAILED {failed}/{len(tests)}")
        return 1
    if not args.quiet:
        print(f"OK {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
