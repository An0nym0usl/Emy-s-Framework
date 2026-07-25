import json
import os
import time
import traceback
import socket
from collections import deque
from pathlib import Path

# Send to RocketSimVis
UDP_IP = "127.0.0.1"
UDP_PORT = 9273

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

# Throttled sidecar for companion HUD / AutoTrainer dashboard (works without AT).
_HUD_INTERVAL_S = float(os.environ.get("GIGA_RENDER_HUD_INTERVAL", "0.2"))
_HISTORY_INTERVAL_S = float(os.environ.get("GIGA_VIS_HISTORY_INTERVAL", "2.0"))
_HISTORY_CAP = int(os.environ.get("GIGA_VIS_HISTORY_CAP", "2000"))
_ROLL_WINDOW_S = float(os.environ.get("GIGA_VIS_ROLL_WINDOW_S", "30.0"))
_NO_TOUCH_TIMEOUT_S = float(os.environ.get("GIGA_VIS_NO_TOUCH_S", "8.0"))
_hud_next_t = 0.0
_hist_next_t = 0.0
_hud_roll = None
_hud_history = []  # recent step totals for sparkline
_HUD_HISTORY_CAP = 120
_session_steps = 0
_episode_steps = 0
_episodes = 0
_last_episode_steps = None
_prev_ball_xy = None
_last_touch_unix = None
_goal_cooldown_until = 0.0
_prev_ball_y = None

# Rolling sample buffer: (unix_t, reward, touch, air, boost_use, speed, head_dict)
_roll_samples = deque(maxlen=4000)
# Per-episode accumulators (reset on kickoff)
_ep_touch = 0
_ep_air = 0
_ep_boost = 0
_ep_speed_sum = 0.0
_ep_reward_sum = 0.0
_ep_samples = 0
_last_episode_summary = None
_goals_for = 0
_goals_against = 0
_events_recent = deque(maxlen=24)  # {t, kind, detail}


def _action_to_controls(action):
    """Map GigaLearn action list (8 floats) to RocketSimVis controls dict."""
    if not isinstance(action, (list, tuple)) or len(action) < 8:
        return None
    return {
        "throttle": float(action[0]),
        "steer": float(action[1]),
        "pitch": float(action[2]),
        "yaw": float(action[3]),
        "roll": float(action[4]),
        "jump": bool(float(action[5]) > 0.0),
        "boost": bool(float(action[6]) > 0.0),
        "handbrake": bool(float(action[7]) > 0.0),
    }


def _watch_dirs():
    """Candidate autotrainer dirs (cwd is usually build/Release)."""
    cwd = Path.cwd()
    dirs = [
        cwd / "autotrainer",
        cwd.parent / "autotrainer",
    ]
    env = os.environ.get("GIGA_WATCH_DIR") or os.environ.get("GIGA_AUTOTRAINER_DIR")
    if env:
        dirs.insert(0, Path(env))
    seen = set()
    out = []
    for d in dirs:
        try:
            key = str(d.resolve())
        except OSError:
            key = str(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _atomic_write(path, text):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def _ball_xy_speed(state):
    """Return (x, y, speed_xy) or None."""
    if not isinstance(state, dict):
        return None
    ball = state.get("ball") or state.get("ball_phys") or {}
    if not isinstance(ball, dict):
        return None
    pos = ball.get("pos") or ball.get("position")
    vel = ball.get("vel") or ball.get("velocity")
    try:
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            x, y = float(pos[0]), float(pos[1])
        else:
            x = float(ball.get("x", 0.0))
            y = float(ball.get("y", 0.0))
        if isinstance(vel, (list, tuple)) and len(vel) >= 2:
            speed = (float(vel[0]) ** 2 + float(vel[1]) ** 2) ** 0.5
        else:
            vx = float(ball.get("vx", ball.get("vel_x", 999.0)))
            vy = float(ball.get("vy", ball.get("vel_y", 999.0)))
            speed = (vx * vx + vy * vy) ** 0.5
        return x, y, speed
    except (TypeError, ValueError):
        return None


def _kickoff_reset(state):
    """Heuristic: ball near center with low speed after having moved away."""
    global _prev_ball_xy
    info = _ball_xy_speed(state)
    if info is None:
        return False
    x, y, speed = info
    near = (x * x + y * y) ** 0.5 < 200.0 and speed < 80.0
    was_away = False
    if _prev_ball_xy is not None:
        px, py = _prev_ball_xy
        was_away = (px * px + py * py) ** 0.5 > 800.0
    _prev_ball_xy = (x, y)
    return near and was_away


def _detect_goal(state, now):
    """Heuristic goal on kickoff teleport: ball was deep in a net, now near center."""
    global _goal_cooldown_until, _prev_ball_y, _goals_for, _goals_against
    info = _ball_xy_speed(state)
    if info is None:
        return None
    x, y, speed = info
    near_center = (x * x + y * y) ** 0.5 < 250.0 and speed < 120.0
    scored = None
    if (
        near_center
        and now >= _goal_cooldown_until
        and _prev_ball_y is not None
        and abs(_prev_ball_y) > 4800.0
    ):
        # Positive Y ≈ orange net → blue scored (team 0 / "for" if we are blue)
        # Negative Y ≈ blue net → orange scored
        if _prev_ball_y > 4800.0:
            _goals_for += 1
            scored = "goal_for"
        else:
            _goals_against += 1
            scored = "goal_against"
        _goal_cooldown_until = now + 2.0
    _prev_ball_y = y
    return scored


def _player0(state):
    if not isinstance(state, dict):
        return None
    players = state.get("players") or []
    if not players:
        return None
    # Prefer labeled "You", else first non-demoed, else [0]
    for p in players:
        if isinstance(p, dict) and str(p.get("name") or "") == "You":
            return p
    for p in players:
        if isinstance(p, dict) and not p.get("is_demoed"):
            return p
    return players[0] if isinstance(players[0], dict) else None


def _proxy_from_player(player):
    """touch (0/1), air (0/1), boost_use (0/1), speed (uu/s)."""
    if not isinstance(player, dict):
        return 0, 0, 0, 0.0
    touch = 1 if player.get("ball_touched") else 0
    on_ground = player.get("on_ground")
    air = 0 if on_ground else 1
    controls = player.get("controls") or {}
    boost_use = 1 if controls.get("boost") else 0
    phys = player.get("phys") or {}
    vel = phys.get("vel") or phys.get("velocity")
    speed = 0.0
    try:
        if isinstance(vel, (list, tuple)) and len(vel) >= 3:
            speed = (float(vel[0]) ** 2 + float(vel[1]) ** 2 + float(vel[2]) ** 2) ** 0.5
        elif isinstance(vel, (list, tuple)) and len(vel) >= 2:
            speed = (float(vel[0]) ** 2 + float(vel[1]) ** 2) ** 0.5
    except (TypeError, ValueError):
        speed = 0.0
    return touch, air, boost_use, speed


def _push_event(kind, detail=None):
    _events_recent.append(
        {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "unix": time.time(),
            "kind": kind,
            "detail": detail,
        }
    )


def _roll_stats(now):
    """Aggregate last _ROLL_WINDOW_S of samples."""
    cutoff = now - _ROLL_WINDOW_S
    while _roll_samples and _roll_samples[0][0] < cutoff:
        _roll_samples.popleft()
    if not _roll_samples:
        return {}
    n = len(_roll_samples)
    r_sum = t_sum = a_sum = b_sum = s_sum = 0.0
    head_acc = {}
    for _, reward, touch, air, boost_use, speed, heads in _roll_samples:
        if reward is not None:
            r_sum += reward
        t_sum += touch
        a_sum += air
        b_sum += boost_use
        s_sum += speed
        if isinstance(heads, dict):
            for k, v in heads.items():
                try:
                    head_acc[k] = head_acc.get(k, 0.0) + float(v)
                except (TypeError, ValueError):
                    pass
    out = {
        "window_s": _ROLL_WINDOW_S,
        "samples": n,
        "avg_reward": r_sum / n,
        "touch_rate": t_sum / n,
        "air_frac": a_sum / n,
        "boost_use": b_sum / n,
        "speed": s_sum / n,
    }
    if head_acc:
        head_avg = {k: v / n for k, v in head_acc.items()}
        top = sorted(head_avg.items(), key=lambda x: abs(x[1]), reverse=True)
        out["reward_heads"] = head_avg
        out["top_heads"] = top[:8]
    return out


def _episode_summary():
    if _ep_samples <= 0:
        return None
    n = float(_ep_samples)
    return {
        "steps": _episode_steps,
        "samples": _ep_samples,
        "avg_reward": _ep_reward_sum / n,
        "touch_rate": _ep_touch / n,
        "air_frac": _ep_air / n,
        "boost_use": _ep_boost / n,
        "speed": _ep_speed_sum / n,
    }


def _reset_episode_acc():
    global _ep_touch, _ep_air, _ep_boost, _ep_speed_sum, _ep_reward_sum, _ep_samples
    _ep_touch = 0
    _ep_air = 0
    _ep_boost = 0
    _ep_speed_sum = 0.0
    _ep_reward_sum = 0.0
    _ep_samples = 0


def _maybe_append_history(dirs, point):
    """Append source:render sample to metrics_history.json (throttled)."""
    global _hist_next_t
    now = time.time()
    if now < _hist_next_t:
        return
    _hist_next_t = now + max(0.5, _HISTORY_INTERVAL_S)

    for d in dirs:
        path = d / "metrics_history.json"
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(data, list):
                    points = data
                    cap = _HISTORY_CAP
                else:
                    points = list(data.get("points") or [])
                    try:
                        cap = int(data.get("cap") or _HISTORY_CAP)
                    except (TypeError, ValueError):
                        cap = _HISTORY_CAP
            else:
                points = []
                cap = _HISTORY_CAP
            # Skip near-duplicate of last render point
            if points:
                last = points[-1]
                if (
                    isinstance(last, dict)
                    and last.get("source") == "render"
                    and last.get("avg_reward") == point.get("avg_reward")
                    and abs(float(last.get("step_reward") or 0) - float(point.get("step_reward") or 0)) < 1e-9
                ):
                    continue
            points.append(point)
            if len(points) > max(50, min(20_000, cap)):
                points = points[-cap:]
            out = {
                "updated_at": point.get("t") or time.strftime("%Y-%m-%dT%H:%M:%S"),
                "cap": cap,
                "points": points,
            }
            _atomic_write(path, json.dumps(out, separators=(",", ":")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass


def _maybe_write_render_hud(extra, state=None):
    """Persist visualizer_telemetry.json + legacy render_hud.json (throttled)."""
    global _hud_next_t, _hud_roll, _hud_history
    global _session_steps, _episode_steps, _episodes, _last_episode_steps
    global _last_touch_unix, _last_episode_summary
    global _ep_touch, _ep_air, _ep_boost, _ep_speed_sum, _ep_reward_sum, _ep_samples
    if not extra:
        return
    training = extra.get("training")
    if not isinstance(training, dict):
        return
    now = time.time()
    if now < _hud_next_t:
        return
    _hud_next_t = now + _HUD_INTERVAL_S

    _session_steps += 1
    _episode_steps += 1

    # Episode / kickoff
    if _kickoff_reset(state):
        _last_episode_summary = _episode_summary()
        _last_episode_steps = _episode_steps
        _episodes += 1
        _episode_steps = 1
        _reset_episode_acc()
        _push_event("kickoff")

    # Goal heuristic (uses ball teleport to center)
    goal_kind = _detect_goal(state, now)
    if goal_kind:
        _push_event(goal_kind)

    total = training.get("total_reward")
    if total is None:
        total = training.get("player0_total")
    try:
        total_f = float(total) if total is not None else None
    except (TypeError, ValueError):
        total_f = None
    if total_f is not None:
        if _hud_roll is None:
            _hud_roll = total_f
        else:
            _hud_roll = 0.92 * _hud_roll + 0.08 * total_f
        _hud_history.append(total_f)
        if len(_hud_history) > _HUD_HISTORY_CAP:
            _hud_history = _hud_history[-_HUD_HISTORY_CAP:]

    rewards = training.get("rewards") if isinstance(training.get("rewards"), dict) else {}
    top = []
    head_weighted = {}
    for name, info in rewards.items():
        try:
            if isinstance(info, dict):
                w = float(info.get("weighted", info.get("raw", 0.0)))
            else:
                w = float(info)
        except (TypeError, ValueError):
            continue
        top.append((str(name), w))
        head_weighted[str(name)] = w
    top.sort(key=lambda x: abs(x[1]), reverse=True)

    # Proxies from player0 / "You"
    p0 = _player0(state)
    touch, air, boost_use, speed = _proxy_from_player(p0)
    if touch:
        _last_touch_unix = now
    elif _last_touch_unix is None:
        _last_touch_unix = now  # start clock on first sample

    no_touch_s = None if _last_touch_unix is None else max(0.0, now - _last_touch_unix)
    if no_touch_s is not None and no_touch_s >= _NO_TOUCH_TIMEOUT_S:
        # Emit at most once per timeout window
        recent_kinds = {e.get("kind") for e in list(_events_recent)[-3:]}
        if "no_touch_timeout" not in recent_kinds:
            _push_event("no_touch_timeout", {"seconds": round(no_touch_s, 1)})

    # Accumulators
    _ep_touch += touch
    _ep_air += air
    _ep_boost += boost_use
    _ep_speed_sum += speed
    if total_f is not None:
        _ep_reward_sum += total_f
    _ep_samples += 1
    _roll_samples.append((now, total_f, touch, air, boost_use, speed, head_weighted))

    roll30 = _roll_stats(now)
    ep_cur = _episode_summary()

    updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload = {
        "updated_at": updated_at,
        "updated_unix": now,
        "source": "render",
        "watch_only": True,
        "total_timesteps": extra.get("total_timesteps") or training.get("timesteps"),
        "total_reward": total_f,
        "player0_total": training.get("player0_total"),
        "roll_avg": _hud_roll,
        "avg_reward": _hud_roll,
        "step_reward": total_f,
        "rewards": rewards,
        "top": top[:8],
        "reward_heads_full": head_weighted,
        "recent": list(_hud_history),
        "session_steps": _session_steps,
        "episode_steps": _episode_steps,
        "episodes": _episodes,
        "last_episode_steps": _last_episode_steps,
        # Proxies (instant + rolling)
        "proxies": {
            "touch": touch,
            "air": air,
            "boost_use": boost_use,
            "speed": speed,
            "no_touch_s": no_touch_s,
        },
        "touch_rate": roll30.get("touch_rate"),
        "air_frac": roll30.get("air_frac"),
        "boost_use_rate": roll30.get("boost_use"),
        "avg_speed": roll30.get("speed"),
        "goals_for": _goals_for,
        "goals_against": _goals_against,
        "roll_30s": roll30,
        "episode_current": ep_cur,
        "episode_last": _last_episode_summary,
        "events_recent": list(_events_recent),
        "limits": {
            "note": "Reads --render process telemetry (trainingDiag + GameState), not RocketSimVis internals",
            "no_touch_timeout_s": _NO_TOUCH_TIMEOUT_S,
            "roll_window_s": _ROLL_WINDOW_S,
        },
    }

    text = json.dumps(payload, separators=(",", ":"))
    dirs = _watch_dirs()
    for d in dirs:
        _atomic_write(d / "visualizer_telemetry.json", text)
        # Legacy companion HUD path
        _atomic_write(d / "render_hud.json", text)

    hist_point = {
        "t": updated_at,
        "ts": payload.get("total_timesteps"),
        "avg_reward": _hud_roll,
        "step_reward": total_f,
        "episode_steps": _episode_steps,
        "session_steps": _session_steps,
        "touch_rate": roll30.get("touch_rate"),
        "air_frac": roll30.get("air_frac"),
        "source": "render",
        "entropy": None,
        "sps": None,
    }
    _maybe_append_history(dirs, hist_point)


def send_data_to_rsvis(j, gamemode, actions=None, extra=None):
    json_out = {}
    json_out["gamemode"] = gamemode

    ball = dict(j["ball"])
    ball.pop("forward", None)
    ball.pop("right", None)
    ball.pop("up", None)
    json_out["ball_phys"] = ball

    json_out["cars"] = []
    players = j.get("players") or []
    for idx, player in enumerate(players):
        car = dict(player)
        # Prefer embedded controls; else attach from top-level actions[]
        if car.get("controls") is None and actions is not None and idx < len(actions):
            controls = _action_to_controls(actions[idx])
            if controls is not None:
                car["controls"] = controls
        if "has_flip" in car:
            car["has_flip"] = bool(car["has_flip"])
        json_out["cars"].append(car)

    if "boost_pads" in j:
        json_out["boost_pad_states"] = j["boost_pads"]

    if extra:
        for key in ("total_timesteps", "render", "training"):
            if key in extra and extra[key] is not None:
                json_out[key] = extra[key]
        _maybe_write_render_hud(extra, state=j)

    sock.sendto(json.dumps(json_out).encode(), (UDP_IP, UDP_PORT))


def render_state(state_json_str):
    j = json.loads(state_json_str)
    try:
        if "state" in j:
            extra = {
                "total_timesteps": j.get("total_timesteps"),
                "render": j.get("render"),
                "training": j.get("training"),
            }
            send_data_to_rsvis(j["state"], j.get("gamemode", "soccar"), j.get("actions"), extra)
        else:
            send_data_to_rsvis(j, j.get("gamemode", "soccar"), j.get("actions"), None)
    except Exception:
        print("Exception while sending data:")
        traceback.print_exc()
