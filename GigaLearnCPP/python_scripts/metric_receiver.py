import site
import sys
import os
import threading

wandb_run = None
_log_lock = threading.Lock()
_log_count = 0


def _make_settings():
	"""Best-effort quiet settings across wandb 0.15–0.25."""
	import wandb
	kwargs_list = [
		dict(console="off", x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True),
		dict(console="off", _disable_stats=True, _disable_meta=True),
		dict(console="off"),
	]
	for kwargs in kwargs_list:
		try:
			return wandb.Settings(**kwargs)
		except TypeError:
			continue
	return None


def _sanitize_key(key):
	# wandb panels are happier without spaces in names
	return str(key).replace(" ", "_").replace("/", "_")


def init(py_exec_path, project, group, name, id=None):
	global wandb_run

	sys.executable = py_exec_path

	os.environ["WANDB_DISABLE_SERVICE"] = "true"
	os.environ.setdefault("WANDB_SILENT", "true")
	os.environ.setdefault("WANDB_CONSOLE", "off")
	os.environ.setdefault("WANDB_DISABLE_CODE", "true")
	os.environ.setdefault("WANDB_DISABLE_GIT", "true")
	os.environ.setdefault("WANDB_ERROR_REPORTING", "false")

	try:
		site_packages_dir = os.path.join(os.path.join(os.path.dirname(py_exec_path), "Lib"), "site-packages")
		sys.path.append(site_packages_dir)
		site.addsitedir(site_packages_dir)
		import wandb
	except Exception as e:
		raise Exception(f"""
			FAILED to import wandb! Make sure GigaLearnCPP isn't using the wrong Python installation.
			This installation's site packages: {site.getsitepackages()}
			Exception: {repr(e)}"""
		)

	print("Calling wandb.init()...", flush=True)
	settings = _make_settings()
	init_kwargs = dict(project=project, group=group, name=name)
	if id is not None and len(id) > 0:
		init_kwargs["id"] = id
		init_kwargs["resume"] = "allow"
	if settings is not None:
		init_kwargs["settings"] = settings

	try:
		wandb_run = wandb.init(**init_kwargs)
	except TypeError:
		init_kwargs.pop("settings", None)
		wandb_run = wandb.init(**init_kwargs)

	url = getattr(wandb_run, "url", None) or ""
	print(f"[metric_receiver] wandb ready id={wandb_run.id} project={project}", flush=True)
	if url:
		print(f"[metric_receiver] dashboard: {url}", flush=True)

	# Seed one point so the Charts tab is never empty after connect.
	try:
		wandb_run.log({"wandb_connected": 1}, step=0)
		print("[metric_receiver] logged seed metric wandb_connected=1", flush=True)
	except Exception as e:
		print(f"[metric_receiver] seed log failed: {e!r}", flush=True)

	return wandb_run.id


def add_metrics(metrics):
	"""Called from C++ under GIL (already throttled). Log sync — async threads drop data on wandb 0.25."""
	global _log_count
	if wandb_run is None:
		return

	raw = dict(metrics) if not isinstance(metrics, dict) else metrics
	payload = {}
	step = None
	for k, v in raw.items():
		try:
			fv = float(v)
		except (TypeError, ValueError):
			continue
		if not (fv == fv):  # NaN
			continue
		sk = _sanitize_key(k)
		payload[sk] = fv
		if sk in ("Total_Timesteps", "TotalTimesteps", "total_timesteps"):
			step = int(fv)

	if not payload:
		return

	try:
		with _log_lock:
			if step is not None and step >= 0:
				wandb_run.log(payload, step=step)
			else:
				wandb_run.log(payload)
			_log_count += 1
			if _log_count <= 3 or (_log_count % 20) == 0:
				print(f"[metric_receiver] logged #{_log_count} keys={len(payload)} step={step}", flush=True)
	except Exception as e:
		print(f"[metric_receiver] wandb.log failed: {e!r}", flush=True)
