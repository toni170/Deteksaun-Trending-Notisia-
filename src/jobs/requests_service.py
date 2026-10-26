"""
Pedidu dadus (koleta lote) ho jadwal automátiku.

- `start_collection_request(...)`: hahú pedidu (manuál ka automátiku) iha thread background.
- `AutoRequestScheduler`: thread ida ne'ebé hala'o pedidu automátiku kada N minutu (padraun 30),
  depois pedidu anterior remata. Hala'o hela maski tab browser taka, se server Streamlit sei moris.
Konfigurasaun rai iha `<batches>/auto_state.json` (la lakon bainhira restart).
"""
import json
import logging
import os
import threading
import time
from datetime import datetime

from src.collection.batch_collector import DEFAULT_TARGET_MB, collect_batch
from src.jobs.job_manager import JobManager

logger = logging.getLogger("requests_service")

COLLECTION_JOB = "koleta"
STATE_NAME = "auto_state.json"


class AutoState:
    """Konfigurasaun persistente (enabled, intervalu, orijen, tempu ikus)."""

    def __init__(self, batches_dir: str, defaults: dict):
        os.makedirs(batches_dir, exist_ok=True)
        self.path = os.path.join(batches_dir, STATE_NAME)
        self._lock = threading.RLock()
        self.data = {"enabled": False, "interval_minutes": 30, "target_mb": DEFAULT_TARGET_MB,
                     "delay_seconds": 1.0, "per_page": 100, "sources": [],
                     "last_finished_at": None, "last_auto_message": "", "last_auto_at": None}
        self.data.update(defaults)
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data.update(json.load(f))
        except (OSError, ValueError):
            pass

    def get(self, key, default=None):
        with self._lock:
            return self.data.get(key, default)

    def update(self, **kw):
        with self._lock:
            self.data.update(kw)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.data)


def _collection_job(progress, stop_event, *, sources, batches_dir, kind, target_mb, delay_seconds,
                    per_page, auto_state=None):
    result = None
    try:
        result = collect_batch(sources, batches_dir, kind=kind, target_mb=target_mb, per_page=per_page,
                               delay_seconds=delay_seconds, progress=progress, stop_event=stop_event)
    finally:
        # Sempre marka tempu remata (maski erru) atu jadwal la repete kada segundu.
        if auto_state is not None:
            extra = {"last_finished_at": datetime.now().isoformat(timespec="seconds")}
            if kind == "automátiku":
                extra["last_auto_at"] = extra["last_finished_at"]
                extra["last_auto_message"] = (
                    f"{result.rows:,} notísia · {result.csv_bytes / 1024 / 1024:.1f} MB · {result.message}"
                    if result is not None else "Pedidu falla (haree log).")
            auto_state.update(**extra)
    if result.status == "falla":
        raise RuntimeError(result.message)
    return result.to_dict()


def start_collection_request(manager: JobManager, batches_dir: str, sources: list, *, kind="manuál",
                             target_mb=DEFAULT_TARGET_MB, delay_seconds=1.0, per_page=100,
                             auto_state: AutoState = None):
    return manager.start(
        COLLECTION_JOB, _collection_job,
        kwargs=dict(sources=[dict(s) for s in sources], batches_dir=batches_dir, kind=kind,
                    target_mb=target_mb, delay_seconds=delay_seconds, per_page=per_page,
                    auto_state=auto_state),
    )


class AutoRequestScheduler:
    def __init__(self, manager: JobManager, batches_dir: str, state: AutoState, tick: float = 1.0):
        self.manager, self.batches_dir, self.state, self.tick = manager, batches_dir, state, tick
        self._thread = None
        self._stop = threading.Event()
        self._boot = time.time()

    # -- kontrola ----------------------------------------------------
    def ensure_started(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="auto-request", daemon=True)
            self._thread.start()

    def shutdown(self):
        self._stop.set()

    def configure(self, **kw):
        """Muda konfigurasaun (enabled, interval_minutes, target_mb, delay_seconds, sources)."""
        clean = {k: v for k, v in kw.items() if v is not None}
        if clean:
            self.state.update(**clean)

    # -- jadwal ------------------------------------------------------
    def next_run_epoch(self):
        """Segundu epoch pedidu automátiku tuir mai, ka None se dezativa."""
        s = self.state.snapshot()
        if not s.get("enabled"):
            return None
        interval = max(1, int(s.get("interval_minutes", 30))) * 60
        last = s.get("last_finished_at")
        if last:
            try:
                return datetime.fromisoformat(last).timestamp() + interval
            except ValueError:
                pass
        return self._boot + 5  # seidauk iha pedidu: hahú lalais depois server moris

    def seconds_until_next(self):
        nxt = self.next_run_epoch()
        return None if nxt is None else max(0.0, nxt - time.time())

    def _loop(self):
        while not self._stop.wait(self.tick):
            try:
                self.run_if_due()
            except Exception:
                logger.exception("Jadwal automátiku falla")

    def run_if_due(self) -> bool:
        wait = self.seconds_until_next()
        if wait is None or wait > 0 or self.manager.is_running(COLLECTION_JOB):
            return False
        s = self.state.snapshot()
        sources = s.get("sources") or []
        ok, msg = start_collection_request(
            self.manager, self.batches_dir, sources, kind="automátiku",
            target_mb=s.get("target_mb", DEFAULT_TARGET_MB), delay_seconds=s.get("delay_seconds", 1.0),
            per_page=s.get("per_page", 100), auto_state=self.state)
        if not ok:
            return False
        logger.info("Pedidu automátiku hahú (kada %s minutu).", s.get("interval_minutes"))
        return True


_SCHEDULERS = {}
_SCHED_LOCK = threading.Lock()


def get_scheduler(manager: JobManager, batches_dir: str, defaults: dict = None) -> AutoRequestScheduler:
    key = os.path.abspath(batches_dir)
    with _SCHED_LOCK:
        if key not in _SCHEDULERS:
            state = AutoState(batches_dir, defaults or {})
            _SCHEDULERS[key] = AutoRequestScheduler(manager, batches_dir, state)
        sched = _SCHEDULERS[key]
    sched.ensure_started()
    return sched
