"""
Jestór servisu background (thread) ba painel.

Servisu naruk (koleta, preprosesamentu, treinu) hala'o iha thread rasik, la iha Streamlit,
entaun la para bainhira utilizadór klik buat seluk iha painel. Painel de'it lee estadu
liuhusi `snapshot(naran)`.

Funsaun servisu: `func(progress, stop_event, **kwargs) -> dict`
  - `progress(fraksaun, mensajen, **extra)` atualiza estadu.
  - `stop_event` (threading.Event) = sinál atu para.
"""
import copy
import threading
import time
import traceback
import logging
from collections import deque
from datetime import datetime

logger = logging.getLogger("job_manager")

IDLE, RUNNING, DONE, STOPPED, FAILED = "idle", "running", "done", "stopped", "failed"
STATUS_TETUN = {IDLE: "Hein", RUNNING: "Hala'o hela", DONE: "Remata", STOPPED: "Para", FAILED: "Falla"}


class JobSink:
    def __init__(self, state: dict, lock: threading.RLock):
        self._state, self._lock = state, lock

    def __call__(self, fraction=None, message=None, **extra):
        with self._lock:
            if fraction is not None:
                self._state["fraction"] = max(0.0, min(float(fraction), 1.0))
            if message is not None:
                if message != self._state["message"]:
                    self._state["log"].append(f"{datetime.now():%H:%M:%S} · {message}")
                self._state["message"] = message
            if extra:
                self._state["extra"].update(extra)


class JobManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._jobs, self._stops, self._groups = {}, {}, {}

    def _new_state(self, name):
        return {"name": name, "status": IDLE, "fraction": 0.0, "message": "", "extra": {},
                "result": None, "error": None, "started_at": None, "finished_at": None,
                "log": deque(maxlen=200)}

    def is_running(self, name: str) -> bool:
        with self._lock:
            return self._jobs.get(name, {}).get("status") == RUNNING

    def any_running(self, group=None) -> bool:
        with self._lock:
            return any(s["status"] == RUNNING and (group is None or self._groups.get(n) == group)
                       for n, s in self._jobs.items())

    def start(self, name: str, func, kwargs=None, group=None, on_finish=None):
        """Hahú servisu. Retorna (susesu: bool, mensajen Tetun)."""
        with self._lock:
            if self.is_running(name):
                return False, "Servisu ne'e hala'o hela ona. Hein to'o remata."
            if group:
                busy = [n for n, s in self._jobs.items()
                        if s["status"] == RUNNING and self._groups.get(n) == group]
                if busy:
                    return False, "Servisu todan seluk hala'o hela. Hein to'o remata molok hahú servisu ne'e."
            state = self._new_state(name)
            state.update(status=RUNNING, started_at=time.time(), message="Hahú...")
            stop = threading.Event()
            self._jobs[name], self._stops[name], self._groups[name] = state, stop, group
        sink = JobSink(state, self._lock)

        def runner():
            try:
                result = func(sink, stop, **(kwargs or {}))
                with self._lock:
                    state["result"] = result
                    state["status"] = STOPPED if stop.is_set() else DONE
                    if state["status"] == DONE:
                        state["fraction"] = 1.0
            except Exception as e:
                logger.error("Servisu '%s' falla:\n%s", name, traceback.format_exc())
                with self._lock:
                    state["status"] = FAILED
                    state["error"] = getattr(e, "message", None) or str(e)
                    state["message"] = f"Erru: {state['error']}"
            finally:
                with self._lock:
                    state["finished_at"] = time.time()
                if on_finish:
                    try:
                        on_finish(self.snapshot(name))
                    except Exception:
                        logger.exception("on_finish falla")

        threading.Thread(target=runner, name=f"job-{name}", daemon=True).start()
        return True, "Servisu hahú."

    def stop(self, name: str) -> bool:
        with self._lock:
            if self.is_running(name):
                self._stops[name].set()
                self._jobs[name]["message"] = "Hein: sei para..."
                return True
        return False

    def snapshot(self, name: str) -> dict:
        with self._lock:
            state = self._jobs.get(name)
            if state is None:
                return self._new_state(name) | {"log": []}
            snap = copy.deepcopy({k: v for k, v in state.items() if k != "log"})
            snap["log"] = list(state["log"])
            return snap


_MANAGER = None
_MANAGER_LOCK = threading.Lock()


def get_job_manager() -> JobManager:
    """Instánsia úniku ba prosesu tomak (hanesan ba sesaun hotu-hotu)."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = JobManager()
        return _MANAGER
