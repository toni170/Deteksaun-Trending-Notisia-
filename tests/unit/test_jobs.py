"""Teste jestór servisu background no jadwal pedidu automátiku."""
import os
import sys
import threading
import time
import types

if "feedparser" not in sys.modules:
    try:
        import feedparser  # noqa: F401
    except ImportError:
        sys.modules["feedparser"] = types.ModuleType("feedparser")

from src.jobs.job_manager import DONE, FAILED, RUNNING, STOPPED, JobManager  # noqa: E402
from src.jobs import requests_service as rs  # noqa: E402


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_job_runs_in_background_and_reports_progress():
    m = JobManager()
    gate = threading.Event()

    def job(progress, stop):
        progress(0.5, "metade", rows=7)
        gate.wait(2)
        return {"ok": True}
    assert m.start("a", job)[0]
    assert wait_for(lambda: m.snapshot("a")["fraction"] == 0.5)
    snap = m.snapshot("a")
    assert snap["status"] == RUNNING and snap["extra"]["rows"] == 7 and snap["message"] == "metade"
    gate.set()
    assert wait_for(lambda: m.snapshot("a")["status"] == DONE)
    assert m.snapshot("a")["result"] == {"ok": True} and m.snapshot("a")["fraction"] == 1.0


def test_same_job_cannot_start_twice_and_exclusive_group():
    m = JobManager()
    gate = threading.Event()
    m.start("x", lambda p, s: gate.wait(2), group="todan")
    ok, msg = m.start("x", lambda p, s: None)
    assert not ok and "hala'o hela" in msg
    ok, msg = m.start("y", lambda p, s: None, group="todan")
    assert not ok and "todan" in msg
    assert m.start("z", lambda p, s: None)[0]  # grupu seluk/laiha grupu bele
    gate.set()


def test_failure_and_stop():
    m = JobManager()
    m.start("f", lambda p, s: 1 / 0)
    assert wait_for(lambda: m.snapshot("f")["status"] == FAILED)
    assert "Erru" in m.snapshot("f")["message"]

    def slow(progress, stop):
        stop.wait(3)
    m.start("s", slow)
    assert m.stop("s")
    assert wait_for(lambda: m.snapshot("s")["status"] == STOPPED)


def test_scheduler_runs_when_due_and_waits_interval_after_finish(tmp_path, monkeypatch):
    calls = []

    def fake_collect(sources, batches_dir, **kw):
        calls.append(kw["kind"])
        from src.collection.batch_collector import BatchResult
        return BatchResult(status="remata", rows=3, csv_bytes=1024, message="ok", finished_at="x")
    monkeypatch.setattr(rs, "collect_batch", fake_collect)

    m = JobManager()
    state = rs.AutoState(str(tmp_path), {"enabled": True, "interval_minutes": 30,
                                         "sources": [{"name": "T", "feed_url": "https://t/feed/"}]})
    sched = rs.AutoRequestScheduler(m, str(tmp_path), state)
    sched._boot = time.time() - 100            # server moris dala hira ona -> due
    assert sched.run_if_due() is True
    assert wait_for(lambda: m.snapshot(rs.COLLECTION_JOB)["status"] == DONE)
    assert calls == ["automátiku"]
    assert "3 notísia" in state.get("last_auto_message")

    wait = sched.seconds_until_next()          # depois remata: hein 30 minutu
    assert 29 * 60 < wait <= 30 * 60
    assert sched.run_if_due() is False and calls == ["automátiku"]


def test_scheduler_disabled_and_persisted(tmp_path):
    m = JobManager()
    state = rs.AutoState(str(tmp_path), {"enabled": False})
    sched = rs.AutoRequestScheduler(m, str(tmp_path), state)
    assert sched.seconds_until_next() is None and sched.run_if_due() is False
    sched.configure(enabled=True, interval_minutes=45)
    again = rs.AutoState(str(tmp_path), {})
    assert again.get("enabled") is True and again.get("interval_minutes") == 45


def test_scheduler_does_not_overlap_running_request(tmp_path, monkeypatch):
    m = JobManager()
    gate = threading.Event()
    m.start(rs.COLLECTION_JOB, lambda p, s: gate.wait(2))
    state = rs.AutoState(str(tmp_path), {"enabled": True})
    sched = rs.AutoRequestScheduler(m, str(tmp_path), state)
    sched._boot = time.time() - 100
    assert sched.run_if_due() is False
    gate.set()
