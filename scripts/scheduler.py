"""Keep this running to auto-fetch + analyze news once per day.

    python scripts/scheduler.py

Alternative on Windows: use Task Scheduler to run scripts/run_daily_update.py
directly — simpler than keeping a Python process alive.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apscheduler.schedulers.blocking import BlockingScheduler
from scripts.run_daily_update import main as run_update


def start(hour: int = 7, minute: int = 0):
    sched = BlockingScheduler()
    sched.add_job(run_update, "cron", hour=hour, minute=minute, id="daily-news")
    sched.add_job(run_update, "date")  # run once on startup too
    print(f"Scheduler running. Daily update at {hour:02d}:{minute:02d}. Ctrl+C to stop.")
    sched.start()


if __name__ == "__main__":
    start()
