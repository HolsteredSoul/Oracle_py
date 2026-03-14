"""Oracle Python Agent — Entry point (scheduler stub)."""

import logging
import signal
import time

from apscheduler.schedulers.background import BackgroundScheduler

from src.logging_setup import configure_logging
from src.scanner.manifold import get_markets

configure_logging()

logger = logging.getLogger(__name__)


def scan_cycle() -> None:
    """Run one scan cycle: fetch markets and log the count."""
    try:
        markets = get_markets()
        logger.info("Scan cycle started, found %d markets", len(markets))
    except Exception as exc:  # noqa: BLE001
        logger.error("Scan cycle failed: %s", exc)


def main() -> None:
    scheduler = BackgroundScheduler()
    scheduler.add_job(scan_cycle, "interval", minutes=30, id="scan", next_run_time=__import__("datetime").datetime.now())
    scheduler.start()
    logger.info("Oracle agent started. Scan interval: 30 minutes. Press Ctrl+C to stop.")

    stop = False

    def _shutdown(signum, frame):  # noqa: ANN001
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while not stop:
        time.sleep(1)

    logger.info("Shutting down scheduler…")
    scheduler.shutdown(wait=False)
    logger.info("Oracle agent stopped.")


if __name__ == "__main__":
    main()
