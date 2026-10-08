"""On-demand recovery for trend plots whose startup computation failed (#157).

Trend plots are computed once at startup (pre-fork) and their HTML used to be
served for the life of the process. If Virtuoso was still coming up at that
moment, the fallback "Data Unavailable" HTML was served until the next restart,
and the plots' downloads 404'd because their data/figure caches were never
populated.

`TrendPlotRecovery` re-runs the startup task behind such a plot when it is
requested. Re-running the task also repopulates the data and figure caches as a
side effect, which fixes the downloads. Attempts are serialised per task and
rate-limited, so a page full of failing plots cannot become a query storm
against an endpoint that is already struggling.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


def run_tasks_with_deadline(
    tasks: Mapping[str, Callable[[], Any]],
    max_workers: int,
    timeout: float,
) -> Dict[str, Any]:
    """Run named tasks in parallel and collect what finishes within ``timeout``.

    Used for the startup precompute (#170). A task that raises, or is still
    running at the deadline, maps to ``None``, which the recovery path treats as
    a failed startup plot. Stragglers are abandoned rather than awaited (a thread
    can't be killed); queued tasks that never started are cancelled.
    """
    results: Dict[str, Any] = {}
    executor = ThreadPoolExecutor(max_workers=max_workers)
    future_to_name = {executor.submit(fn): name for name, fn in tasks.items()}
    done, not_done = wait(future_to_name, timeout=timeout)
    for future in done:
        name = future_to_name[future]
        try:
            results[name] = future.result()
            logger.info(f"Plot {name} completed successfully")
        except Exception as e:
            logger.error(f"Plot {name} failed: {e}")
            results[name] = None
    for future in not_done:
        name = future_to_name[future]
        logger.error(
            f"Plot {name} did not finish within the {timeout}s startup deadline; "
            f"it will be recomputed on demand"
        )
        results[name] = None
    executor.shutdown(wait=False, cancel_futures=True)
    return results


def is_usable_plot_html(html: Any) -> bool:
    """True for real plot HTML; False for empty output or a create_fallback_plot() placeholder."""
    return isinstance(html, str) and bool(html.strip()) and 'Data Unavailable' not in html


def result_has_usable_html(result: Any) -> bool:
    """True if a plot function's result carries real plot HTML.

    A result is a single HTML string or a tuple that mixes HTML strings with
    data (e.g. ``(abs_html, delta_html, df)``); every HTML string in it must be
    usable. None, fallbacks, and results without any HTML count as failures.
    """
    if isinstance(result, str):
        return is_usable_plot_html(result)
    if isinstance(result, (tuple, list)):
        html = [item for item in result if isinstance(item, str)]
        return bool(html) and all(is_usable_plot_html(item) for item in html)
    return False


class TrendPlotRecovery:
    """Recompute failed trend plots on demand.

    Args:
        functions: startup task name -> plot function.
        outputs: plot name -> (task name, index of the plot's HTML in the task's
            result tuple, or None when the task returns a single HTML string).
        run: wrapper used to execute a plot function (safe_plot_execution).
        cooldown: minimum seconds between recompute attempts for one task.
        clock: time source, injectable for tests.
    """

    def __init__(self, functions: Mapping[str, Callable], outputs: Mapping[str, Tuple[str, Optional[int]]],
                 run: Callable[[Callable], Any], cooldown: float, clock: Callable[[], float] = time.time):
        self._functions = functions
        self._outputs = outputs
        self._run = run
        self._cooldown = cooldown
        self._clock = clock
        self._recovered: Dict[str, str] = {}
        self._last_attempt: Dict[str, float] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def __contains__(self, plot_name: str) -> bool:
        return plot_name in self._outputs

    def is_recovered(self, plot_name: str) -> bool:
        """True once `plot_name` has been successfully recomputed."""
        return plot_name in self._recovered

    def html(self, plot_name: str, startup_html: Any) -> Any:
        """Return the best HTML for `plot_name`, recovering it if the startup copy is unusable."""
        if plot_name in self._recovered:
            return self._recovered[plot_name]
        if callable(startup_html) or is_usable_plot_html(startup_html):
            return startup_html
        return self.recover(plot_name) or startup_html

    def recover(self, plot_name: str) -> Optional[str]:
        """Recompute the task behind `plot_name`. Returns its HTML, or None if still unavailable."""
        output = self._outputs.get(plot_name)
        if output is None:
            return None
        task = output[0]

        with self._guard:
            lock = self._locks.setdefault(task, threading.Lock())

        with lock:
            # Another thread may have recovered this task while we waited.
            if plot_name in self._recovered:
                return self._recovered[plot_name]
            now = self._clock()
            last = self._last_attempt.get(task)
            if last is not None and now - last < self._cooldown:
                return None
            self._last_attempt[task] = now

            logger.info(f"Recomputing trend task {task} for {plot_name} after a failed startup run")
            try:
                result = self._run(self._functions[task])
            except Exception as e:  # run() is normally safe_plot_execution, which should not raise
                logger.error(f"Recompute of trend task {task} failed: {e}")
                return None

            for name, (output_task, index) in self._outputs.items():
                if output_task != task:
                    continue
                try:
                    html = result if index is None else result[index]
                except (TypeError, IndexError, KeyError):
                    continue
                if is_usable_plot_html(html):
                    self._recovered[name] = html

        recovered = self._recovered.get(plot_name)
        if recovered is None:
            logger.warning(f"Trend plot {plot_name} still unavailable after recompute")
        return recovered
