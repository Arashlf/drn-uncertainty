import sys
import threading
import time

from tqdm import tqdm

_stdout = sys.stdout
_stderr = sys.stderr

BOLD = "\033[1m"
DIM  = "\033[2m"
RESET = "\033[0m"

_pending_writes = []

step_info: dict = {}

SHOW_BANNERS = True

_depth = 0


def deferred_write(msg):
    _pending_writes.append(msg)


def run_pipeline(title, items, steps, item_label="items", item_format=None):
    global _depth

    if item_format is None:
        item_format = str

    show = SHOW_BANNERS and _depth == 0
    _depth += 1
    try:
        return _run_pipeline(title, items, steps, item_label, item_format, show)
    finally:
        _depth -= 1


def _run_pipeline(title, items, steps, item_label, item_format, show):
    items = list(items)
    timings = {}

    if show:
        _stdout.write(f"\n{BOLD}{'═'*56}\n")
        _stdout.write(f"  🚀  {title}  —  {len(items)} {item_label}, {len(steps)} steps each\n")
        _stdout.write(f"{'═'*56}{RESET}\n\n")
        _stdout.flush()

    t_start = time.time()

    for item in items:
        item_t0 = time.time()
        ctx = {"item": item}
        label_str = item_format(item)

        pbar = None
        if show:
            pbar = tqdm(
                total=len(steps),
                desc=f"  {label_str:<14s}",
                bar_format="{desc} {bar}| {n_fmt}/{total_fmt} {postfix} [{elapsed}]",
                leave=False,
                file=_stderr,
            )

        for emoji, step_label, fn in steps:
            step_info.clear()
            if pbar is not None:
                pbar.set_postfix_str(f"{emoji} {step_label}")
                pbar.refresh()

            done = threading.Event()
            exc_holder = [None]

            def _worker(fn=fn, ctx=ctx):
                try:
                    fn(ctx)
                except Exception as exc:
                    exc_holder[0] = exc
                finally:
                    done.set()

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()
            step_t0 = time.time()
            info_shown = False

            while not done.wait(timeout=1.0):
                if pbar is None:
                    continue
                step_elapsed = time.time() - step_t0

                if info_shown:
                    _stderr.write("\033[1A")

                pbar.set_postfix_str(f"{emoji} {step_label} [{step_elapsed:.0f}s]")
                pbar.refresh()

                if step_info:
                    info_text = "    " + "  |  ".join(f"{k}: {v}" for k, v in step_info.items())
                    _stderr.write(f"\r\n{info_text}\r")
                    _stderr.flush()
                    info_shown = True

            if pbar is not None and info_shown:
                _stderr.write("\r\033[2K\033[1A")
                _stderr.flush()

            thread.join()

            if exc_holder[0] is not None:
                raise exc_holder[0]

            if pbar is not None:
                pbar.update(1)
                for msg in _pending_writes:
                    pbar.write(msg)
            _pending_writes.clear()

        dt = time.time() - item_t0
        timings[item] = dt

        if pbar is not None:
            pbar.close()
            n = len(steps)
            bar = "█" * 20
            tqdm.write(
                f"  {DIM}{label_str:<14s} {bar}| {n}/{n} ✅ {dt:.1f}s{RESET}",
                file=_stderr,
            )

    if show:
        elapsed = time.time() - t_start
        _stdout.write(f"\n{BOLD}{'─'*56}\n")
        _stdout.write(f"  ✅  {title} complete in {elapsed:.1f}s\n")
        _stdout.write(f"{'─'*56}{RESET}\n\n")
        _stdout.flush()
    return timings
