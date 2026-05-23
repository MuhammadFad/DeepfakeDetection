"""
Append-only training log — writes to output/training_log.txt with timestamps.
Flushes after every line so a crash leaves the log intact up to that point.
"""

import datetime
import os

_f = None
_path = None


def init(path: str):
    global _f, _path
    _path = path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _f = open(path, 'a', encoding='utf-8')
    _write('=' * 60)
    _write(f'Session started  {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    _write('=' * 60)


def log(msg: str):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    _write(f'[{ts}]  {msg}')


def _write(line: str):
    if _f is None:
        return
    _f.write(line + '\n')
    _f.flush()


def close():
    if _f:
        log('Session ended')
        _f.close()
