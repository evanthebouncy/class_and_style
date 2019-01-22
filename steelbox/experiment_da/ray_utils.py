import math
import resource
import signal
import traceback

import ray


class TimeoutError(Exception):
    pass


class Timeout:
    def __init__(self, num_seconds):
        self.num_seconds = int(math.ceil(num_seconds))

    def __enter__(self):
        signal.signal(signal.SIGALRM, Timeout.raise_timeout)
        signal.alarm(self.num_seconds)

    def __exit__(self, *args):
        signal.alarm(0)

    @staticmethod
    def raise_timeout(signum, frame):
        raise Timeout.TimeoutError


@ray.remote
def time_and_memory_limited(fn, num_seconds=None, num_bytes=None):
    _, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if num_bytes is None:
        resource.setrlimit(resource.RLIMIT_DATA, (hard, hard))
    else:
        resource.setrlimit(resource.RLIMIT_DATA, (num_bytes, hard))

    try:
        if num_seconds is not None:
            with Timeout(num_seconds):
                return fn()
        return fn()
    except Exception as e:
        if hasattr(e, 'args'):
            arg_types = [type(arg).__name__ for arg in e.args]
        else:
            arg_types = []

        return ('exception', type(e).__name__, arg_types, traceback.format_exc())
    finally:
        resource.setrlimit(resource.RLIMIT_DATA, (hard, hard))

