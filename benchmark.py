import os
import time

import numpy as np

def benchmark_model(name, run_fn, sync_fn=None, warmup=20, iters=1000):
    def sync():
        if sync_fn is not None:
            sync_fn()

    for _ in range(warmup):
        run_fn()
        sync()

    print(f"BENCHMARK PID: {os.getpid()}")
    input("Attach Instrument and press Enter...")

    times = []

    for _ in range(iters):
        start = time.perf_counter()
        run_fn()
        sync()

        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    print(f"\n=== {name} ===")
    print(f"mean: {times.mean()*1000:.3f} ms")
    print(f"p50 : {np.percentile(times, 50)*1000:.3f} ms")
    print(f"p95 : {np.percentile(times, 95)*1000:.3f} ms")
    print(f"p99 : {np.percentile(times, 99)*1000:.3f} ms")

    return times
