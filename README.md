Benchmarking and profiling pre-trained ResNet50 inference across PyTorch Eager, PyTorch Compiled, and Core ML backends using Metal System Trace on a MacBook Pro M4 Pro

NOTE: torch 2.7.0 is the latest version that has been tested with coremltools 9.0, but torch.compile on mps is experimental on version 2.7.0

Inference Benchmarking & Profiling Data:
https://docs.google.com/spreadsheets/d/1vuUf47wj6R930EpwFpNKTHKsXw0Jz0PX1xqlmF9NMDQ/edit?gid=0#gid=0

PyTorch Compiled execution granted a 1.26x speedup in _Mean Inference Latency_ over PyTorch Eager, while Core ML granted a 1.97x speedup in _Mean Inference Latency_ over PyTorch Eager. This resulted in proportional gains in _Inference Throughput_ over PyTorch Eager for PyTorch Compiled and Core ML (25.84% and 97.08% respectively).

In comparison to PyTorch Eager execution, PyTorch Compiled allowed for greater kernel dispatch and efficiency. _Average Compute Shader Launch Utilization_ was 126.47% higher, resulting in _Average Compute SIMD Groups Inflight_ being 48.04% higher. Additionally, _Average Kernel Occupancy_ was 48.08% higher, resulting in _Average Total Occupancy_ being 45.19% higher.
