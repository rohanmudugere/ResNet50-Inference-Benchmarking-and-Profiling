Benchmarking and profiling pre-trained ResNet50 inference across PyTorch Eager, PyTorch Compiled, and Core ML backends using Metal System Trace on a MacBook Pro M4 Pro

NOTE: torch 2.7.0 is the latest version that has been tested with coremltools 9.0, but torch.compile on mps is experimental on version 2.7.0

Inference Benchmarking & Profiling Data:
https://docs.google.com/spreadsheets/d/1vuUf47wj6R930EpwFpNKTHKsXw0Jz0PX1xqlmF9NMDQ/edit?gid=0#gid=0

PyTorch Compiled execution granted a 1.26x speedup in _Mean Inference Latency_ over PyTorch Eager, while Core ML granted a 1.97x speedup in _Mean Inference Latency_ over PyTorch Eager. This resulted in proportional gains in _Inference Throughput_ over PyTorch Eager for both PyTorch Compiled and Core ML (25.84% and 97.08% respectively).

In comparison to PyTorch Eager execution, PyTorch Compiled allowed for greater kernel dispatch and efficiency. _Average Compute Shader Launch Utilization_ was 126.47% higher, resulting in _Average Compute SIMD Groups Inflight_ being 48.04% higher. Additionally, _Average Kernel Occupancy_ was 48.08% higher, resulting in _Average Total Occupancy_ being 45.19% higher. These performance gains be explained by the differences between PyTorch Eager and Compiled execution. In Eager mode, PyTorch executes operations immediately as Python interprets them line by line. This is convenient in terms of debugging and program control flow, but incurs Python interpreter/kernel launch overhead on each operation and does not allow for cross-operation optimization (as operations are "seen" independently line by line). However in Compiled mode, PyTorch traces the Python code to capture a computational graph. This graph is then further optimized (via techniques such as operator fusion), producing a joint forward/backward graph ahead of time. This optimized computational graph is then compiled to kernels, and the artifact is cached. This results in lower Python interpreter/kernel launch overhead, as subsequent calls skip re-tracing and simply execute the compiled optimized kernels.  

In comparison to PyTorch Eager execution, Core ML allowed for more efficient memory usage and access. _Average GPU Bandwidth_ was 79.84% lower, while _Average Buffer L1 Miss Rate_ was 64.27% lower.
