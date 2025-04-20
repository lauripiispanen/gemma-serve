# gemma-serve

Testing/tuning llama.cpp for model serving using Gemma 3 1b.

**Some benchmarks (M1 Max 32GB):**

* Simple inference (108 tok/sec)
* Static batches of 4 (172 tok/sec, total time 19sec)
* Continuous batching (177 tok/sec, total time 16sec)
