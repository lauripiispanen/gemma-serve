# gemma-serve

Testing/tuning llama.cpp for model serving using Gemma 3.

**Some benchmarks (M1 Max 32GB):**

* Single stream inference (108 tok/sec)
* Static batches of 4 (172 tok/sec, total time 19sec)
* Continuous batching (177 tok/sec, total time 16sec)
  * `-O3 -Wall -Wextra` (182 tok/sec, total time 16sec)


**Gemma 3 12b (Q5_K_M)**

* Single stream inference (22 tok/sec)
* Static batches of 4 (21.5 tok/sec, total time 680sec)
* Continuous batching (24.55 tok/sec, total time 599sec)
