# ⚙️ Results Summary

| Version                                          | Concurrency Type                          | Time (seconds) | Speedup vs Sync  | Notes                                                                                               |
| ------------------------------------------------ | ----------------------------------------- | -------------- | ---------------- | --------------------------------------------------------------------------------------------------- |
| **Synchronous**                                  | Single-threaded (sequential blocking I/O) | **7.94 s**     | 1×               | Baseline — every request waits for the previous one to finish                                       |
| **Threading (`concurrent.futures`, 20 workers)** | OS threads (I/O-bound concurrency)        | **0.43 s**     | **~18× faster**  | Excellent scaling — threads efficiently overlap I/O waits                                           |
| **Asyncio (`aiohttp`)**                          | Single-threaded event loop (async I/O)    | **1.23 s**     | **~6.4× faster** | Still very fast, but slower than threads due to per-request coroutine overhead and connection setup |

---

### 🧩 Why These Numbers Make Sense

#### 1. **Threading Wins Here**

* `requests` uses blocking I/O under the hood.
* `ThreadPoolExecutor` + 20 workers → true parallelism for I/O waits (network latency hidden).
* Each thread runs `session.get()` independently, reusing its own TCP session via `thread_local`.

#### 2. **Asyncio Slightly Slower**

* `aiohttp` connections are async, but each call still needs to open its own socket (unless you reuse sessions carefully).
* Python’s async model avoids thread-switching but adds coroutine scheduling overhead.
* It excels when the concurrency count is **hundreds to thousands** — at 80 requests, threading has less overhead.

#### 3. **Synchronous**

* No concurrency, total latency is basically the sum of all network response times.

---

### ⚖️ **When Each Is Best**

| Use Case                   | Best Model                          | Reason                                                          |
| -------------------------- | ----------------------------------- | --------------------------------------------------------------- |
| Network requests (10–200)  | ✅ **Threading**                     | Simpler, high performance, minimal setup                        |
| High-volume HTTP (1000+)   | ✅ **Asyncio**                       | Scales without threads, lower memory footprint                  |
| CPU-heavy tasks            | ❌ Neither (use **multiprocessing**) | Threads and asyncio don’t parallelize CPU-bound code due to GIL |
| Simple scripts / debugging | ✅ **Sync**                          | Straightforward and predictable                                 |

---

### 🧠 Optional Experiment

If you want to **see asyncio outperform threading**, increase workload:

```python
DO_NOT_ABUSE_THE_SERVER = 400  # or use a mock local server
```

Then you’ll likely observe:

* Threaded version slows down (context-switch overhead)
* Asyncio stays more stable (event loop scales better)
