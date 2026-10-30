| Backend | Size (MB) | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (img/s) | Val acc | Val macro-F1 | Agreement |
|---|---|---|---|---|---|---|---|---|
| pytorch fp32 | 44.8 | 61.61 | 152.99 | 213.83 | 19.2 | 1.000 | 1.000 | 1.000 |
| onnxruntime fp32 | 44.7 | 34.43 | 36.31 | 49.98 | 31.6 | 1.000 | 1.000 | 1.000 |
| onnxruntime int8 | 11.3 | 20.43 | 21.55 | 26.23 | 52.6 | 0.981 | 0.981 | 0.981 |

_Intel64 Family 6 Model 165 Stepping 2, GenuineIntel · 12 cores available · 1 thread(s) · onnxruntime 1.30.0 · latency = batch 1, throughput = batch 32_
