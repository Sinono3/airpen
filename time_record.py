import time
import torch
import ml.record

start = time.perf_counter()
ml.record.record(500, torch.device("cpu"))
end = time.perf_counter()
elapsed = end - start
print(f"Elapsed: {elapsed:.6f} seconds")
