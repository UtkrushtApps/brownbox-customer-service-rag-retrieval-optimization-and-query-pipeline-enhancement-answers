import time
from typing import List

def recall_at_k(relevant: List[List[int]], retrieved: List[List[int]], k: int = 5) -> float:
    total, found = 0, 0
    for rel_list, ret_list in zip(relevant, retrieved):
        rel_set = set(rel_list)
        found += int(len(rel_set.intersection(set(ret_list[:k]))) > 0)
        total += 1
    return found / total if total else 0.0

def precision_at_k(relevant: List[List[int]], retrieved: List[List[int]], k: int = 5) -> float:
    total, match = 0, 0
    for rel_list, ret_list in zip(relevant, retrieved):
        rel_set = set(rel_list)
        hit = sum(1 for ix in ret_list[:k] if ix in rel_set)
        match += hit
        total += min(k, len(ret_list))
    return match / total if total else 0.0

def measure_latency(fn, *args, **kwargs):
    start = time.time()
    out = fn(*args, **kwargs)
    end = time.time()
    return out, end - start

def log_run(query, context, retrieval, response, log_file='runlog.txt'):
    with open(log_file, 'a') as f:
        f.write(f"Query: {query}\n")
        f.write(f"Retrieved Context: {context}\n")
        f.write(f"Answer: {response}\n")
        f.write(f"---\n")
