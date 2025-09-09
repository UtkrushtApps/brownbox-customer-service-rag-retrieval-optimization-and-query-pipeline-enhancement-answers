import re
from typing import List, Dict, Any

def overlapping_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks

def semantic_boundary_chunking(log: str, min_len: int = 40, max_len: int = 150) -> List[str]:
    # Chunk by semantic boundaries: sentences, paragraphs, agent/customer turn
    lines = [l.strip() for l in re.split(r'\n+', log) if l.strip()]
    chunks, curr = [], []
    curr_len = 0
    for line in lines:
        tokens = line.split()
        curr_len += len(tokens)
        curr.append(line)
        if min_len <= curr_len <= max_len:
            chunks.append(' '.join(curr))
            curr, curr_len = [], 0
        elif curr_len > max_len:
            # force break
            chunks.append(' '.join(curr))
            curr, curr_len = [], 0
    if curr:
        chunks.append(' '.join(curr))
    return chunks

def attach_metadata(chunks: List[str], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Attach metadata to each chunk
    results = []
    for chunk in chunks:
        entry = meta.copy()
        entry['text'] = chunk
        results.append(entry)
    return results
