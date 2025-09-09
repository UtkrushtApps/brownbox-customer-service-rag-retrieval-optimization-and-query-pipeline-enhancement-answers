from typing import List, Dict

class ContextBuilder:
    def __init__(self, tokenizer, max_tokens=1024, overlap_delim="<END_CHUNK>"):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_delim = overlap_delim

    def build_context(self, chunks: List[Dict], query: str, reserved_tokens: int = 100) -> str:
        # Build as much context as fits inside max_tokens - reserved_tokens
        context = []
        used_tokens = 0
        for chunk in chunks:
            chunk_text = chunk['text']
            chunk_tokens = len(self.tokenizer.tokenize(chunk_text))
            if used_tokens + chunk_tokens > (self.max_tokens - reserved_tokens):
                break
            context.append(chunk_text)
            used_tokens += chunk_tokens + 1 # Account for delimiter
        return f"{self.overlap_delim}\n".join(context)
