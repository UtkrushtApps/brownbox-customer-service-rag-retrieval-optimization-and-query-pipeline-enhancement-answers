import sys
import os
import json
from embedding import EmbeddingGenerator
from chunking import overlapping_chunks, semantic_boundary_chunking, attach_metadata
from retrieval import VectorRetriever
from context_builder import ContextBuilder
from prompting import build_prompt
from evaluation import recall_at_k, precision_at_k, measure_latency, log_run
# import tokenizer and vector DB per your infra

# Placeholder tokenizer (replace with your LLM's tokenizer)
class DummyTokenizer:
    def tokenize(self, text):
        return text.split()

def load_sample_data():
    # Load support logs and metadata (simulate or read from file/db)
    # This would be replaced by actual data loading logic
    # For demo, returns: [(log_str, {meta})]
    return [
      ("[Agent:Bob,Exp:Senior,Sentiment:Positive] Hi, how may I assist?\n[Customer] I can't login.\n[Agent] Have you tried resetting your password?", {'agent': 'Bob', 'exp': 'Senior', 'sentiment': 'Positive'}),
      ("[Agent:Alice,Exp:Junior,Sentiment:Negative] Sorry for the wait.\n[Customer] Item not received.\n[Agent] I'll check your order.", {'agent': 'Alice', 'exp': 'Junior', 'sentiment': 'Negative'}),
    ]

def main():
    # STEP 1: Load and Process Data
    data = load_sample_data()
    chunked_all = []
    for log, meta in data:
        chunks = semantic_boundary_chunking(log, min_len=30, max_len=80)
        meta_chunks = attach_metadata(chunks, meta)
        chunked_all.extend(meta_chunks)

    # STEP 2: Embedding
    texts = [c['text'] for c in chunked_all]
    embedder = EmbeddingGenerator()
    vectors = embedder.embed_texts(texts)

    # STEP 3: Index Data to Vector DB
    # Placeholder: replace with your actual vector DB interface
    class DummyVectorDB:
        def __init__(self):
            # Each entry: {'id': i, 'text': ..., 'embedding': ..., 'metadata': {...}}
            self.entries = []
        def add(self, batches):
            offset = len(self.entries)
            for i, b in enumerate(batches):
                entry = b.copy()
                entry['id'] = offset+i
                self.entries.append(entry)
        def query(self, query_embeddings, n_results, where, include, distance_metric):
            # Simple cosine distance for demo
            import numpy as np
            results = {'documents': [], 'metadatas': [], 'distances': []}
            for qv in query_embeddings:
                dists = []
                for entry in self.entries:
                    if where:
                        match = True
                        for k, v in where.items():
                            if entry.get(k) != v:
                                match = False
                        if not match:
                            continue
                    v = entry['embedding']
                    v = np.array(v)
                    score = np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v) + 1e-9) if distance_metric=='cosine' else np.dot(qv, v)
                    dists.append((score, entry))
                # Sort by score desc
                dists.sort(key=lambda x: -x[0])
                picks = dists[:n_results]
                results['documents'].append([e['text'] for _,e in picks])
                results['metadatas'].append([e for _,e in picks])
                results['distances'].append([s for s,_ in picks])
            return results
    vecdb = DummyVectorDB()
    for chunk, vec in zip(chunked_all, vectors):
        chunk_cp = chunk.copy()
        chunk_cp['embedding'] = vec.tolist()
        vecdb.add([chunk_cp])

    # STEP 4: Retrieval and Query Pipeline
    retriever = VectorRetriever(vecdb, embedder.embed_texts)
    tokenizer = DummyTokenizer()
    ctx_builder = ContextBuilder(tokenizer, max_tokens=128)

    # Load actual customer queries from sample_queries.txt
    queries = []
    with open('sample_queries.txt', 'r') as f:
        for line in f:
            if line.strip():
                queries.append(line.strip())
    # Simulate expected relevant doc indices, or leave blank
    expected_relevant = [[0],[1]] # For metric demo

    # Demonstration loop
    for i, query in enumerate(queries):
        # Example of metadata filter: only senior agents, positive sentiment
        meta_filter = {'exp':'Senior'} if 'login' in query else None
        # Retrieval
        results, latency = measure_latency(
            retriever.retrieve, [query], k=3, metric='cosine', filter_metadata=meta_filter
        )
        top_chunks = results[0]
        # Mitigation 1: Rerank to reduce dilution
        top_chunks = retriever.rerank_by_score([top_chunks], query, rerank_fn=None, top_k=2)[0]

        context = ctx_builder.build_context(top_chunks, query, reserved_tokens=24)
        # Prompt Engineering
        instructions = "Answer from the chat logs below as a BrownBox agent. Cite context if relevant."
        examples = [
            {'query': 'Customer: How do I reset my password?', 'answer': 'You can reset your password by following the link at ...'},
            {'query': 'My order did not arrive.', 'answer': "I'll check the tracking status for you."}
        ]
        prompt = build_prompt(context, query, examples=examples, instructions=instructions)

        # Replace with LLM call
        print(f'Prompt for query {i+1}:\n{prompt}\n---')
        response = f"[Dummy LLM] Simulated answer for '{query}'."

        # Evaluation
        log_run(query, context, top_chunks, response)
        print(f"Retrieval/Context Latency: {round(latency,3)}s")

    # Evaluation Example
    # Suppose you retrieved top3 indices from vecdb.entries; compute metrics
    predicted_indices = [[0,1,2],[1,0,2]]
    r5 = recall_at_k(expected_relevant, predicted_indices, k=2)
    p5 = precision_at_k(expected_relevant, predicted_indices, k=2)
    print(f"Recall@2: {round(r5,2)} Precision@2: {round(p5,2)}")
    print("See 'runlog.txt' for spot check logs.")

    # Failure Case Analysis and Mitigation
    # 1. Context Dilution (irrelevant context): Fix - rerank, reduce k, filter via metadata
    # 2. Retrieval miss for sentiment: Fix - add hybrid search fallback (BM25 or keyword if no relevant vector found)
    #   Here, as an example, retry with less strict filter if nothing found.
    #   (Omitted for brevity, but to apply: check if retrieval returns nothing, then rerun with less filter or do keyword search.)

if __name__ == '__main__':
    main()
