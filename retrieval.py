from typing import List, Dict, Any, Optional
import time

class VectorRetriever:
    def __init__(self, vector_db, embedding_fn):
        self.vector_db = vector_db
        self.embedding_fn = embedding_fn

    def retrieve(
        self,
        queries: List[str],
        k: int = 6,
        metric: str = 'cosine',
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        # Batch embedding
        query_vecs = self.embedding_fn(queries)
        if hasattr(self.vector_db, 'query'): # ChromaDB
            results = self.vector_db.query(
                query_embeddings=query_vecs,
                n_results=k,
                where=filter_metadata,
                include=['documents', 'metadatas', 'distances'],
                distance_metric=metric
            )
            grouped = []
            for docs, metas, dists in zip(results['documents'], results['metadatas'], results['distances']):
                # For each query group
                group = []
                for doc, meta, dist in zip(docs, metas, dists):
                    meta = meta or {}
                    meta['text'] = doc
                    meta['score'] = dist
                    group.append(meta)
                grouped.append(group)
            return grouped
        elif hasattr(self.vector_db, 'query_batch'): # Pinecone or similar
            batch_results = self.vector_db.query_batch(
                queries=query_vecs,
                top_k=k,
                metric=metric,
                filter=filter_metadata,
                include_metadata=True
            )
            return [r['matches'] for r in batch_results]
        else:
            raise ValueError('Unsupported vector database interface')

    @staticmethod
    def rerank_by_score(results: List[List[Dict[str, Any]]], query: str, rerank_fn=None, top_k=5) -> List[List[Dict[str, Any]]]:
        # Optionally rerank results by e.g. cross-encoder or semantic similarity to reduce context dilution
        if rerank_fn is None:
            # Default: return top_k by score
            return [sorted(group, key=lambda x: x.get('score', 0))[:top_k] for group in results]
        else:
            reranked = []
            for group in results:
                reranked_group = rerank_fn([x['text'] for x in group], query)
                # suppose rerank_fn returns indices
                final_group = [group[i] for i in reranked_group[:top_k]]
                reranked.append(final_group)
            return reranked
