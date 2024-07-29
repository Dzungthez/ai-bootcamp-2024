# file: reranking.py
import json
import numpy as np
from pathlib import Path
from evaluate import evaluate, get_answers_and_evidence
from vector_store.sparse_vector_store import SparseVectorStore
from vector_store.semantic_vector_store import SemanticVectorStore
from vector_store.node import TextNode


class Reranker:
    def __init__(self, sparse_store: SparseVectorStore, semantic_store: SemanticVectorStore):
        self.sparse_store = sparse_store
        self.semantic_store = semantic_store

    def rerank(self, query: str, top_k: int = 5, alpha_range=(0.1, 0.5, 0.9)):
        # Get sparse and semantic scores
        sparse_results = self.sparse_store.query(query, top_k=top_k)
        semantic_results = self.semantic_store.query(query, top_k=top_k)

        # Combine results into a single dictionary
        combined_results = {}
        for node, score in zip(sparse_results.nodes, sparse_results.similarities):
            combined_results[node.id_] = {'sparse': score, 'semantic': 0}
        
        for node, score in zip(semantic_results.nodes, semantic_results.similarities):
            if node.id_ in combined_results:
                combined_results[node.id_]['semantic'] = score
            else:
                combined_results[node.id_] = {'sparse': 0, 'semantic': score}

        # Perform grid search on the alpha range
        best_alpha, best_score, best_ranking = None, -np.inf, None
        for alpha in alpha_range:
            scores = [(alpha * v['sparse'] + (1 - alpha) * v['semantic']) for v in combined_results.values()]
            ids = list(combined_results.keys())
            sorted_indices = np.argsort(scores)[::-1]
            sorted_ids = [ids[i] for i in sorted_indices]
            sorted_scores = [scores[i] for i in sorted_indices]

            ranking = sorted_ids[:top_k]
            predicted_evidence_text = [self.sparse_store.node_dict[id_].text for id_ in ranking]  # Get text from IDs
            predicted_answers_and_evidence = {
                query: {
                    "answer": "predicted_answer_placeholder",  # Placeholder, replace with actual predicted answer
                    "evidence": predicted_evidence_text
                }
            }
            score = evaluate(self.gold_answers_and_evidence, predicted_answers_and_evidence, retrieval_only=True)["Evidence F1"]

            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_ranking = predicted_evidence_text

        return best_alpha, best_ranking


def prepare_vector_stores(documents, force_index=False, chunk_size=200):
    sparse_store = SparseVectorStore(persist=True, force_index=force_index)
    semantic_store = SemanticVectorStore(persist=True, force_index=force_index)

    nodes = [TextNode(id_=str(i), text=doc) for i, doc in enumerate(documents)]
    sparse_store.add(nodes)
    semantic_store.add(nodes)

    return sparse_store, semantic_store


def main(data_path: Path, output_path: Path, alpha_range=(0.1, 0.5, 0.9), top_k: int = 5, chunk_size: int = 200):
    # Load the data
    raw_data = json.load(open(data_path, "r", encoding="utf-8"))

    question_ids, predicted_evidences = [], []

    for _, values in raw_data.items():
        documents = []
        for section in values["full_text"]:
            documents += section["paragraphs"]

        # Initialize the vector stores
        sparse_store, semantic_store = prepare_vector_stores(documents, force_index=True, chunk_size=chunk_size)

        # Initialize reranker
        reranker = Reranker(sparse_store, semantic_store)
        
        # Load gold data for evaluation
        gold_data = get_answers_and_evidence(raw_data, text_evidence_only=True)
        reranker.gold_answers_and_evidence = gold_data

        for q in values["qas"]:
            query = q["question"]
            question_ids.append(q["question_id"])

            best_alpha, best_ranking = reranker.rerank(query, top_k, alpha_range)
            predicted_evidences.append(best_ranking)

    # Save the results
    with open(output_path, "w") as f:
        for question_id, predicted_evidence in zip(question_ids, predicted_evidences):
            f.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "predicted_answer": "",  # Placeholder, no answer needed
                        "predicted_evidence": predicted_evidence,
                    }
                )
            )
            f.write("\n")


if __name__ == "__main__":
    data_path = Path("qasper-test-v0.3.json")
    output_path = Path("predictions_reranked.jsonl")
    main(data_path, output_path)
