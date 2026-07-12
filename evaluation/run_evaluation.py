"""Run the agent over the evaluation dataset and score it with RAGAS.

Usage:  python evaluation/run_evaluation.py

Produces evaluation/results.csv (per-question scores) and prints a summary.
On CPU with a 3B judge model the absolute scores are noisy — what matters is
tracking them consistently across changes. Set EVAL_MODEL to a larger model
for more reliable judgments if you have the hardware.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, evaluate
from ragas.run_config import RunConfig
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from agentic_rag.config import settings
from agentic_rag.graph.build import build_graph
from agentic_rag.ingestion import ensure_index, get_embeddings

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.csv"


def collect_samples(graph, cases: list[dict]) -> list[SingleTurnSample]:
    """Run every eval question through the agent and record its traces."""
    samples = []
    for i, case in enumerate(cases, start=1):
        start = time.perf_counter()
        result = graph.invoke({"question": case["question"]})
        elapsed = time.perf_counter() - start
        print(f"[{i}/{len(cases)}] ({elapsed:.1f}s) {case['question']}")
        samples.append(
            SingleTurnSample(
                user_input=case["question"],
                response=result["generation"],
                retrieved_contexts=[
                    doc.page_content for doc in result.get("documents", [])
                ],
                reference=case["ground_truth"],
            )
        )
    return samples


def main() -> None:
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    print("== Running the agent over the evaluation dataset ==")
    ensure_index()
    samples = collect_samples(build_graph(), cases)

    print("\n== Scoring with RAGAS ==")
    judge = LangchainLLMWrapper(
        ChatOllama(
            model=settings.eval_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge,
        embeddings=embeddings,
        # CPU inference serialises concurrent requests: parallel judge calls
        # just queue up and hit the default timeout, leaving NaN scores.
        run_config=RunConfig(timeout=600, max_workers=1),
    )

    df = result.to_pandas()
    df.to_csv(RESULTS_PATH, index=False)

    print("\n== Average scores ==")
    metric_columns = df.select_dtypes("number").columns
    for metric in metric_columns:
        print(f"  {metric:35s} {df[metric].mean():.3f}")
    print(f"\nPer-question results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
