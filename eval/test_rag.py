"""
RAG evaluation using DeepEval.
Run: python eval/test_rag.py
"""

from golden_dataset import golden_dataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from deepeval.models import GeminiModel
from dotenv import load_dotenv
load_dotenv()

model = GeminiModel(
    model="gemini-2.5-flash-lite"
)

faithfulness = FaithfulnessMetric(threshold=0.7, model=model)
relevancy = AnswerRelevancyMetric(threshold=0.7, model=model)
precision = ContextualPrecisionMetric(threshold=0.7, model=model)
recall = ContextualRecallMetric(threshold=0.7, model=model)


from generation.chain import query

def evaluate():
    """Run questions through live pipeline, score with all 4 metrics."""

    scores = {"faithfulness": [], "relevancy": [], "precision": [], "recall": []}
    details = []

    for idx, item in enumerate(golden_dataset, start=1):
        print(f"\n[{idx}/{len(golden_dataset)}] {item['input']}")

        result = query(item["input"], top_k=5, rerank=True)

        case = LLMTestCase(
            input=item["input"],
            actual_output=result["answer"],
            expected_output=item["expected_output"],
            retrieval_context=[s["preview"] for s in result["sources"]],
        )

        # Score each metric individually - don't let one failure kill everything
        metric_scores = {}
        for name, metric in [
            ("faithfulness", faithfulness),
            ("relevancy", relevancy),
            ("precision", precision),
            ("recall", recall),
        ]:
            try:
                metric.measure(case)
                metric_scores[name] = metric.score
            except Exception as e:
                print(f"   {name} FAILED: {str(e)[:80]}")
                metric_scores[name] = None  # skip this metric for this question

        # Only record non-None scores
        for name, score in metric_scores.items():
            if score is not None:
                scores[name].append(score)

        details.append({
            "question": item["input"],
            "answer": result["answer"][:200],
            "faithfulness": round(metric_scores["faithfulness"], 2) if metric_scores["faithfulness"] is not None else "FAILED",
            "relevancy": round(metric_scores["relevancy"], 2) if metric_scores["relevancy"] is not None else "FAILED",
            "precision": round(metric_scores["precision"], 2) if metric_scores["precision"] is not None else "FAILED",
            "recall": round(metric_scores["recall"], 2) if metric_scores["recall"] is not None else "FAILED",
        })

        print(f"   F:{metric_scores['faithfulness']}  R:{metric_scores['relevancy']}  P:{metric_scores['precision']}  C:{metric_scores['recall']}")

        # Save after EVERY question so you never lose progress
        _save_results(scores, details)

    # Final summary
    print("\n" + "=" * 50)
    print("OVERALL SCORES")
    print("=" * 50)
    for metric, values in scores.items():
        if values:
            avg = sum(values) / len(values)
            status = "PASS" if avg >= 0.7 else "FAIL"
            print(f"  {metric:20s}: {avg:.2f}  [{status}]  ({len(values)} measured)")
        else:
            print(f"  {metric:20s}: NO DATA")


def _save_results(scores, details):
    """Save partial or final results to JSON."""

    import json
    from pathlib import Path

    summary = {}
    for metric, values in scores.items():
        if values:
            avg = sum(values) / len(values)
            summary[metric] = {
                "average": round(avg, 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "pass": avg >= 0.7,
                "measured": len(values),
            }

    output = {
        "summary": summary,
        "details": details,
        "total_questions": len(details),
    }

    Path("eval").mkdir(exist_ok=True)
    Path("eval/results.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    evaluate()