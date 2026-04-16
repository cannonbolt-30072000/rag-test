"""
Hosted Gradio UI - no ingestion, pre-computed data only.
Used in Docker / Azure deployment.
"""

import gradio as gr
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from generation.chain import query


def ask(question: str, history: list) -> tuple[str, str]:
    if not question.strip():
        return "Please ask a question.", ""

    result = query(question, top_k=5, rerank=True)

    sources_text = ""
    for i, s in enumerate(result["sources"], start=1):
        sources_text += (
            f"**[{i}] {s['source']} — Page {s['page']}**\n"
            f"Score: {s['score']:.4f}\n"
            f"{s['preview']}...\n\n"
        )

    return result["answer"], sources_text


def load_eval_results():
    eval_path = Path("eval/results.json")
    if not eval_path.exists():
        return "No evaluation results found.", []

    data = json.loads(eval_path.read_text())

    summary_md = "### Overall Scores\n\n"
    summary_md += "| Metric | Score | Min | Max | Status |\n"
    summary_md += "|--------|-------|-----|-----|--------|\n"
    for metric, vals in data["summary"].items():
        status = "PASS" if vals["pass"] else "FAIL"
        summary_md += f"| {metric} | **{vals['average']:.2f}** | {vals['min']:.2f} | {vals['max']:.2f} | {status} |\n"

    summary_md += f"\n*Evaluated on {data['total_questions']} golden questions*"

    details = [
        [d["question"][:60], d["faithfulness"], d["relevancy"], d["precision"], d["recall"]]
        for d in data["details"]
    ]

    return summary_md, details


with gr.Blocks(title="Pharma RAG Q&A") as demo:

    gr.Markdown("## Pharma RAG Q&A")
    gr.Markdown("Domain-specific Q&A over Bayer crop science product labels. Powered by hybrid RAG (dense + sparse + reranking).")

    with gr.Tabs():

        # Tab 1: Chat only - no upload
        with gr.TabItem("Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Chat", height=400)
                    msg = gr.Textbox(
                        label="Your question",
                        placeholder="e.g. What active ingredient is in Sivanto Prime?",
                    )
                    with gr.Row():
                        send_btn = gr.Button("Ask", variant="primary")
                        clear_btn = gr.ClearButton([msg, chatbot], value="Clear")

                with gr.Column(scale=1):
                    sources_panel = gr.Markdown(
                        value="*Ask a question to see retrieved sources.*",
                    )

            def respond(question, chat_history):
                answer, sources = ask(question, chat_history)
                chat_history.append({"role": "user", "content": question})
                chat_history.append({"role": "assistant", "content": answer})
                return "", chat_history, sources

            send_btn.click(
                fn=respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, sources_panel],
            )
            msg.submit(
                fn=respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, sources_panel],
            )

        # Tab 2: Evaluation dashboard
        with gr.TabItem("Evaluation"):
            summary_md, details = load_eval_results()
            gr.Markdown(summary_md)

            if details:
                gr.Dataframe(
                    value=details,
                    headers=["Question", "Faithfulness", "Relevancy", "Precision", "Recall"],
                    label="Per-Question Scores",
                )

        # Tab 3: Architecture - show what's under the hood
        with gr.TabItem("Architecture"):
            gr.Markdown("""
### System Architecture

**Ingestion pipeline**
- PDF loading → recursive chunking (512 chars, 50 overlap) → dual embedding (dense + sparse)
- Dense: `all-MiniLM-L6-v2` (384-dim semantic vectors)
- Sparse: `Qdrant/BM25` (keyword term vectors)
- Storage: Qdrant with named vector collections

**Retrieval pipeline**
- Hybrid search: dense similarity + sparse BM25 via Qdrant prefetch
- Fusion: Reciprocal Rank Fusion (RRF) — no manual score normalization
- Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2` on top-20 candidates → top-5

**Generation**
- LLM: GPT-4o-mini (temperature=0 for deterministic output)
- Prompt: context-only grounding with citation instructions
- Output: answer + source citations + latency tracking

**Evaluation**
- Framework: DeepEval + RAGAS metrics
- Metrics: Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall
- Judge model: Gemini 2.5 Flash Lite
- Threshold: 0.7 across all metrics

**Stack**: FastAPI · Gradio · Qdrant · LangChain · sentence-transformers · DeepEval
""")

# Mount FastAPI routes on Gradio's internal FastAPI app
from fastapi import HTTPException
import time
from DTO.request_response_models import QueryRequest, QueryResponse, Source

@demo.app.get("/health")
def health():
    return {"status": "ok"}

@demo.app.get("/eval")
def get_eval():
    eval_path = Path("eval/results.json")
    if not eval_path.exists():
        raise HTTPException(status_code=404, detail="No eval results")
    return json.loads(eval_path.read_text())

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)