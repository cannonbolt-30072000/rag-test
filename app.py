"""
Gradio chat UI for the RAG pipeline.
Wraps chain.py in a simple chat interface with a sources panel.
"""

import gradio as gr

from dotenv import load_dotenv
load_dotenv()
import json
from pathlib import Path
from ingestion.loader import load_pdf
from ingestion.chunker import create_chunks
from ingestion.embedder import create_collection, embed_and_store
from generation.chain import query


def ingest(file_path: str) -> str:
    """Ingest a PDF and return status message."""

    docs = load_pdf(file_path)
    chunks = create_chunks(docs)
    create_collection()
    count = embed_and_store(chunks)
    return f"Ingested {len(docs)} pages → {count} chunks"


def ask(question: str, history: list) -> tuple[str, str]:
    """
    Handle a user question. Returns (answer, sources_panel).

    Args:
        question: User's message from the chat input.
        history: Gradio's conversation history (unused but required by ChatInterface).

    Returns:
        Tuple of (answer text, formatted sources string).
    """

    if not question.strip():
        return "Please ask a question.", ""

    result = query(question, top_k=5, rerank=True)

    # Format sources for the side panel
    sources_text = ""
    for i, s in enumerate(result["sources"], start=1):
        sources_text += (
            f"**[{i}] {s['source']} — Page {s['page']}**\n"
            f"Score: {s['score']:.4f}\n"
            f"{s['preview']}...\n\n"
        )

    return result["answer"], sources_text

def load_eval_results():
    """Load pre-computed eval results for the dashboard."""

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
# --- Build the UI ---

with gr.Blocks(title="Pharma RAG Q&A") as demo:

    gr.Markdown("## Pharma RAG Q&A")
    gr.Markdown("Ask questions about ingested pharma/agriculture documents.")

    with gr.Tabs():

        with gr.TabItem("Chat"):
            # Row 1: File upload + ingest button
            with gr.Row():
                file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                ingest_btn = gr.Button("Ingest Document", variant="primary")
                ingest_status = gr.Textbox(label="Status", interactive=False)

            ingest_btn.click(
                fn=lambda f: ingest(f.name) if f else "No file selected",
                inputs=file_input,
                outputs=ingest_status,
            )

            # Row 2: Chat + sources side by side
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Chat", height=400)
                    msg = gr.Textbox(
                        label="Your question",
                        placeholder="e.g. What is myGenAssist?",
                    )
                    send_btn = gr.Button("Ask", variant="primary")
                    clear_btn = gr.ClearButton([msg, chatbot], value="Clear chat")

                with gr.Column(scale=1):
                    sources_panel = gr.Markdown(
                        label="Sources",
                        value="*Sources will appear here after you ask a question.*",
                    )

            # Wire the chat interaction
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

        with gr.TabItem("Evaluation"):
            summary_md, details = load_eval_results()
            gr.Markdown(summary_md)

            if details:
                gr.Dataframe(
                    value=details,
                    headers=["Question", "Faithfulness", "Relevancy", "Precision", "Recall"],
                    label="Per-Question Scores",
                )

# --- Run directly ---

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)