"""
RAG chain - the orchestrator.
Takes a user query, retrieves context, generates a cited answer.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from retrieval.search import search


SYSTEM_PROMPT = """You are a helpful AI assistant for Bayer life sciences.
Answer the user's question using ONLY the provided context.

Rules:
- If the context doesn't contain enough information, say "I don't have enough information to answer this."
- Cite your sources using [Page X] notation after each claim.
- Give a thorough answer - cover all relevant details found in the context.
- Structure longer answers with clear sections if multiple aspects are covered.
- Never make up information not present in the context.

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


#LLM
# llm = ChatOpenAI(
#     model="gpt-4o-mini",      
#     temperature=0.0,         
# )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
parser = StrOutputParser() 


def build_context(results: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt.
    Each chunk is labeled with its source and page so the LLM can cite them.
    """

    context_parts = []

    for i, result in enumerate(results, start=1):
        source = result["metadata"]["source"]
        page = result["metadata"]["page"]
        text = result["text"]

        context_parts.append(
            f"[Source {i}: {source}, Page {page}]\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)


def query(question: str, top_k: int = 5, rerank: bool = True) -> dict:
    """
    Args:
        question: User's natural language question.
        top_k: Number of chunks to retrieve.
        rerank: Whether to apply cross-encoder reranking.

    Returns:
        Dict with 'answer', 'sources', and 'context_used'.
    """

    results = search(question, top_k=top_k, rerank=rerank)

    if not results:
        return {
            "answer": "I couldn't find any relevant information in the documents.",
            "sources": [],
            "context_used": "",
        }

    context = build_context(results)

    chain = prompt | llm | parser

    answer = chain.invoke({
        "context": context,
        "question": question,
    })
    sources = [
        {
            "source": r["metadata"]["source"],
            "page": r["metadata"]["page"],
            "score": r.get("rerank_score", r["score"]),
            "preview": r["text"][:100],
        }
        for r in results
    ]

    return {
        "answer": answer,
        "sources": sources,
        "context_used": context,
    }
