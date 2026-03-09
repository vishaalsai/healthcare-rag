#!/usr/bin/env python
"""
Interactive query CLI for Healthcare RAG.
Streams the answer to stdout with references.

Usage:
    python scripts/query.py "What is the treatment for hypertension?"
    python scripts/query.py --stream "What are WHO diabetes criteria?"
    python scripts/query.py --interactive   # REPL mode
"""

from __future__ import annotations

import argparse
import sys

# Windows console defaults to cp1252; force UTF-8 so Claude's medical
# symbols (≥, →, µ, etc.) print correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

load_dotenv()


def build_pipeline(settings: dict):
    """Wire up the full RAG pipeline from settings."""
    from src.ingestion.embedder import EmbeddingModel
    from src.retrieval.vector_store import ChromaVectorStore
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker
    from src.generation.llm_client import AnthropicClient
    from src.generation.answer_generator import AnswerGenerator
    from src.utils.prompt_manager import PromptManager
    from src.utils.citation_utils import CitationEnforcer

    emb_cfg = settings["embeddings"]
    chr_cfg = settings["chromadb"]
    ret_cfg = settings["retrieval"]
    gen_cfg = settings["generation"]

    embedding_model = EmbeddingModel(
        model_name=emb_cfg["model"],
        device=emb_cfg["device"],
        batch_size=emb_cfg["batch_size"],
    )
    vector_store = ChromaVectorStore(
        embedding_model=embedding_model,
        persist_directory=chr_cfg["persist_directory"],
        collection_name=chr_cfg["collection_name"],
    )
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        bm25_top_k=ret_cfg["bm25_top_k"],
        vector_top_k=ret_cfg["vector_top_k"],
        rrf_k=ret_cfg["rrf_k"],
        final_top_k=ret_cfg["final_top_k"],
    )
    reranker = CrossEncoderReranker(
        model_name=ret_cfg["reranker_model"],
        top_k=ret_cfg["reranker_top_k"],
    )
    llm = AnthropicClient(
        model=gen_cfg["model"],
        max_tokens=gen_cfg["max_tokens"],
        temperature=gen_cfg["temperature"],
    )
    prompts = PromptManager()
    enforcer = CitationEnforcer()

    return AnswerGenerator(
        llm_client=llm,
        retriever=hybrid,
        reranker=reranker,
        prompt_manager=prompts,
        citation_enforcer=enforcer,
    )


def run_query(generator, question: str, stream: bool = False) -> None:
    print(f"\nQuestion: {question}")
    print("-" * 60)

    if stream:
        accumulated = ""
        for item in generator.answer_stream(question):
            if isinstance(item, str):
                print(item, end="", flush=True)
                accumulated += item
            else:
                result = item
        print()  # newline after stream
        refs = result.citation_result.formatted_references()
        if refs:
            print("\nReferences:")
            print(refs)
    else:
        result = generator.answer(question)
        print(result.pretty_print())

    print("-" * 60)


def interactive_mode(generator) -> None:
    print("\nHealthcare RAG — Interactive Mode")
    print("Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break
        run_query(generator, question)


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcare RAG query CLI")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive REPL mode"
    )
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    with open(config_path) as f:
        settings = yaml.safe_load(f)

    logger.remove()  # suppress debug logs in CLI
    logger.add(sys.stderr, level="WARNING")

    print("Initialising pipeline …", end="", flush=True)
    generator = build_pipeline(settings)
    print(" done.")

    if args.interactive:
        interactive_mode(generator)
    elif args.question:
        run_query(generator, args.question, stream=args.stream)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
