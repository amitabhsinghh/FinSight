import os
import json
import torch
import argparse
import logging
from sentence_transformers import CrossEncoder

from financerag.common import post_process, save_results_top_k
from financerag.rerank import CrossEncoderReranker
from financerag.tasks import (
    FinDER, FinQABench, ConvFinQA, FinanceBench, MultiHiertt, TATQA, FinQA
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Rerank retrieved documents using a CrossEncoder")

    parser.add_argument("--task", type=str, required=True, choices=[
        "FinDER", "FinQABench", "ConvFinQA", "FinanceBench", "MultiHiertt", "TATQA", "FinQA"
    ], help="The dataset/task to rerank")

    parser.add_argument("--model", type=str, required=True, choices=[
        "jinaai/jina-reranker-v2-base-multilingual",
        "Alibaba-NLP/gte-multilingual-reranker-base",
        "BAAI/bge-reranker-v2-m3"
    ], help="Pretrained CrossEncoder model")

    parser.add_argument("--top_k", type=int, default=200, help="Top-K results to rerank")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for reranking")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="Directory of dataset files")
    parser.add_argument("--dataset_filename", type=str, default="merge.json", help="Filename of rerank input")
    parser.add_argument("--save_dir", type=str, default="./results/rerank", help="Directory to save output")
    parser.add_argument("--save_top_k", type=int, help="Save only top-K results (optional)")
    parser.add_argument("--do_post_process", action="store_true", help="Apply post-processing to outputs")

    return parser.parse_args()


def rerank():
    args = parse_args()

    if args.do_post_process:
        logger.info("Post-processing results...")
        post_process(args.save_dir, args.dataset_dir)
        return

    task_classes = {
        "FinDER": FinDER, "FinQABench": FinQABench, "ConvFinQA": ConvFinQA,
        "FinanceBench": FinanceBench, "MultiHiertt": MultiHiertt,
        "TATQA": TATQA, "FinQA": FinQA
    }

    task = task_classes[args.task]()

    config_args = {"torch_dtype": torch.float16 if args.model == "Alibaba-NLP/gte-multilingual-reranker-base"
                   else torch.bfloat16, "attn_implementation": "sdpa" if "gte" in args.model else "eager"}

    reranker_model = CrossEncoderReranker(CrossEncoder(
        args.model, trust_remote_code=True, **config_args
    ))

    dataset_path = os.path.join(args.dataset_dir, args.task, args.dataset_filename)
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at: {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        data = json.load(f)

    try:
        logger.info(f"Starting reranking for task: {args.task}")
        results = task.rerank(
            reranker=reranker_model,
            results=data,
            top_k=args.top_k,
            batch_size=args.batch_size
        )

        task.save_results(output_dir=args.save_dir)
        logger.info(f"Reranking complete. Results saved to: {args.save_dir}")

        if args.save_top_k:
            save_path = os.path.join(args.save_dir, args.task)
            os.makedirs(save_path, exist_ok=True)
            save_results_top_k(results, args.save_top_k, save_path)
            logger.info(f"Top-{args.save_top_k} results saved to: {save_path}")

    except Exception as e:
        logger.error(f"Error during reranking for task {args.task}: {str(e)}")


if __name__ == "__main__":
    rerank()
