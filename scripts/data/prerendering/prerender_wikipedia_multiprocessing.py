import argparse
import logging
import sys
from PIL import Image
import pandas as pd
import ast
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

from pixel import PyGameTextRenderer, log_example_while_rendering, push_rendered_chunk_as_split_to_hub

logger = logging.getLogger(__name__)

def process_single_doc(doc, target_seq_length, renderer_path, auth_token):
    # Init renderer in subprocess
    text_renderer = PyGameTextRenderer.from_pretrained(renderer_path, use_auth_token=auth_token)

    doc = doc.strip()
    chunks = []

    width = text_renderer.font.get_rect(doc).width
    if width >= target_seq_length:
        midpoint = len(doc) // 2
        split_point = doc.rfind(" ", 0, midpoint)
        if split_point == -1:
            split_point = midpoint
        parts = [doc[:split_point].strip(), doc[split_point:].strip()]
    else:
        parts = [doc]

    for chunk in parts:
        encoding = text_renderer(text=chunk)
        chunks.append((
            Image.fromarray(encoding.pixel_values),
            encoding.num_text_patches,
            chunk
        ))
    return chunks

def main(args: argparse.Namespace):
    # Load renderer to compute target size
    tmp_renderer = PyGameTextRenderer.from_pretrained(args.renderer_name_or_path, use_auth_token=args.auth_token)
    target_seq_length = tmp_renderer.pixels_per_patch * tmp_renderer.max_seq_length - 2 * tmp_renderer.pixels_per_patch
    del tmp_renderer

    df = pd.read_csv(args.data_path)
    df["article_script_list"] = df["article_script_list"].apply(ast.literal_eval)

    docs = [" ".join(row["article_script_list"]) for _, row in df.iterrows()]

    dataset_stats = {
        "total_uploaded_size": 0,
        "total_dataset_nbytes": 0,
        "total_num_shards": 0,
        "total_num_examples": 0,
        "total_num_words": 0,
    }

    data = {"pixel_values": [], "num_patches": []}
    idx = 0

    logger.info(f"Rendering with {args.num_workers} workers...")
    with mp.Pool(args.num_workers) as pool:
        process_fn = partial(
            process_single_doc,
            target_seq_length=target_seq_length,
            renderer_path=args.renderer_name_or_path,
            auth_token=args.auth_token,
        )
        with tqdm(total=len(docs), desc="Rendering Docs", ncols=100) as pbar:
            for result_chunks in pool.imap(process_fn, docs, chunksize=10):
                for img, patch_count, chunk_text in result_chunks:
                    idx += 1
                    data["pixel_values"].append(img)
                    data["num_patches"].append(patch_count)
                    dataset_stats["total_num_words"] += len(chunk_text.split(" "))

                    if idx % args.chunk_size == 0:
                        log_example_while_rendering(idx, chunk_text, patch_count)
                        dataset_stats = push_rendered_chunk_as_split_to_hub(args, data, dataset_stats, idx)
                        data = {"pixel_values": [], "num_patches": []}
                pbar.update(1)

    # Final push
    if data["pixel_values"]:
        push_rendered_chunk_as_split_to_hub(args, data, dataset_stats, idx)

    logger.info(f"Done. Total words: {dataset_stats['total_num_words']}, Total examples: {idx}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=150000) # Doing this because there's error with multi-split, so, yeah.
    parser.add_argument("--max_lines", type=int, default=-1)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--auth_token", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=12, help="Number of CPU workers")

    parsed_args = parser.parse_args()
    main(parsed_args)
