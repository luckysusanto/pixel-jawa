"""
Script used to prerender a Wikipedia dump that has been downloaded to disk.
Processes the dataset line-by-line, extracting documents (articles) and uploads the rendered examples in chunks
to HuggingFace. Tries to filter out title lines as these are typically short and provide little value.
Examples are stored and compressed in parquet files.
Relies on a modified version of the datasets library installed through git submodule.
"""

import argparse
import logging
import sys

from PIL import Image
from pixel import PyGameTextRenderer, log_example_while_rendering, push_rendered_chunk_as_split_to_hub

logger = logging.getLogger(__name__)

def process_doc(
    args: argparse.Namespace,
    text_renderer: PyGameTextRenderer,
    idx: int,
    data: dict,
    dataset_stats: dict,
    doc: str,
    target_seq_length: int,
):
    doc = doc.strip()
    dataset_stats["total_num_words"] += len(doc.split(" "))

    width = text_renderer.font.get_rect(doc).width

    if width >= target_seq_length:
        # If one doc is too long, just break in half (rare case safeguard)
        midpoint = len(doc) // 2
        split_point = doc.rfind(" ", 0, midpoint)
        if split_point == -1:
            split_point = midpoint  # fallback if no space found

        chunks = [doc[:split_point].strip(), doc[split_point:].strip()]
    else:
        chunks = [doc]

    for chunk in chunks:
        idx += 1
        encoding = text_renderer(text=chunk)

        data["pixel_values"].append(Image.fromarray(encoding.pixel_values))
        data["num_patches"].append(encoding.num_text_patches)

        if idx % args.chunk_size == 0:
            log_example_while_rendering(idx, chunk, encoding.num_text_patches)
            dataset_stats = push_rendered_chunk_as_split_to_hub(args, data, dataset_stats, idx)
            data = {"pixel_values": [], "num_patches": []}

    return idx, data, dataset_stats


def main(args: argparse.Namespace):
    # Load PyGame renderer
    text_renderer = PyGameTextRenderer.from_pretrained(args.renderer_name_or_path, use_auth_token=args.auth_token)

    data = {"pixel_values": [], "num_patches": []}
    dataset_stats = {
        "total_uploaded_size": 0,
        "total_dataset_nbytes": 0,
        "total_num_shards": 0,
        "total_num_examples": 0,
        "total_num_words": 0,
    }

    max_pixels = text_renderer.pixels_per_patch * text_renderer.max_seq_length - 2 * text_renderer.pixels_per_patch
    target_seq_length = max_pixels

    import pandas as pd
    import ast

    ## Change the logics here:
    ## If word level
    # df = pd.read_csv(args.data_path)
    # df["article_script_list"] = df["article_script_list"].apply(ast.literal_eval) 

    # idx = 0
    # for doc_id, row in df.iterrows():
    #     title = f"article_{row['article_id']}"
    #     doc = " ".join(row["article_script_list"])  # reconstruct article text from word list

    #     num_examples = idx
    #     num_words = dataset_stats["total_num_words"]

    #     logger.info(f"{doc_id}: {title}, {target_seq_length=}px, {num_examples=}, {num_words=}")

    #     idx, data, dataset_stats = process_doc(
    #         args=args,
    #         text_renderer=text_renderer,
    #         idx=idx,
    #         data=data,
    #         dataset_stats=dataset_stats,
    #         doc=doc,
    #         target_seq_length=target_seq_length,
    #     )

    ## If article level
    df = pd.read_csv(args.data_path)
    df['id'] = range(len(df))
    idx = 0
    for _, row in df.iterrows():
        title = f"article_{row['id']}"
        doc = row['transcription']

        num_examples = idx 
        num_words = dataset_stats["total_num_words"]

        logger.info(f"{_}: {title}, {target_seq_length=}px, {num_examples=}, {num_words=}")

        idx, data, dataset_stats = process_doc(
            args=args,
            text_renderer=text_renderer,
            idx=idx,
            data=data,
            dataset_stats=dataset_stats,
            doc=doc,
            target_seq_length=target_seq_length,
        )


    # Push final chunk to hub
    push_rendered_chunk_as_split_to_hub(args, data, dataset_stats, idx)

    logger.info(f"Total num words in wikipedia: {dataset_stats['total_num_words']}")


if __name__ == "__main__":
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--renderer_name_or_path", # configs/renderers/java_renderer
        type=str,
        help="Path or Huggingface identifier of the text renderer",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Path to a dataset on disk") # "./article_merged_chunk_000.csv"
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000, # NO CHANGE
        help="Push data to hub in chunks of N lines",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=-1, # NO CHANGE
        help="Only look at the first N non-empty lines",
    )
    parser.add_argument("--repo_id", type=str, help="Name of dataset to upload") # https://huggingface.co/datasets/Exqrch/prerender-java-test
    parser.add_argument("--split", type=str, help="Name of dataset split to upload") # train
    parser.add_argument(
        "--auth_token",
        type=str,
        help="Huggingface auth token with write access to the repo id",
    )

    parsed_args = parser.parse_args()

    main(parsed_args)
