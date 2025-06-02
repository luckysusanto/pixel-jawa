import argparse
import logging
from typing import Any, Dict

import datasets

logger = logging.getLogger(__name__)


def log_example_while_rendering(idx: int, line: str, num_patches: int):
    """
    Logs a line of text at index idx with its corresponding number of patches to see what we are rendering
    """

    logger.info(f"Index = {idx}")
    logger.info(f"Line = {line}")
    logger.info(f"Line length = {len(line)}")
    logger.info(f"Num patches = {num_patches}\n")


def push_rendered_chunk_to_hub(
    args: argparse.Namespace, chunk: Dict[str, Any], dataset_stats: Dict[str, int], current_num_examples: int
) -> Dict[str, int]:
    r"""
    Pushes an arbitrarily sized chunk of rendered texts to a specified HuggingFace data repository

     Args:
            args (`argparse.Namespace`):
                Arguments containing the fields repo_id (the repository we push to), split (the dataset split we push
                 to), and auth_token (in case we push to a private repo)
            chunk (`Dict[str, Any]`):
                The chunk of data that we construct a `Dataset` from
            dataset_stats (`Dict[str, int]`):
                A dictionary containing meta-information on the dataset which are continuously updated everytime
                we push a new chunk
            current_num_examples (`int`):
                The number of examples that have been processed so far. Will also be stored in the dataset_infos file

    Returns:
        A dataset stats dictionary of type [`~Dict[str, int]`].

    """
    logger.info(f"Pushing batch {current_num_examples // args.chunk_size} to HuggingFace")
    logger.info(f"Current uploaded size = {dataset_stats['total_uploaded_size']}")
    logger.info(f"Current dataset nbytes = {dataset_stats['total_dataset_nbytes']}")
    logger.info(f"Current shards = {dataset_stats['total_num_shards']}")
    logger.info(f"Current num examples = {current_num_examples}")
    logger.info(f"Current num words = {dataset_stats['total_num_words']}\n")

    chunk_dataset = datasets.Dataset.from_dict(chunk)
    new_uploaded_size, new_dataset_nbytes, new_num_shards = chunk_dataset.push_to_hub(
        repo_id=args.repo_id,
        split=args.split,
        private=True,
        token=args.auth_token,
        branch="main",
        embed_external_files=True,
        existing_uploaded_size=dataset_stats["total_uploaded_size"],
        existing_nbytes=dataset_stats["total_dataset_nbytes"],
        existing_nshards=dataset_stats["total_num_shards"],
        num_examples=current_num_examples,
    )

    dataset_stats.update(
        {
            "total_uploaded_size": new_uploaded_size,
            "total_dataset_nbytes": new_dataset_nbytes,
            "total_num_shards": dataset_stats["total_num_shards"] + new_num_shards,
            "total_num_examples": current_num_examples,
        }
    )
    return dataset_stats

from typing import Dict, Any
import logging
import datasets
import argparse

logger = logging.getLogger(__name__)

def push_rendered_chunk_as_split_to_hub(
    args: argparse.Namespace,
    chunk: dict,
    dataset_stats: dict,
    current_num_examples: int,
) -> dict:
    r"""
    Pushes a chunk of rendered texts to a separate dataset split on HuggingFace Hub.

    Args:
        args (`argparse.Namespace`): Must contain repo_id, auth_token, and split_prefix (string prefix for split names)
        chunk (`dict`): Chunk of data to push as a dataset
        dataset_stats (`dict`): Dictionary for metadata (you can update it as needed)
        current_num_examples (`int`): Number of examples processed so far (used for naming split)

    Returns:
        Updated dataset_stats dictionary
    """
    split_name = f"chunk_{current_num_examples // args.chunk_size}"

    import datasets
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Pushing batch {current_num_examples // args.chunk_size} as split '{split_name}' to HuggingFace")

    chunk_dataset = datasets.Dataset.from_dict(chunk)

    # Push without unsupported kwargs
    chunk_dataset.push_to_hub(
        repo_id=args.repo_id,
        config_name=split_name,
        private=False,
        token=args.auth_token,
        embed_external_files=True,
    )

    # Optionally update dataset_stats here, but without new_uploaded_size etc.
    dataset_stats["total_num_examples"] = current_num_examples
    dataset_stats["total_num_splits"] = dataset_stats.get("total_num_splits", 0) + 1

    logger.info(f"Finished pushing split '{split_name}'. Total splits pushed: {dataset_stats['total_num_splits']}")

    return dataset_stats

