import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml
from transformers import HfArgumentParser

from configs.config_maps import MODEL_PROTOTYPE_CONFIGS, TRAINING_CONFIGS

# Pick defaults from available keys, for example:
DEFAULT_PROTOTYPE_CONFIG = next(iter(MODEL_PROTOTYPE_CONFIGS.keys()))
DEFAULT_TRAINING_CONFIG = next(iter(TRAINING_CONFIGS.keys()))

@dataclass
class TrainingArguments:
    job_dir: str = field(metadata={"help": "Output directory for training"})
    prototype_config_name: Optional[str] = field(
        default=DEFAULT_PROTOTYPE_CONFIG,
        metadata={"help": "Name of the model prototype config to train"},
    )
    training_config_name: Optional[str] = field(
        default=DEFAULT_TRAINING_CONFIG,
        metadata={"help": "Name of the training config to use"},
    )


def load_json(json_name: Union[str, os.PathLike]) -> Dict[str, Any]:
    with open(json_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_config_dict(**kwargs):
    config_dict = {"output_dir": kwargs.pop("job_dir"), "run_name": "pixel-local"}

    # Remove keys not used locally
    for k in ["ngpus", "nodes", "timeout", "partition", "exclude_nodes", "comment"]:
        kwargs.pop(k, None)

    # Load model config
    prototype_config_name = kwargs.pop("prototype_config_name", None)
    if prototype_config_name:
        model_config = load_json(MODEL_PROTOTYPE_CONFIGS[prototype_config_name])
        for key in model_config.keys():
            kwargs.pop(key, None)
        config_dict.update(model_config)
        config_dict["run_name"] = f"{prototype_config_name}-local"

    # Load training config
    training_config_name = kwargs.pop("training_config_name", None)
    if training_config_name:
        training_config = load_json(TRAINING_CONFIGS[training_config_name])
        for key in training_config.keys():
            kwargs.pop(key, None)
        config_dict.update(training_config)

    config_dict.update(kwargs)

    return config_dict


def process_remaining_strings(remaining_strings: Union[str, List[str]]):
    def parse_string(s: str):
        s = s.strip().replace("--", "")
        if " " in s:
            k, v = s.split(" ", 1)
        elif "=" in s:
            k, v = s.split("=", 1)
        else:
            k, v = s, "True"
        return {k: yaml.safe_load(v)}

    if isinstance(remaining_strings, str):
        return parse_string(remaining_strings)
    else:
        remaining_strings_dict = {}
        for rs in remaining_strings:
            remaining_strings_dict.update(parse_string(rs))
        return remaining_strings_dict


def main():
    parser = HfArgumentParser(TrainingArguments)
    args, remaining_strings = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    args_dict = asdict(args)

    if remaining_strings:
        args_dict.update(process_remaining_strings(remaining_strings))

    # Ensure local_rank = -1 to disable distributed training
    args_dict["local_rank"] = -1
    args_dict["train_batch_size"] = 4
    args_dict["gradient_accumulation_step"] = 8 # Total batch size = 32
    args_dict["num_train_epochs"] = 1

    # Remove env vars that trigger distributed, if any
    for var in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]:
        if var in os.environ:
            del os.environ[var]

    config_dict = get_config_dict(**args_dict)

    import scripts.training.run_pretraining as trainer

    print(f"Starting training with config:\n{json.dumps(config_dict, indent=2)}")
    trainer.main(config_dict)



if __name__ == "__main__":
    main()
