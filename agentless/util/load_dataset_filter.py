import json
import os

from datasets import load_dataset as _load_dataset


def load_dataset(*args, **kwargs):
    # Check existance of target_inst_ids.json at repo root
    if not os.path.exists("target_inst_ids.json"):
        return _load_dataset(*args, **kwargs)

    with open("target_inst_ids.json") as f:
        target_inst_ids = json.load(f)["target_inst_ids"]
    return _load_dataset(*args, **kwargs).filter(
        input_columns=["instance_id"],
        function=lambda x: x in target_inst_ids,
    )
