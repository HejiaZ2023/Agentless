from datasets import load_dataset as _load_dataset
import re

filter_instance = "^(astropy__astropy-12907)$"

def load_dataset(*args, **kwargs):
    return _load_dataset(*args, **kwargs).filter(
        input_columns=["instance_id"],
        function=lambda x: bool(re.match(filter_instance, x)),
    )