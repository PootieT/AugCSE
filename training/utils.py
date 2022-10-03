import hashlib
import json
from typing import List, Optional

from interfaces.Operation import Operation


def normalize_dataset_names(dataset_name: str) -> str:
    if "/" in dataset_name:  # local dataset files
        dataset_name = ".".join(dataset_name.split("/")[-1].split(".")[:-1])
    return dataset_name


def capital_first_letter_acronym(s: str) -> str:
    return "".join([span[0].upper() for span in s.split("_")])


def expand_hyper_identifier_str(args_list: List, hyper_string) -> str:
    segments = hyper_string.split("-")
    result = ""
    for segment in segments:
        arg_found = False

        for args in args_list:
            if hasattr(args, segment):
                result += f"{capital_first_letter_acronym(segment)}={getattr(args, segment)}"
                arg_found = True

        if not arg_found:
            result += segment

    return result


def initialize_implementations(implementations: List[Operation], init_args_str: Optional[str]) -> List[Operation]:
    result = []
    if init_args_str is not None:
        for impl in implementations:
            try:
                impl = impl(**json.loads(init_args_str))
            except:
                impl = impl()
            result.append(impl)
    else:
        result = [impl() for impl in implementations]
    return result
