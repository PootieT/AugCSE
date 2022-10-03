from typing import List, Dict, Tuple, Optional

from tqdm import tqdm
import pandas as pd

from TestRunner import OperationRuns
from interfaces.SentenceOperation import SentenceOperation
from tasks.TaskTypes import TaskType


def collect_transformations(query_task_types=List[TaskType]):
    df = []
    for transformation in OperationRuns.get_all_operations("transformations"):
        if any([t in transformation.tasks for t in query_task_types]) and issubclass(transformation, SentenceOperation):
            feat = {k: True for k in transformation.keywords} if transformation.keywords else {}
            feat["languages"] = transformation.languages
            feat["max_outputs"] = transformation.max_outputs
            feat["heavy"] = transformation.heavy
            feat["name"] = transformation.name()
            print(transformation.name())
            df.append(feat)

    df = pd.DataFrame(df).fillna(False)
    br = "\n"
    print(f"All highly meaning preserving augmentations: {br.join(list(df[df['highly-meaning-preserving']]['name']))}")
    pass


if __name__ == "__main__":
    collect_transformations(query_task_types=[
        TaskType.TEXT_CLASSIFICATION,
        TaskType.PARAPHRASE_DETECTION,
        TaskType.QUALITY_ESTIMATION,
        TaskType.SENTIMENT_ANALYSIS
    ])