from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class TaskType(Enum):
    CLEAN = "CleanData"
    CORRELATION = "FindCorrelation"
    INTERPRET = "Interpret"

@dataclass
class Task:
    type: TaskType
    raw_message: str
    variables: Optional[List[str]] = None