from datasets import load_dataset
import datasets
from collections import Counter
import json
from pydantic import BaseModel
from typing import List

ds = load_dataset("Salesforce/xlam-function-calling-60k")

train_ds = ds["train"]

class Answer(BaseModel):
    name: str
    arguments: dict

class Tool(BaseModel):
    name: str
    description: str

    def __hash__(self):
        return hash(self.name)

class ToolWithParameters(Tool):
    parameters: dict

class Example(BaseModel):
    id: int
    query: str
    answers: List[Answer]
    tools: List[ToolWithParameters]

class ClassificationExample(BaseModel):
    id: int
    query: str
    answer: List[str]
    tools: List[Tool]

def _parse_row(row):
    id: int = row["id"]
    query: str = row["query"]
    answers: List[Answer] = [Answer.model_validate(answer) for answer in json.loads(row["answers"])]
    tools: List[ToolWithParameters] = [ToolWithParameters.model_validate(tool) for tool in json.loads(row["tools"])]
    return Example(id=id, query=query, answers=answers, tools=tools)

rows = []
for row in train_ds:
    example = _parse_row(row)
    classification_example = ClassificationExample(
        id=example.id,
        query=example.query,
        answer=list(set([answer.name for answer in example.answers])),
        tools=list(set([Tool(**tool.model_dump()) for tool in example.tools]))
    )
    rows.append(classification_example)


new_ds = datasets.Dataset.from_list([classification_example.model_dump() for classification_example in rows])

train_testdev = new_ds.train_test_split(test_size=0.2, seed=42)

test_dev = train_testdev["test"].train_test_split(test_size=0.5, seed=42)

final_dataset = datasets.DatasetDict({
    "train": train_testdev["train"],
    "test": test_dev["test"],
    "dev": test_dev["train"]
})

final_dataset.save_to_disk("data/tool-selection/tool_selection_dataset")