from typing import List
import json
import random

from tqdm import tqdm

from datasets import load_dataset
import datasets
from dotenv import load_dotenv

from src.domain import Answer, ToolWithParameters, FunctionCallingExample
from src.scenarios.clarification_scenario import ClarificationScenario
from src.scenarios.missing_tool import MissingToolScenario
from src.scenarios.single_turn import SingleTurnScenario
from src.scenarios.system_prompt import SystemPromptScenario
from src.scenarios.distractor import DistractorScenario
from src.scenarios.follow_up import FollowUpScenario
from src.scenarios.reference_resolution import ReferenceResolutionScenario

load_dotenv()

SCENARIOS = {
    "single_turn": SingleTurnScenario(),
    "system_prompt": SystemPromptScenario(),
    "clarification": ClarificationScenario(),
    "missing_tool": MissingToolScenario(),
    "distractor": DistractorScenario(),
    "follow_up": FollowUpScenario(),
    "reference_resolution": ReferenceResolutionScenario(),
}

TARGETS = {
    "single_turn": 600,
    "system_prompt": 450,
    "clarification": 600,
    "follow_up": 450,
    "reference_resolution": 450,
    "distractor": 300,
    "missing_tool": 150,
}

# TARGETS = {
#     "single_turn": 1,
#     "system_prompt": 1,
#     "clarification": 1,
#     "follow_up": 1,
#     "reference_resolution": 1,
#     "distractor": 1,
#     "missing_tool": 1,
# }

def _parse_row(row):
    id: int = row["id"]
    query: str = row["query"]
    answers: List[Answer] = [Answer.model_validate(answer) for answer in json.loads(row["answers"])]
    tools: List[ToolWithParameters] = [ToolWithParameters.model_validate(tool) for tool in json.loads(row["tools"])]
    return FunctionCallingExample(id=id, query=query, answers=answers, tools=tools)


ds = load_dataset("Salesforce/xlam-function-calling-60k")
train_ds = ds["train"]
examples = [_parse_row(row) for row in train_ds]


indices = list(
    range(len(examples))
)

random.seed(42)
random.shuffle(indices)

offset = 0

scenario_assignments = {}
for scenario, count in TARGETS.items():
    scenario_assignments[scenario] = (
        indices[offset : offset + count]
    )
    offset += count

# TODO: Add progressbar and error handling

for scenario_name, target_count in tqdm(TARGETS.items()):
    scenario = SCENARIOS[scenario_name]
    for id in tqdm(scenario_assignments[scenario_name]):
        example = examples[id]
        generated_example = scenario.generate(example)
        if len(generated_example.answer) == 0:
            continue
        
        with open(f"tmp/tool-selection.jsonl", "a") as f:
            json.dump(generated_example.model_dump(), f)
            f.write("\n")

    print(f"Generated {len(scenario_assignments[scenario_name])} examples for scenario {scenario_name}")