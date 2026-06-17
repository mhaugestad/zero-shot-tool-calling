from datasets import Dataset, DatasetDict
import json


JSONL_PATH = "tmp/tool-selection.jsonl"
REPO_ID = "mhaugestad/tool-selection"


def load_jsonl(path: str) -> list[dict]:

    rows = []

    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def main():

    rows = load_jsonl(
        JSONL_PATH
    )

    dataset = Dataset.from_list(
        rows
    )

    train_test = dataset.train_test_split(
        test_size=0.2,
        seed=42,
    )

    test_dev = train_test[
        "test"
    ].train_test_split(
        test_size=0.5,
        seed=42,
    )

    dataset_dict = DatasetDict(
        {
            "train": train_test["train"],
            "dev": test_dev["train"],
            "test": test_dev["test"],
        }
    )

    print(dataset_dict)

    dataset_dict.push_to_hub(
        REPO_ID
    )

if __name__ == "__main__":
    main()