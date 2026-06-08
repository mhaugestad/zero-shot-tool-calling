import torch
from torchmetrics import Metric


class ExactMatch(Metric):

    def __init__(self):
        super().__init__()

        self.add_state(
            "correct",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "total",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
    ):
        print("logits", logits.shape)
        print("labels", labels.shape)
        print("label_mask", label_mask.shape)

        predictions = (
            torch.sigmoid(logits) >= 0.5
        )

        batch_size = labels.shape[0]

        for i in range(batch_size):

            # Restrict evaluation to non-padded tool labels.
            num_tools = int(
                label_mask[i].sum()
            )
            pred = predictions[i][:num_tools]
            gold = labels[i][:num_tools]

            if torch.equal(pred, gold):
                self.correct += 1

            self.total += 1

    def compute(self):

        return self.correct.float() / self.total