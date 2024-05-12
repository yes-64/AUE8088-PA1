from torchmetrics import Metric
import torch
import src.config as cfg

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = cfg.NUM_CLASSES
        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds, target):

        preds = torch.argmax(preds, dim=1)
        
        assert preds.shape == target.shape, 'error msg: check if preds and target have equal shape'

        for idx in range(self.num_classes):
            self.true_positives[idx] += torch.sum((preds == idx) & (target == idx))
            self.false_positives[idx] += torch.sum((preds == idx) & (target != idx))
            self.false_negatives[idx] += torch.sum((preds != idx) & (target == idx))


    def compute(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-6)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-6)

        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        f1_score = f1_score.mean()

        return f1_score


class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape, 'error msg: check if preds and target have equal shape'

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
