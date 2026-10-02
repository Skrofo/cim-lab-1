import torch

from cim_lab_1.metrics import accuracy, prediction_consistency, representation_similarity


def test_accuracy():
    logits = torch.tensor([[3.0, 1.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1])
    assert torch.isclose(accuracy(logits, targets), torch.tensor(1.0))


def test_prediction_consistency_identical_logits():
    logits = torch.tensor([[3.0, 1.0], [0.0, 2.0]])
    assert torch.isclose(prediction_consistency(logits, logits), torch.tensor(1.0))


def test_representation_similarity_identical_features():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert torch.isclose(representation_similarity(features, features), torch.tensor(1.0))
