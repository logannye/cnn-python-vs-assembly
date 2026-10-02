import torch
from torch import nn

from classifier.model import SimpleCNN


def test_model_outputs_logits_and_backpropagates():
    torch.manual_seed(42)
    model = SimpleCNN(num_classes=3)
    images = torch.randn(4, 3, 16, 20)
    labels = torch.tensor([0, 1, 2, 0])

    logits = model(images)
    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()

    nn.CrossEntropyLoss()(logits, labels).backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
