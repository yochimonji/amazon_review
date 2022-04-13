import torch
from amazon_review.util.activation import Relu, Tanh


def test_tanh_1():
    input = torch.tensor([0.8986, -0.7279, 1.1745, 0.2611, 0])
    actual = torch.tensor([0.7156, -0.6218, 0.8257, 0.2553, 0]).to(torch.float16)
    expected = Tanh().forward(input).to(torch.float16)
    assert torch.equal(actual, expected)


def test_tanh_2():
    input = torch.tensor([])
    actual = torch.tensor([])
    expected = Tanh().forward(input)
    assert torch.equal(actual, expected)


def test_tanh_3():
    input = torch.tensor([-float("inf"), float("inf")])
    actual = torch.tensor([-1.0, 1.0])
    expected = Tanh().forward(input)
    assert torch.equal(actual, expected)


def test_relu_1():
    input = torch.tensor([-0.1, 0.0, 0.1])
    actual = torch.tensor([0.0, 0.0, 0.1])
    expected = Relu().forward(input)
    assert torch.equal(actual, expected)


def test_relu_2():
    input = torch.tensor([])
    actual = torch.tensor([])
    expected = Relu().forward(input)
    assert torch.equal(actual, expected)


def test_relu_3():
    input = torch.tensor([-float("inf"), float("inf")])
    actual = torch.tensor([0, float("inf")])
    expected = Relu().forward(input)
    assert torch.equal(actual, expected)
