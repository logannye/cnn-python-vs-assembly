"""Independent PyTorch parity gate for handwritten ARM64 numerical functions."""

import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from prepare import digest, flat_parameters

from classifier.model import SimpleCNN


class Assembly:
    def __init__(self, library):
        self.lib = C.CDLL(str(Path(library).resolve()))
        signatures = {
            "cnn_workspace_bytes": ([], C.c_size_t),
            "cnn_forward": ([C.c_void_p] * 4, None),
            "cnn_backward": ([C.c_void_p] * 5, None),
            "cnn_loss": ([C.c_void_p, C.c_uint32, C.c_void_p, C.c_void_p], C.c_float),
            "cnn_adam": ([C.c_void_p] * 4 + [C.c_uint32], None),
            "cnn_conv_forward": ([C.c_void_p] * 5, None),
            "cnn_conv_backward": ([C.c_void_p] * 7, None),
            "cnn_exp": ([C.c_float], C.c_float),
            "cnn_log": ([C.c_float], C.c_float),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.workspace = C.create_string_buffer(self.lib.cnn_workspace_bytes())

    @staticmethod
    def ptr(array):
        assert array.flags.c_contiguous
        return array.ctypes.data_as(C.c_void_p)

    def forward(self, weights, inputs):
        output = np.zeros(10, dtype=np.float32)
        self.lib.cnn_forward(self.ptr(weights), self.ptr(inputs), self.workspace, self.ptr(output))
        return output

    def backward(self, weights, inputs, grad_logits, grads):
        self.lib.cnn_backward(
            self.ptr(weights),
            self.ptr(inputs),
            self.workspace,
            self.ptr(grad_logits),
            self.ptr(grads),
        )

    def loss(self, logits, label):
        probs, grads = np.zeros(10, np.float32), np.zeros(10, np.float32)
        value = self.lib.cnn_loss(self.ptr(logits), label, self.ptr(probs), self.ptr(grads))
        return value, probs, grads

    def adam(self, weights, grads, m, v, step):
        self.lib.cnn_adam(self.ptr(weights), self.ptr(grads), self.ptr(m), self.ptr(v), step)


def verify(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    asm = Assembly(args.library)
    rng = np.random.default_rng(20261002)
    checks = []
    assert np.isnan(asm.lib.cnn_exp(float("nan")))
    assert asm.lib.cnn_exp(float("-inf")) == 0
    assert asm.lib.cnn_exp(float("inf")) == float("inf")
    assert np.isnan(asm.lib.cnn_log(float("nan")))
    assert np.isnan(asm.lib.cnn_log(-1.0))
    assert asm.lib.cnn_log(0.0) == float("-inf")
    assert asm.lib.cnn_log(float("inf")) == float("inf")

    def check(name, actual, expected, atol=3e-6, rtol=3e-5):
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        error = np.abs(a - b)
        entry = {
            "name": name,
            "values": int(a.size),
            "max_absolute_error": float(error.max()),
            "max_tolerance_ratio": float(np.max(error / (atol + rtol * np.abs(b)))),
            "atol": atol,
            "rtol": rtol,
        }
        checks.append(entry)
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=name)
        print(json.dumps(entry), flush=True)

    for name, values, reference in (
        ("exp", np.linspace(-90, 0, 401, dtype=np.float32), np.exp),
        ("log", np.geomspace(1e-30, 1e30, 401, dtype=np.float32), np.log),
    ):
        actual = [getattr(asm.lib, f"cnn_{name}")(float(v)) for v in values]
        check(name, actual, reference(values.astype(np.float64)).astype(np.float32), 3e-7, 3e-6)
    for i, logits in enumerate(
        [
            np.zeros(10, dtype=np.float32),
            np.linspace(-1000, 1000, 10, dtype=np.float32),
            *[
                rng.normal(0, scale, 10).astype(np.float32)
                for scale in (0.1, 1, 10, 100)
                for _ in range(5)
            ],
        ]
    ):
        label = i % 10
        x = torch.tensor(logits, requires_grad=True)
        target = F.cross_entropy(x[None], torch.tensor([label]))
        target.backward()
        loss, probs, grads = asm.loss(logits, label)
        check(f"loss.{i}", loss, target.item(), 5e-6, 2e-6)
        check(f"probabilities.{i}", probs, x.detach().softmax(0).numpy(), 3e-7, 3e-6)
        check(f"logit_gradients.{i}", grads, x.grad.numpy(), 3e-7, 3e-6)
    for channels, size, outputs in ((3, 32, 16), (16, 16, 32)):
        shape = np.array([channels, size, outputs], dtype=np.uint32)
        x = rng.normal(0, 0.5, (channels, size, size)).astype(np.float32)
        w = rng.normal(0, 0.05, (outputs, channels, 3, 3)).astype(np.float32)
        b = rng.normal(0, 0.05, outputs).astype(np.float32)
        upstream = rng.normal(0, 0.01, (outputs, size, size)).astype(np.float32)
        actual = np.full_like(upstream, np.nan)
        asm.lib.cnn_conv_forward(*map(asm.ptr, (x, w, b, actual, shape)))
        tx, tw, tb = [torch.tensor(a, requires_grad=True) for a in (x, w, b)]
        target = F.conv2d(tx[None], tw, tb, padding=1)[0]
        (target * torch.from_numpy(upstream)).sum().backward()
        check(f"conv{channels}.forward", actual, target.detach().numpy())
        dx, dw, db = np.full_like(x, np.nan), np.zeros_like(w), np.zeros_like(b)
        asm.lib.cnn_conv_backward(*map(asm.ptr, (x, w, upstream, dx, dw, db, shape)))
        check(f"conv{channels}.dx", dx, tx.grad.numpy())
        check(f"conv{channels}.dw", dw, tw.grad.numpy(), 1e-5, 5e-5)
        check(f"conv{channels}.db", db, tb.grad.numpy(), 1e-5, 5e-5)
        # Accumulation and nullable grad_input are separate API contracts.
        asm.lib.cnn_conv_backward(
            asm.ptr(x),
            asm.ptr(w),
            asm.ptr(upstream),
            None,
            asm.ptr(dw),
            asm.ptr(db),
            asm.ptr(shape),
        )
        check(f"conv{channels}.dw_accumulation", dw, 2 * tw.grad.numpy(), 2e-5, 5e-5)
        check(f"conv{channels}.db_accumulation", db, 2 * tb.grad.numpy(), 2e-5, 5e-5)
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        model = SimpleCNN(10)
        weights = flat_parameters(model).copy()
        gradients = np.zeros(5418, dtype=np.float32)
        # Zero input stresses ReLU/pool ties; other inputs cover sign changes/borders.
        inputs = [np.zeros((3, 32, 32), np.float32)] + [
            rng.uniform(-1, 1, (3, 32, 32)).astype(np.float32) for _ in range(4)
        ]
        for i, x in enumerate(inputs):
            logits = asm.forward(weights, x)
            target = model(torch.from_numpy(x)[None])[0]
            check(f"model{seed}.logits{i}", logits, target.detach().numpy())
            label = (seed + i) % 10
            loss, _, upstream = asm.loss(logits, label)
            asm.backward(weights, x, upstream, gradients)
            F.cross_entropy(target[None], torch.tensor([label])).backward()
        expected = torch.cat([p.grad.reshape(-1) for p in model.parameters()]).numpy()
        check(f"model{seed}.accumulated_gradients", gradients, expected, 1e-5, 1e-4)
    torch.manual_seed(42)
    params = torch.nn.Parameter(torch.randn(5418) * 0.1)
    weights = params.detach().numpy().copy()
    m, v = np.zeros_like(weights), np.zeros_like(weights)
    optimizer = torch.optim.Adam([params], lr=0.001, foreach=False)
    for step in range(1, 101):
        gradient = rng.normal(0, 0.03, 5418).astype(np.float32)
        gradient[::11] = 0  # zero and tiny denominators must be handled
        gradient[1::19] *= 1e-6
        params.grad = torch.from_numpy(gradient.copy())
        optimizer.step()
        asm.adam(weights, gradient, m, v, step)
        if step in (1, 2, 10, 25, 100):
            check(f"adam{step}.weights", weights, params.detach().numpy(), 2e-6, 2e-5)
            check(f"adam{step}.moment1", m, optimizer.state[params]["exp_avg"].numpy(), 2e-7, 2e-5)
            check(
                f"adam{step}.moment2", v, optimizer.state[params]["exp_avg_sq"].numpy(), 2e-8, 2e-5
            )
    # A coupled 10-update smoke catches layer/optimizer interaction errors.
    torch.manual_seed(42)
    model = SimpleCNN(10)
    weights = flat_parameters(model).copy()
    m, v = np.zeros_like(weights), np.zeros_like(weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False)
    x = rng.uniform(-1, 1, (8, 3, 32, 32)).astype(np.float32)
    labels = np.arange(8, dtype=np.int64)
    for step in range(1, 11):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(torch.from_numpy(x)), torch.from_numpy(labels))
        loss.backward()
        grads = np.zeros_like(weights)
        for sample, label in zip(x, labels, strict=True):
            logits = asm.forward(weights, sample)
            _, _, upstream = asm.loss(logits, int(label))
            asm.backward(weights, sample, upstream, grads)
        grads /= len(x)
        asm.adam(weights, grads, m, v, step)
        optimizer.step()
        check(f"coupled_update{step}", weights, flat_parameters(model), 1e-4, 2e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "library_sha256": digest(args.library),
                "source_hashes": {
                    str(path): digest(path)
                    for path in (Path("assembly/model.S"), Path("assembly/convolution.S"))
                },
                "checks": checks,
                "check_count": len(checks),
                "additional_math_domain_checks_passed": 7,
                "workspace_bytes": len(asm.workspace),
                "interpretation": (
                    "Tolerance-based parity of loss, probabilities, convolution forward/backward, "
                    "full model gradients, Adam and short training trajectory. "
                    "Does not imply bitwise identity or identical long-run trajectories."
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("assembly/build/libcnn.dylib"))
    parser.add_argument("--output", type=Path, default=Path("results/assembly-parity-v1.json"))
    verify(parser.parse_args())
