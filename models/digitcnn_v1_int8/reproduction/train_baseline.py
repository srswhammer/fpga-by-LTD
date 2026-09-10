"""Step 2: CPU float32 training; the official test set is never loaded."""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import DigitCNN

ROOT = Path(__file__).resolve().parent
CONFIG = dict(seed=42, split_seed=42, epochs=5, batch_size=64,
              learning_rate=0.001, optimizer="Adam", loss="CrossEntropyLoss",
              train_size=55000, val_size=5000, device="cpu", threads=4,
              preprocessing="ToTensor only; NCHW float32 [0,1]")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_data(config):
    pool = datasets.MNIST(ROOT / "data", train=True, download=False,
                          transform=transforms.ToTensor())
    return random_split(pool, [config["train_size"], config["val_size"]],
                        generator=torch.Generator().manual_seed(config["split_seed"]))


def evaluate(model, loader):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images)
            loss_sum += nn.functional.cross_entropy(scores, labels, reduction="sum").item()
            correct += (scores.argmax(1) == labels).sum().item()
            count += labels.numel()
    return dict(loss=loss_sum / count, accuracy=correct / count,
                correct=correct, count=count)


def smoke_check(train_set):
    """Separate disposable model; checks gradients and repeated-batch learning."""
    seed_all(123)
    model = DigitCNN()
    images = torch.stack([train_set[i][0] for i in range(32)])
    labels = torch.tensor([train_set[i][1] for i in range(32)])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial = nn.functional.cross_entropy(model(images), labels).item()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in model.parameters())
        optimizer.step()
    final = nn.functional.cross_entropy(model(images), labels).item()
    assert final < initial, (initial, final)
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    return dict(initial_loss=initial, final_loss=final, steps=20,
                sample_count=32, passed=True,
                note="Disposable model; not validation accuracy or final weights")


def train(run_dir):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    config = CONFIG.copy()
    config["model_sha256"] = hashlib.sha256((ROOT / "model.py").read_bytes()).hexdigest()
    config["torch_version"] = str(torch.__version__)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    train_set, val_set = make_data(config)
    assert not set(train_set.indices).intersection(val_set.indices)
    np.savez(run_dir / "split_indices.npz", train=np.array(train_set.indices),
             validation=np.array(val_set.indices))
    smoke = smoke_check(train_set)
    (run_dir / "smoke_check.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    print("Smoke check passed:", smoke, flush=True)

    # Start the actual experiment from fresh initialization after the smoke check.
    seed_all(config["seed"])
    model = DigitCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(config["seed"]), num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    initial_val = evaluate(model, val_loader)
    print("Untrained validation accuracy:", initial_val["accuracy"], flush=True)
    history, best_acc, best_scores, best_epoch = [], -1.0, None, None
    fixed_images, fixed_labels = next(iter(val_loader))
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            scores = model(images)
            loss = nn.functional.cross_entropy(scores, labels)
            assert bool(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            correct += (scores.argmax(1) == labels).sum().item()
            count += labels.numel()
        val = evaluate(model, val_loader)
        row = dict(epoch=epoch, train_loss=loss_sum/count, train_accuracy=correct/count,
                   val_loss=val["loss"], val_accuracy=val["accuracy"],
                   seconds=time.perf_counter()-epoch_start)
        history.append(row)
        if val["accuracy"] > best_acc:
            best_acc, best_epoch = val["accuracy"], epoch
            with torch.no_grad():
                best_scores = model(fixed_images).clone()
            torch.save(dict(model_state_dict=model.state_dict(), config=config,
                            epoch=epoch, val_accuracy=best_acc), run_dir / "best_model.pt")
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch}/5: train loss={row['train_loss']:.4f}, "
              f"train acc={row['train_accuracy']:.2%}, val acc={row['val_accuracy']:.2%}, "
              f"time={row['seconds']:.1f}s", flush=True)

    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    reloaded = DigitCNN().eval()
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        reloaded_scores = reloaded(fixed_images)
    torch.testing.assert_close(best_scores, reloaded_scores, rtol=0, atol=0)
    np.savez(run_dir / "reload_check.npz", inputs=fixed_images.numpy(),
             labels=fixed_labels.numpy(), expected_scores=best_scores.numpy())
    summary = dict(initial_validation=initial_val, best_epoch=best_epoch,
                   best_validation_accuracy=best_acc, training_seconds=time.perf_counter()-started,
                   parameter_count=sum(p.numel() for p in model.parameters()),
                   reload_exact_match=True, reload_sample_count=len(fixed_labels),
                   official_test_set_used=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved best model; exact reload check PASS. Official test set NOT used.", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "baseline_v1")
    args = parser.parse_args()
    train(args.run_dir)
