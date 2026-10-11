"""watch_grads, the HF callback and the Lightning logger."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import modgrad
from modgrad.torch import layer_assignment


def test_layer_assignment_naming():
    names = ["model.embed_tokens.weight", "model.layers.0.attn.q.weight", "model.layers.0.mlp.up.weight",
             "model.layers.1.attn.q.weight", "model.norm.weight", "lm_head.weight"]
    layers, a = layer_assignment(names)
    assert layers == ["model.layers.0", "model.layers.1"]
    assert a["model.layers.0.mlp.up.weight"] == ("layer", 0) and a["model.layers.1.attn.q.weight"] == ("layer", 1)
    assert a["model.embed_tokens.weight"] == ("params", "model.embed_tokens") and a["lm_head.weight"] == ("params", "lm_head")
    # encoder-decoder: one index space in model order
    layers, a = layer_assignment(["encoder.block.0.w", "encoder.block.1.w", "decoder.block.0.w", "decoder.block.0.b"])
    assert layers == ["encoder.block.0", "encoder.block.1", "decoder.block.0"] and a["decoder.block.0.b"] == ("layer", 2)
    # no numbered blocks: each parameter-owning module is a layer
    layers, a = layer_assignment(["fc1.weight", "fc1.bias", "fc2.weight"])
    assert layers == ["fc1", "fc2"] and a["fc2.weight"] == ("layer", 1)


def test_hf_callback_fake_loop(server):
    """A tiny Trainer-shaped loop: works whether or not transformers is installed."""
    from modgrad.integrations.huggingface import ModgradCallback

    cb = ModgradCallback(project="hf-fake")
    args = SimpleNamespace(run_name="fake-trainer", output_dir="out", learning_rate=5e-5, to_dict=lambda: {"learning_rate": 5e-5, "run_name": "fake-trainer"})
    state = SimpleNamespace(is_world_process_zero=True, global_step=0, best_metric=None)
    cb.on_train_begin(args, state, None, model=None)
    for step in range(1, 7):
        state.global_step = step
        if step % 2 == 0:
            cb.on_log(args, state, None, logs={"loss": 1.0 / step, "learning_rate": 5e-5, "epoch": step / 6})
    cb.on_log(args, state, None, logs={"eval_loss": 0.4, "epoch": 1.0})
    cb.on_log(args, state, None, logs={"train_runtime": 1.5, "train_loss": 0.3})
    cb.on_train_end(args, state, None)
    path = cb.run.path
    modgrad.finish()
    e = server.runs("hf-fake")[path]
    assert server.config(path)["learning_rate"] == 5e-5 and e["summary"]["train_runtime"] == 1.5
    h = server.history(path)
    assert h["steps"] == [2, 4, 6]
    assert h["columns"]["train/loss"] == [0.5, 0.25, pytest.approx(1 / 6)] and h["columns"]["eval/loss"][-1] == 0.4
    assert "train_runtime" not in h["columns"] and "train/train_runtime" not in h["columns"]


def test_hf_callback_non_zero_rank_is_silent():
    from modgrad.integrations.huggingface import ModgradCallback

    cb = ModgradCallback(project="hf-fake")
    state = SimpleNamespace(is_world_process_zero=False, global_step=1)
    cb.on_train_begin(None, state, None)
    cb.on_log(None, state, None, logs={"loss": 1.0})
    assert cb.run is None and modgrad.run is None


def test_watch_grads_plain_torch(server):
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return torch.relu(self.lin(x)) + x

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(4, 8)
            self.layers = nn.ModuleList([Block() for _ in range(3)])
            self.head = nn.Linear(8, 1)

        def forward(self, x):
            x = self.embed(x)
            for b in self.layers:
                x = b(x)
            return self.head(x)

    model = Net()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    run = modgrad.init(project="grads", name="torch")
    w = modgrad.watch_grads(model, every=2)
    for step in range(6):
        loss = model(torch.randn(16, 4)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        expect = {i: torch.sqrt(sum(p.grad.pow(2).sum() for p in model.layers[i].parameters())).item() for i in range(3)}
        opt.step()
        opt.zero_grad(set_to_none=True)  # grads gone before log(): hooks must have captured them
        modgrad.log({"loss": loss.item()})
        if step == 0:
            first = expect
    w.remove()
    path = run.path
    modgrad.finish()
    h = server.history(path)
    cols = h["columns"]
    assert {"layers/0/grad_norm", "layers/1/grad_norm", "layers/2/grad_norm", "params/embed/grad_norm", "params/head/grad_norm", "grad_norm/total"} <= set(cols)
    g0 = cols["layers/0/grad_norm"]
    assert [v is not None for v in g0] == [True, False, True, False, True, False]  # every=2
    assert g0[0] == pytest.approx(first[0], rel=1e-5) and cols["layers/2/grad_norm"][0] == pytest.approx(first[2], rel=1e-5)
    e = server.runs("grads")[path]
    assert server.config(path)["modgrad/layers"] == ["layers.0", "layers.1", "layers.2"]


def test_wandb_watch_maps_to_grad_norms(server):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    run = modgrad.init(project="grads", name="watch")
    modgrad.watch(model, log_freq=1)
    model(torch.randn(2, 4)).sum().backward()
    modgrad.log({"loss": 1.0})
    path = run.path
    modgrad.finish()
    cols = server.history(path)["columns"]
    assert "layers/0/grad_norm" in cols and "layers/1/grad_norm" in cols  # Sequential "0" and "2"


def test_hf_trainer_real(server, tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import torch
    from modgrad.integrations.huggingface import ModgradCallback

    cfg = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=64, n_positions=32)
    model = transformers.GPT2LMHeadModel(cfg)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, i):
            ids = torch.randint(0, 64, (16,))
            return {"input_ids": ids, "labels": ids}

    args = transformers.TrainingArguments(output_dir=str(tmp_path / "out"), max_steps=6, logging_steps=2, per_device_train_batch_size=4,
                                          report_to="none", save_strategy="no", run_name="tiny-gpt2", use_cpu=True, disable_tqdm=True)
    cb = ModgradCallback(project="hf-real", watch_grads_every=2)
    transformers.Trainer(model=model, args=args, train_dataset=DS(), callbacks=[cb]).train()
    path = cb.run.path
    assert path.startswith("hf-real/tiny-gpt2-")
    modgrad.finish()
    e = server.runs("hf-real")[path]
    assert server.config(path)["n_layer"] == 2 and server.config(path)["max_steps"] == 6 and "train_runtime" in e["summary"]
    h = server.history(path)
    assert h["steps"][:3] == [2, 4, 6] and len([v for v in h["columns"]["train/loss"] if v is not None]) == 3
    assert {"layers/0/grad_norm", "layers/1/grad_norm"} <= set(h["columns"])
    assert server.config(path)["modgrad/layers"] == ["transformer.h.0", "transformer.h.1"]


def test_lightning_logger_real(server, tmp_path):
    pytest.importorskip("torch")
    L = pytest.importorskip("lightning")
    import torch
    from modgrad.integrations.lightning import ModgradLogger

    class M(L.LightningModule):
        def __init__(self, lr=0.1):
            super().__init__()
            self.save_hyperparameters()
            self.lin = torch.nn.Linear(4, 1)

        def training_step(self, batch, i):
            loss = self.lin(batch[0]).pow(2).mean()
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)

    data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(64, 4)), batch_size=8)
    logger = ModgradLogger(project="lightning", name="tiny")
    trainer = L.Trainer(max_epochs=1, logger=logger, log_every_n_steps=2, enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, default_root_dir=str(tmp_path), accelerator="cpu")
    trainer.fit(M(), data)
    path = logger.experiment.path
    logger.finish()
    e = server.runs("lightning")[path]
    assert e["finished"] and server.config(path)["lr"] == 0.1
    h = server.history(path)
    assert len([v for v in h["columns"]["train_loss"] if v is not None]) == 4
