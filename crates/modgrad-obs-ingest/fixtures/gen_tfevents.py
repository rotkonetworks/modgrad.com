"""Regenerate the TensorBoard fixtures with the real writers.

    python3 -m venv --system-site-packages env && env/bin/pip install tensorboard
    env/bin/python gen_tfevents.py   # (needs torch for torch_*)

* tb_torch/       torch.utils.tensorboard.SummaryWriter, default (simple_value)
                  plus a histogram and a text summary that must be skipped
* tb_torch_new/   SummaryWriter scalars with new_style=True (tensor float_val)
* tb_tf2/         TF2 tf.summary.scalar layout (tensor_content, plugin
                  "scalars", DATA_CLASS_SCALAR), written with tensorboard's
                  own protos and EventFileWriter, float and double
Each directory gets expected.json from tensorboard's EventAccumulator, the
reference the Rust tests compare against.
"""
import json, os, shutil, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def reset(d):
    p = os.path.join(HERE, d)
    shutil.rmtree(p, ignore_errors=True)
    os.makedirs(p)
    return p

def expected(d):
    from tensorboard.backend.event_processing import event_accumulator as ea
    acc = ea.EventAccumulator(d, size_guidance={ea.SCALARS: 0, ea.TENSORS: 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        out[tag] = {"step": [e.step for e in acc.Scalars(tag)], "value": [e.value for e in acc.Scalars(tag)]}
    for tag in acc.Tags().get("tensors", []):
        evs = acc.Tensors(tag)
        from tensorboard.util import tensor_util
        vals = [tensor_util.make_ndarray(e.tensor_proto) for e in evs]
        if all(v.size == 1 for v in vals) and all(v.dtype.kind in "fiu" for v in vals):
            out[tag] = {"step": [e.step for e in evs], "value": [float(v.reshape(())) for v in vals]}
    with open(os.path.join(d, "expected.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)

def torch_runs():
    from torch.utils.tensorboard import SummaryWriter
    d = reset("tb_torch")
    w = SummaryWriter(log_dir=d)
    rng = np.random.default_rng(0)
    for s in range(0, 200, 10):
        w.add_scalar("loss/train", 3.0 * np.exp(-s / 80) + 0.01 * rng.standard_normal(), s)
        if s % 50 == 0:
            w.add_scalar("loss/val", 3.1 * np.exp(-s / 90), s)
            w.add_histogram("weights", rng.standard_normal(100), s)
            w.add_text("note", f"step {s}", s)
    w.add_scalar("lr", 3e-4, 0)
    w.close()
    expected(d)
    d = reset("tb_torch_new")
    w = SummaryWriter(log_dir=d)
    for s in range(1, 31):
        w.add_scalar("acc", s / 30.0, s, new_style=True)
        w.add_scalar("grad_norm", 1.0 / s, s, new_style=True, double_precision=True)
    w.close()
    expected(d)

def tf2_run():
    from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, tensor_shape_pb2, types_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    d = reset("tb_tf2")
    w = EventFileWriter(d)
    meta = summary_pb2.SummaryMetadata(plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name="scalars"), data_class=summary_pb2.DATA_CLASS_SCALAR)
    for s in range(0, 100, 5):
        vals = []
        for tag, dt, fmt, x in (("epoch_loss", types_pb2.DT_FLOAT, "<f", 2.0 / (1 + s)), ("epoch_accuracy", types_pb2.DT_DOUBLE, "<d", s / 100.0)):
            t = tensor_pb2.TensorProto(dtype=dt, tensor_shape=tensor_shape_pb2.TensorShapeProto(), tensor_content=struct.pack(fmt, x))
            vals.append(summary_pb2.Summary.Value(tag=tag, tensor=t, metadata=meta))
        w.add_event(event_pb2.Event(wall_time=1.7e9 + s, step=s, summary=summary_pb2.Summary(value=vals)))
    w.close()
    expected(d)

if __name__ == "__main__":
    tf2_run()
    if "--no-torch" not in sys.argv:
        torch_runs()
