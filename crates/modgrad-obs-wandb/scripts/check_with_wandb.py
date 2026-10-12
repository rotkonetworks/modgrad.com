"""Check a .wandb file with W&B's own record reader (datastore.py + the
generated wandb_internal_pb2 Record) from a locally installed `wandb`
package, without importing the wandb package itself (whose __init__ may not
load, e.g. when the protobuf runtime is newer than wandb supports).

usage: python3 -P -E check_with_wandb.py <run-<id>.wandb>
         prints one JSON line per record and a final {"ok": ...} verdict.
       python3 -P -E check_with_wandb.py --write-sample <out.wandb>
         writes a small run with W&B's own writer (DataStore.write), to test
         `modgrad-wandb import` against files W&B produces.
"""
import importlib.util, json, os, shutil, sys, tempfile

def wandb_dir():
    spec = importlib.util.find_spec("wandb")  # locates without executing
    if spec is None or not spec.submodule_search_locations:
        sys.exit("wandb is not installed")
    return list(spec.submodule_search_locations)[0]

def shim():
    import google.protobuf
    src = wandb_dir()
    major = int(google.protobuf.__version__.split(".")[0])
    avail = sorted(int(d[1:]) for d in os.listdir(os.path.join(src, "proto")) if d[:1] == "v" and d[1:].isdigit())
    v = major if major in avail else avail[-1]
    root = tempfile.mkdtemp(prefix="wandb-shim-")
    pkg = os.path.join(root, "wandb")
    os.makedirs(os.path.join(pkg, "proto"))
    os.makedirs(os.path.join(pkg, "sdk", "internal"))
    for p in ["", "sdk"]:
        open(os.path.join(pkg, p, "__init__.py"), "w").close()
    open(os.path.join(pkg, "sdk", "internal", "__init__.py"), "w").close()
    open(os.path.join(pkg, "proto", "__init__.py"), "w").close()
    for f in os.listdir(os.path.join(src, "proto", f"v{v}")):
        if f.endswith("_pb2.py"):
            shutil.copy(os.path.join(src, "proto", f"v{v}", f), os.path.join(pkg, "proto", f))
    shutil.copy(os.path.join(src, "sdk", "internal", "datastore.py"), os.path.join(pkg, "sdk", "internal", "datastore.py"))
    sys.path.insert(0, root)
    return v, google.protobuf.__version__

def main(path):
    v, pbv = shim()
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto.wandb_internal_pb2 import Record
    ds = DataStore()
    ds.open_for_scan(path)
    kinds, steps, nums, run_id = [], [], [], None
    while True:
        data = ds.scan_data()
        if data is None:
            break
        r = Record()
        r.ParseFromString(data)
        k = r.WhichOneof("record_type")
        kinds.append(k)
        nums.append(r.num)
        line = {"num": r.num, "type": k}
        if k == "run":
            run_id = r.run.run_id
            line.update(run_id=r.run.run_id, project=r.run.project, name=r.run.display_name, config_keys=len(r.run.config.update))
        elif k == "history":
            steps.append(r.history.step.num)
            line.update(step=r.history.step.num, items={i.key: json.loads(i.value_json) for i in r.history.item if not i.value_json.startswith("{")})
        elif k == "summary":
            line.update(items=len(r.summary.update))
        elif k == "exit":
            line.update(exit_code=r.exit.exit_code, runtime=r.exit.runtime)
        print(json.dumps(line))
    ok = (kinds[:2] == ["header", "run"] and kinds[-2:] == ["exit", "final"] and run_id
          and steps == sorted(steps) and len(set(steps)) == len(steps) and nums == list(range(1, len(nums) + 1)))
    print(json.dumps({"ok": bool(ok), "records": len(kinds), "history": len(steps), "reader": f"wandb datastore.py + proto v{v} (protobuf runtime {pbv})"}))
    return 0 if ok else 1

def write_sample(path):
    shim()
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2 as pb
    ds = DataStore()
    ds.open_for_write(path)
    n = 0
    def put(**kw):
        nonlocal n
        n += 1
        ds.write(pb.Record(num=n, **kw))
    put(header=pb.HeaderRecord())
    run = pb.RunRecord(run_id="abc12345", project="sample", display_name="written-by-wandb")
    run.config.update.add(key="lr", value_json="0.001")
    run.start_time.seconds = 1760000000
    put(run=run)
    for step in range(5):
        h = pb.HistoryRecord()
        h.step.num = step
        h.item.add(key="_runtime", value_json=json.dumps(step * 1.5))
        h.item.add(key="loss", value_json=json.dumps(1.0 / (step + 1)))
        h.item.add(key="hist", value_json=json.dumps({"_type": "histogram", "values": [1, step], "bins": [0, 1, 2]}))
        h.item.add(key="img", value_json=json.dumps({"_type": "image-file", "path": "x.png"}))
        put(history=h)
    big = pb.HistoryRecord()  # spans blocks: FIRST/MIDDLE/LAST framing
    big.step.num = 5
    big.item.add(key="loss", value_json="0.1")
    big.item.add(key="blob", value_json=json.dumps("x" * 70000))
    put(history=big)
    s = pb.SummaryRecord()
    s.update.add(key="loss", value_json="0.1")
    put(summary=s)
    put(exit=pb.RunExitRecord(exit_code=0, runtime=9))
    put(final=pb.FinalRecord())
    ds.close()
    print(json.dumps({"wrote": path, "records": n}))

if __name__ == "__main__":
    if sys.argv[1] == "--write-sample":
        write_sample(sys.argv[2])
    else:
        sys.exit(main(sys.argv[1]))
