use super::*;
use std::path::PathBuf;

fn fx(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join(name)
}

/// Ingest a fixture in small chunks (exercises the carried partial line).
fn load(name: &str, chunk: usize) -> (LocalRun, FileReport) {
    let bytes = std::fs::read(fx(name)).unwrap();
    let mut run = LocalRun::new(name);
    let rep = ingest_bytes(&mut run, name, &bytes, chunk);
    run.finalize();
    (run, rep)
}

fn series<'a>(run: &'a LocalRun, k: &str) -> &'a Series {
    run.series.get(k).unwrap_or_else(|| panic!("no series {k}; have {:?}", run.series.keys().collect::<Vec<_>>()))
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-9 * y.abs().max(1.0))
}

#[test]
fn detect_formats() {
    let d = |n: &str| detect(n, &std::fs::read(fx(n)).unwrap());
    assert_eq!(d("train.csv"), Format::Csv(b','));
    assert_eq!(d("iters.tsv"), Format::Csv(b'\t'));
    assert_eq!(d("history.jsonl"), Format::Jsonl);
    assert_eq!(d("text_step.log"), Format::Text);
    assert_eq!(d("ctmlm_pilot_log.txt"), Format::Legacy);
    assert_eq!(d("bkk07_mid.wandb"), Format::Wandb);
    assert_eq!(d("rundir_bkk07_mid/metrics.jsonl"), Format::RunMetrics);
    // content sniffing without a helpful name
    assert_eq!(detect("data", b"step,loss\n1,2.0\n2,1.5\n"), Format::Csv(b','));
    assert_eq!(detect("out", b"{\"step\": 1, \"loss\": 2}\n"), Format::Jsonl);
    let tb = std::fs::read_dir(fx("tb_tf2")).unwrap().map(|e| e.unwrap().path()).find(|p| p.to_string_lossy().contains("tfevents")).unwrap();
    assert_eq!(detect("renamed.bin", &std::fs::read(tb).unwrap()), Format::TfEvents);
    assert_eq!(detect("x", b"\x00\x01\x02garbage"), Format::Binary);
}

#[test]
fn csv_quotes_gaps_units() {
    for chunk in [7, 1 << 20] {
        let (run, rep) = load("train.csv", chunk);
        assert_eq!(rep.format, "csv");
        assert_eq!(rep.error, None);
        assert_eq!(run.series.keys().collect::<Vec<_>>(), ["lr", "train/loss", "val acc"]);
        let loss = series(&run, "train/loss");
        assert_eq!(loss.step, [0.0, 100.0, 200.0, 300.0, 400.0]);
        assert!(same(&loss.value, &[3.5, 2.91, f64::NAN, 2.10, 1.95]));
        // empty cell = no point; "12.5%" = 12.5; inf = NaN point (a break)
        let acc = series(&run, "val acc");
        assert_eq!(acc.step, [100.0, 200.0, 300.0, 400.0]);
        assert!(same(&acc.value, &[12.5, 25.0, f64::NAN, 40.0]));
        assert!(rep.notes.iter().any(|n| n.contains("note")), "{:?}", rep.notes);
        assert_eq!((rep.metrics, rep.steps, rep.points), (3, 5, 14));
    }
}

#[test]
fn csv_pandas_index_and_tsv() {
    let (run, rep) = load("pandas_index.csv", 64);
    // epoch repeats -> not the x axis; the unnamed increasing index is
    assert!(rep.notes.iter().any(|n| n.contains("\"index\"")), "{:?}", rep.notes);
    assert_eq!(series(&run, "loss").step, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(series(&run, "epoch").value, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    assert_eq!(series(&run, "throughput").value, [1200.0, 1300.0, 1300.0, 1400.0, 1400.0]);
    let (run, rep) = load("iters.tsv", 5);
    assert_eq!(rep.format, "tsv");
    assert_eq!(series(&run, "loss").step, [10.0, 20.0, 30.0]);
    assert_eq!(series(&run, "time").value, [1.0, 1.1, 1.0]);
}

#[test]
fn jsonl_wandb_history() {
    let (run, rep) = load("history.jsonl", 13);
    assert_eq!(rep.format, "jsonl");
    let loss = series(&run, "loss");
    assert_eq!(loss.step, [0.0, 1.0, 2.0, 3.0, 4.0]);
    assert!(same(&loss.value, &[3.2, 2.9, f64::NAN, f64::NAN, 2.1]));
    assert_eq!(series(&run, "eval/acc").value, [0.4]);
    assert_eq!(series(&run, "done").value, [0.0, 1.0]);
    assert_eq!(series(&run, "acc_str").value, [55.0]);
    assert!(!run.series.contains_key("_runtime") && !run.series.contains_key("_timestamp") && !run.series.contains_key("weights"));
    // the broken line and the unterminated last line are skipped
    assert_eq!(rep.skipped, 2);
    assert!(rep.notes.iter().any(|n| n.contains("non-scalar")), "{:?}", rep.notes);
    assert_eq!(run.time, [(0.0, 1.5), (1.0, 2.5), (2.0, 3.5), (4.0, 5.5)]);
}

#[test]
fn jsonl_nested_metrics() {
    let (run, rep) = load("nested.jsonl", 1 << 20);
    assert_eq!(run.series.keys().collect::<Vec<_>>(), ["acc/top1", "acc/top5", "loss"]);
    assert_eq!(series(&run, "acc/top5").step, [10.0, 20.0, 30.0]);
    assert_eq!(rep.steps, 3);
}

#[test]
fn text_step_lines() {
    let (run, rep) = load("text_step.log", 9);
    assert_eq!(run.series.keys().collect::<Vec<_>>(), ["acc", "grad_norm", "loss"]);
    assert_eq!(series(&run, "loss").step, [1000.0, 1250.0, 1500.0, 1750.0]);
    assert_eq!(series(&run, "loss").value, [3.92, 3.48, 3.30, 3.21]);
    assert_eq!(series(&run, "acc").value, [5.0, 7.5, 9.1, 10.2]);
    // grad_norm is on one step line only: kept (the rare-key filter is for stepless logs)
    assert_eq!(series(&run, "grad_norm").value, [0.82]);
    assert_eq!(rep.steps, 4);
}

#[test]
fn text_iter_lines() {
    let (run, rep) = load("text_iter.log", 1 << 20);
    assert_eq!(run.series.keys().collect::<Vec<_>>(), ["loss", "lr", "tok/s"]);
    assert_eq!(series(&run, "lr").value, [3e-4, 3e-4, 2.5e-4, 2e-4]);
    assert_eq!(series(&run, "tok/s").step, [100.0, 200.0, 300.0, 400.0]);
    assert!(series(&run, "loss").value[2].is_nan());
    assert_eq!(rep.skipped, 1); // the "starting run" line
}

#[test]
fn text_epoch_and_hf() {
    let (run, _) = load("text_epoch.log", 1 << 20);
    assert_eq!(series(&run, "train_loss").step, [1.0, 2.0, 3.0]);
    assert_eq!(series(&run, "val_acc").value, [61.0, 74.5, 77.0]);
    let (run, rep) = load("text_hf.log", 11);
    // tqdm lines have no k=v pairs: skipped; dict lines: x = line order (epoch repeats)
    assert_eq!(series(&run, "loss").value, [2.3127, 2.0101, 1.8]);
    assert_eq!(series(&run, "eval_loss").value, [2.2]);
    assert_eq!(series(&run, "learning_rate").value, [5e-5, 4.5e-5, 4e-5]);
    assert!(!run.series.keys().any(|k| k.contains("Training")), "{:?}", run.series.keys());
    assert!(rep.notes.iter().any(|n| n.contains("line order")), "{:?}", rep.notes);
}

#[test]
fn text_extract_shapes() {
    let e = |l: &str| text::extract(l);
    assert_eq!(e("step 1250: loss 3.48 acc 7.5%"), [("step".into(), 1250.0), ("loss".into(), 3.48), ("acc".into(), 7.5)]);
    assert_eq!(e("iter 100 | loss=2.31 | lr 3e-4"), [("iter".into(), 100.0), ("loss".into(), 2.31), ("lr".into(), 3e-4)]);
    assert_eq!(e("[epoch 2/10] train_loss: 0.45"), [("epoch".into(), 2.0), ("train_loss".into(), 0.45)]);
    assert_eq!(e("train loss 0.4, val loss 0.5"), [("train_loss".into(), 0.4), ("val_loss".into(), 0.5)]);
    assert_eq!(e("Epoch 1: 45%|####  | 450/1000 [00:12<00:15, 36.2it/s, loss=0.234, v_num=0]"), [("loss".into(), 0.234), ("v_num".into(), 0.0)]);
    assert_eq!(e("2026-10-11 12:00:00 saved to /tmp/x"), []);
}

#[test]
fn legacy_pilot_log_via_modgrad_obs() {
    let (run, rep) = load("ctmlm_pilot_log.txt", 100);
    assert_eq!(rep.format, "ctmlm_pilot log");
    assert_eq!(rep.error, None);
    assert_eq!(run.config["model"], "SmolLM2-360M");
    let l = series(&run, "train_loss_ema");
    assert_eq!(l.step.first(), Some(&1.0));
    // same numbers as the on-disk converter wrote for this run
    let rd = modgrad_obs::read_run(&fx("rundir_bkk07_mid")).unwrap();
    let want: Vec<(u64, f64)> = rd.series("train_loss_ema");
    assert_eq!(l.step, want.iter().map(|p| p.0 as f64).collect::<Vec<_>>());
    assert_eq!(l.value, want.iter().map(|p| p.1).collect::<Vec<_>>());
}

#[test]
fn wandb_offline_file() {
    let (run, rep) = load("bkk07_mid.wandb", 4096);
    assert_eq!(rep.error, None);
    assert_eq!(rep.format, "wandb");
    let rd = modgrad_obs::read_run(&fx("rundir_bkk07_mid")).unwrap();
    for k in ["train_loss_ema", "brain_heldout_nll", "nat_brain"] {
        let want: Vec<(u64, f64)> = rd.series(k).into_iter().filter(|p| p.1.is_finite()).collect();
        let s = series(&run, k);
        assert_eq!(s.step, want.iter().map(|p| p.0 as f64).collect::<Vec<_>>(), "{k}");
        assert_eq!(s.value, want.iter().map(|p| p.1).collect::<Vec<_>>(), "{k}");
    }
    assert_eq!(run.config["model"], "SmolLM2-360M");
}

#[test]
fn tensorboard_matches_tensorboards_own_reader() {
    for dir in ["tb_torch", "tb_torch_new", "tb_tf2"] {
        let mut run = LocalRun::new(dir);
        let mut reps = Vec::new();
        for e in std::fs::read_dir(fx(dir)).unwrap() {
            let p = e.unwrap().path();
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            if name.contains("tfevents") {
                reps.push(ingest_bytes(&mut run, &name, &std::fs::read(&p).unwrap(), 1000));
            }
        }
        run.finalize();
        let exp: serde_json::Value = serde_json::from_slice(&std::fs::read(fx(dir).join("expected.json")).unwrap()).unwrap();
        let exp = exp.as_object().unwrap();
        assert_eq!(run.series.keys().collect::<Vec<_>>(), exp.keys().collect::<Vec<_>>(), "{dir}");
        for (k, v) in exp {
            let s = series(&run, k);
            let st: Vec<f64> = v["step"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            let va: Vec<f64> = v["value"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            assert_eq!(s.step, st, "{dir}/{k}");
            assert!(same(&s.value, &va), "{dir}/{k}: {:?} vs {:?}", s.value, va);
        }
        assert!(reps.iter().all(|r| r.error.is_none() && r.format == "tensorboard"));
        if dir == "tb_torch" {
            let n = &reps[0].notes.join(" ");
            assert!(n.contains("histogram") && n.contains("text"), "{n}");
        }
    }
}

#[test]
fn tensorboard_truncated_and_corrupt() {
    use prost::Message;
    let mut buf = Vec::new();
    let ev = |step: i64, v: f32| tfevents::Event { wall_time: 100.0 + step as f64, step, file_version: None, summary: Some(tfevents::Summary { value: vec![tfevents::SummaryValue { tag: "loss".into(), simple_value: Some(v), ..Default::default() }] }) };
    buf.extend(tfevents::frame(&tfevents::Event { file_version: Some("brain.Event:2".into()), ..Default::default() }.encode_to_vec()));
    for s in 0..10 {
        buf.extend(tfevents::frame(&ev(s, s as f32 * 0.5).encode_to_vec()));
    }
    // a file still being written: half a record at the end
    let full = buf.len();
    buf.extend(&tfevents::frame(&ev(10, 5.0).encode_to_vec())[..20]);
    let mut run = LocalRun::new("t");
    let rep = ingest_bytes(&mut run, "events.out.tfevents.1.host", &buf, 333);
    assert_eq!(series(&run, "loss").step.len(), 10);
    assert!(rep.notes.iter().any(|n| n.contains("truncated")), "{:?}", rep.notes);
    assert_eq!(run.time.last(), Some(&(9.0, 9.0)));
    // flip a byte inside record 5's data: stop there, keep 0..4
    let mut bad = buf[..full].to_vec();
    let (recs, _) = tfevents::records(&bad);
    let off = recs[5].as_ptr() as usize - bad.as_ptr() as usize;
    bad[off + 3] ^= 0xff;
    let mut run = LocalRun::new("t");
    let rep = ingest_bytes(&mut run, "events.out.tfevents.2.host", &bad, 1 << 20);
    assert_eq!(series(&run, "loss").step, [0.0, 1.0, 2.0, 3.0]);
    assert!(rep.notes.iter().any(|n| n.contains("CRC")), "{:?}", rep.notes);
}

#[test]
fn own_run_directory() {
    // a run written by modgrad-obs, with arrays, dropped as a folder
    let root = std::env::temp_dir().join(format!("obs-ingest-rd-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let mut r = modgrad_obs::Run::init(&root, "proj", "r1", &serde_json::json!({"lr": 1e-3, "brain_params": 1000})).unwrap();
    for s in 0..20u64 {
        r.log_with_time(s, s as f64 * 2.0, &[("loss", 3.0 - s as f64 * 0.1)]).unwrap();
    }
    r.log_array_with_time(10, 20.0, "recall", &[3], &[1.0, 2.0, 3.0]).unwrap();
    r.log_histogram_with_time(10, 20.0, "w", &modgrad_obs::Histogram { edges: vec![0.0, 1.0, 2.0], counts: vec![3.0, 4.0] }).unwrap();
    r.event_with_time(5, 10.0, "checkpoint").unwrap();
    r.finish().unwrap();
    let dir = root.join("runs/r1");
    let mut run = LocalRun::new("r1");
    for f in ["config.json", "metrics.jsonl", "arrays.jsonl", "events.jsonl", "summary.json"] {
        let rep = ingest_bytes(&mut run, &format!("r1/{f}"), &std::fs::read(dir.join(f)).unwrap(), 100);
        assert_eq!(rep.error, None, "{f}");
    }
    run.finalize();
    assert_eq!(run.config["lr"], 1e-3);
    assert_eq!(run.meta["project"], "proj");
    assert!(run.finished);
    assert_eq!(series(&run, "loss").step.len(), 20);
    assert!(!run.series.contains_key("t") && !run.series.contains_key("step"));
    assert_eq!(run.time[3], (3.0, 6.0));
    assert_eq!(run.events.len(), 1);
    let m = run.meta_json();
    assert_eq!(m["arrays"]["w"]["kind"], "histogram");
    assert_eq!(m["arrays"]["recall"]["n_steps"], 1);
    assert_eq!(m["array_files"]["w"]["edges"][0], serde_json::json!([0.0, 1.0, 2.0]));
    assert_eq!(m["last_step"], 19.0);
    assert_eq!(m["n_rows"], 20);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn real_rundir_fixture() {
    let mut run = LocalRun::new("mid");
    for f in ["config.json", "metrics.jsonl", "events.jsonl", "summary.json"] {
        ingest_bytes(&mut run, &format!("mid/{f}"), &std::fs::read(fx("rundir_bkk07_mid").join(f)).unwrap(), 512);
    }
    run.finalize();
    let rd = modgrad_obs::read_run(&fx("rundir_bkk07_mid")).unwrap();
    assert_eq!(run.series.len(), rd.metric_keys().len());
    for k in rd.metric_keys() {
        let want = rd.series(&k);
        assert_eq!(series(&run, &k).step, want.iter().map(|p| p.0 as f64).collect::<Vec<_>>(), "{k}");
    }
}

#[test]
fn large_text_log_streams() {
    // 200k lines in 1 MiB chunks: correctness of the carried line at scale
    let mut s = String::new();
    for i in 0..200_000 {
        s.push_str(&format!("step {i}: loss {:.4} lr 3e-4\n", 4.0 / (1.0 + i as f64 * 1e-3)));
    }
    let mut run = LocalRun::new("big");
    let rep = ingest_bytes(&mut run, "train.log", s.as_bytes(), 1 << 20);
    run.finalize();
    assert_eq!(series(&run, "loss").step.len(), 200_000);
    assert_eq!(rep.skipped, 0);
    assert_eq!(rep.steps, 200_000);
}

#[test]
fn duplicate_steps_keep_last() {
    let mut run = LocalRun::new("d");
    ingest_bytes(&mut run, "a.csv", b"step,loss\n1,5\n2,4\n2,3.5\n0,6\n", 1 << 20);
    run.finalize();
    assert_eq!(series(&run, "loss").step, [0.0, 1.0, 2.0]);
    assert_eq!(series(&run, "loss").value, [6.0, 5.0, 3.5]);
}
