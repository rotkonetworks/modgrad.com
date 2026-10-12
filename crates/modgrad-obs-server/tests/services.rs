//! Service-level tests: no HTTP; services are called with `oneshot`.

use modgrad_obs::Item;
use modgrad_obs_server::*;
use proptest::prelude::*;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tower::{Service, ServiceExt};

fn tmp(tag: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!("obs-server-{tag}-{}-{}", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn log(run: &str, step: u64, x: f64) -> IngestRequest {
    IngestRequest::Log { run: run.into(), step, t: Some(step as f64), items: vec![("x".into(), Item::Scalar(x))] }
}

async fn history(read: &ReadState, run: &str) -> HistoryBlock {
    match read.clone().oneshot(ReadRequest::History { run: run.into(), keys: vec![], range: 0..u64::MAX, stride: 1 }).await.unwrap() {
        ReadResponse::History(b) => b,
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn state_and_read_oneshot() {
    let root = tmp("state");
    let (state, rx) = State::open(&root).unwrap();
    let read = ReadState::new(rx);
    let mut state = tower::buffer::Buffer::new(state, 16);
    let r = state.ready().await.unwrap().call(StateRequest::CreateRun { name: "proj/a".into(), project: Some("p".into()), config: json!({"d": 8}), meta: None }).await.unwrap();
    assert_eq!(r, StateResponse::Created { run: "proj/a".into() });
    let recs = vec![
        Record::Log { step: 1, t: Some(1.0), items: vec![("loss".into(), Item::Scalar(2.0)), ("acc".into(), Item::Scalar(0.1))] },
        Record::Log { step: 2, t: Some(2.0), items: vec![("loss".into(), Item::Scalar(1.0))] },
        Record::Array { step: 2, t: None, key: "recall".into(), kind: "tensor".into(), shape: vec![4], data: vec![1.0, 2.0, 3.0, 4.0], labels: None },
        Record::Event { step: 2, t: None, text: "hello".into() },
    ];
    let r = state.ready().await.unwrap().call(StateRequest::Append { run: "proj/a".into(), records: recs }).await.unwrap();
    assert!(matches!(r, StateResponse::Appended { records: 4, .. }));
    // invalid record rejects the whole append, nothing written
    let before = read.snapshot().runs["proj/a"].committed;
    let bad = vec![Record::Log { step: 3, t: None, items: vec![("x".into(), Item::Scalar(1.0))] }, Record::Array { step: 3, t: None, key: "k".into(), kind: "tensor".into(), shape: vec![3], data: vec![1.0], labels: None }];
    let e = state.ready().await.unwrap().call(StateRequest::Append { run: "proj/a".into(), records: bad }).await.unwrap_err();
    assert!(matches!(e.downcast_ref::<StateError>(), Some(StateError::Obs(_))), "{e}");
    assert_eq!(read.snapshot().runs["proj/a"].committed, before);
    state.ready().await.unwrap().call(StateRequest::Finish { run: "proj/a".into(), summary: json!({"best": 1.0}) }).await.unwrap();

    let b = history(&read, "proj/a").await;
    assert_eq!(b.steps, vec![1, 2]);
    let acc = &b.columns.iter().find(|c| c.0 == "acc").unwrap().1;
    assert!(acc[1].is_nan());
    let ReadResponse::Arrays { blocks, .. } = read.clone().oneshot(ReadRequest::Arrays { run: "proj/a".into(), key: "recall".into(), steps: None, index_range: Some(1..3) }).await.unwrap() else { panic!() };
    assert_eq!((blocks[0].offset, blocks[0].data.clone(), blocks[0].shape.clone()), (1, vec![2.0, 3.0], vec![4]));
    let ReadResponse::Summary(Some(s)) = read.clone().oneshot(ReadRequest::Summary("proj/a".into())).await.unwrap() else { panic!() };
    assert_eq!(s["summary"]["best"], 1.0);
    assert_eq!(s["summary"]["loss"], 1.0);
    let ReadResponse::Runs(runs) = read.clone().oneshot(ReadRequest::ListRuns { filter: None }).await.unwrap() else { panic!() };
    assert_eq!(runs[0]["id"], "proj/a");
    assert_eq!(runs[0]["config"]["d"], 8);
    assert!(matches!(read.clone().oneshot(ReadRequest::Config("nope".into())).await, Err(ReadError::UnknownRun(_))));
    // invalid names are refused
    let e = state.ready().await.unwrap().call(StateRequest::CreateRun { name: "../x".into(), project: None, config: json!({}), meta: None }).await.unwrap_err();
    assert!(matches!(e.downcast_ref::<StateError>(), Some(StateError::InvalidRunName(_))));
}

#[tokio::test]
async fn ingest_stack_end_to_end() {
    let root = tmp("ingest");
    let (ingest, read) = open(&root, IngestConfig::default()).unwrap();
    ingest.clone().oneshot(IngestRequest::CreateRun { name: "r".into(), project: None, config: json!({}), meta: None }).await.unwrap();
    let futs: Vec<_> = (0..50u64).map(|s| ingest.clone().oneshot(log("r", s, s as f64))).collect();
    for f in futures_join(futs).await {
        assert_eq!(f.unwrap(), IngestResponse::Accepted);
    }
    let e = ingest.clone().oneshot(log("missing", 0, 0.0)).await.unwrap_err();
    assert!(matches!(IngestError::classify(e), IngestError::State(StateError::UnknownRun(_))));
    ingest.clone().oneshot(IngestRequest::Finish { run: "r".into(), summary: json!({}) }).await.unwrap();
    let b = history(&read, "r").await;
    assert_eq!(b.steps.len(), 50);
    assert!(read.snapshot().runs["r"].finished);
}

async fn futures_join<F: std::future::Future + Send + 'static>(fs: Vec<F>) -> Vec<F::Output>
where
    F::Output: Send + 'static,
{
    let hs: Vec<_> = fs.into_iter().map(tokio::spawn).collect();
    let mut out = Vec::new();
    for h in hs {
        out.push(h.await.unwrap());
    }
    out
}

#[tokio::test]
async fn batch_layer_groups_per_run_and_barriers_flush_first() {
    let (mock, mut handle) = tower_test::mock::pair::<StateRequest, StateResponse>();
    let batcher = Batcher::new(mock, BatchConfig { max_items: 100, max_latency: Duration::from_millis(50), queue: 64 });
    let mut calls = Vec::new();
    for s in 0..3 {
        calls.push(tokio::spawn(batcher.clone().oneshot(log("a", s, 0.0))));
    }
    calls.push(tokio::spawn(batcher.clone().oneshot(log("b", 0, 0.0))));
    let mut seen = Vec::new();
    for _ in 0..2 {
        let (req, send) = handle.next_request().await.unwrap();
        match req {
            StateRequest::Append { run, records } => {
                seen.push((run.clone(), records.len()));
                send.send_response(StateResponse::Appended { run, records: records.len(), committed: [0; 3] });
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(seen, vec![("a".to_string(), 3), ("b".to_string(), 1)]);
    for c in calls {
        assert_eq!(c.await.unwrap().unwrap(), IngestResponse::Accepted);
    }
    // a log followed by a Finish: the log's append must reach State first
    let c1 = tokio::spawn(batcher.clone().oneshot(log("a", 9, 0.0)));
    tokio::task::yield_now().await;
    let c2 = tokio::spawn(batcher.clone().oneshot(IngestRequest::Finish { run: "a".into(), summary: json!({}) }));
    let (req, send) = handle.next_request().await.unwrap();
    assert!(matches!(req, StateRequest::Append { ref records, .. } if records.len() == 1), "{req:?}");
    send.send_response(StateResponse::Appended { run: "a".into(), records: 1, committed: [0; 3] });
    let (req, send) = handle.next_request().await.unwrap();
    assert!(matches!(req, StateRequest::Finish { .. }));
    send.send_response(StateResponse::Finished { run: "a".into() });
    c1.await.unwrap().unwrap();
    assert_eq!(c2.await.unwrap().unwrap(), IngestResponse::Finished { run: "a".into() });
    // a State error reaches every caller in the batch, typed
    let c = tokio::spawn(batcher.clone().oneshot(log("z", 0, 0.0)));
    let (_, send) = handle.next_request().await.unwrap();
    send.send_error(StateError::UnknownRun("z".into()));
    assert!(matches!(c.await.unwrap(), Err(IngestError::State(StateError::UnknownRun(_)))));
}

#[tokio::test]
async fn load_shed_when_concurrency_exhausted() {
    let (mock, mut handle) = tower_test::mock::pair::<StateRequest, StateResponse>();
    let ingest = ingest_service(mock, IngestConfig { batch: BatchConfig { max_items: 1, max_latency: Duration::from_millis(1), queue: 8 }, concurrency: 1, timeout: Duration::from_secs(5) });
    let first = tokio::spawn(ingest.clone().oneshot(log("a", 0, 0.0)));
    let (_req, send) = handle.next_request().await.unwrap(); // first is now in flight, holding the permit
    let second = ingest.clone().oneshot(log("a", 1, 0.0)).await;
    assert!(matches!(IngestError::classify(second.unwrap_err()), IngestError::Overloaded));
    send.send_response(StateResponse::Appended { run: "a".into(), records: 1, committed: [0; 3] });
    first.await.unwrap().unwrap();
}

#[tokio::test]
async fn timeout_is_typed() {
    let (mock, mut handle) = tower_test::mock::pair::<StateRequest, StateResponse>();
    let ingest = ingest_service(mock, IngestConfig { batch: BatchConfig { max_items: 1, max_latency: Duration::from_millis(1), queue: 8 }, concurrency: 4, timeout: Duration::from_millis(30) });
    let call = tokio::spawn(ingest.clone().oneshot(log("a", 0, 0.0)));
    let _held = handle.next_request().await.unwrap(); // never answered
    assert!(matches!(IngestError::classify(call.await.unwrap().unwrap_err()), IngestError::Timeout));
}

#[tokio::test]
async fn crash_mid_batch_leaves_store_readable() {
    let root = tmp("crash");
    {
        let (ingest, _read) = open(&root, IngestConfig::default()).unwrap();
        ingest.clone().oneshot(IngestRequest::CreateRun { name: "r".into(), project: None, config: json!({}), meta: None }).await.unwrap();
        for s in 0..5 {
            ingest.clone().oneshot(log("r", s, s as f64)).await.unwrap();
        }
    }
    // the process died inside write_all of the next batch: a torn tail
    let m = root.join("runs/r/metrics.jsonl");
    let mut f = std::fs::OpenOptions::new().append(true).open(&m).unwrap();
    std::io::Write::write_all(&mut f, b"{\"step\":5,\"t\":5,\"x\":5}\n{\"step\":6,\"t\":6,").unwrap();
    drop(f);
    // a reader on the old store never sees the torn line (raw reader skips it)
    assert_eq!(modgrad_obs::read_run(&root.join("runs/r")).unwrap().series("x").len(), 6);
    // restart: the torn tail is repaired, no partial line remains, appends continue cleanly
    let (ingest, read) = open(&root, IngestConfig::default()).unwrap();
    assert!(std::fs::read(&m).unwrap().ends_with(b"\n"));
    let b = history(&read, "r").await;
    assert_eq!(b.steps, vec![0, 1, 2, 3, 4, 5]);
    ingest.clone().oneshot(log("r", 7, 7.0)).await.unwrap();
    let rd = modgrad_obs::read_run(&root.join("runs/r")).unwrap();
    assert_eq!(rd.skipped, 0);
    assert_eq!(rd.series("x").last(), Some(&(7, 7.0)));
}

#[tokio::test]
async fn reader_never_sees_uncommitted_bytes() {
    let root = tmp("commit");
    let (ingest, read) = open(&root, IngestConfig::default()).unwrap();
    ingest.clone().oneshot(IngestRequest::CreateRun { name: "r".into(), project: None, config: json!({}), meta: None }).await.unwrap();
    ingest.clone().oneshot(log("r", 0, 0.0)).await.unwrap();
    // bytes beyond the committed length (a write in progress) are invisible
    let m = root.join("runs/r/metrics.jsonl");
    let mut f = std::fs::OpenOptions::new().append(true).open(&m).unwrap();
    std::io::Write::write_all(&mut f, b"{\"step\":1,\"t\":1,\"x\":1}\n{\"st").unwrap();
    let b = history(&read, "r").await;
    assert_eq!(b.steps, vec![0]);
}

fn store_bytes(root: &Path, runs: usize) -> Vec<Vec<u8>> {
    (0..runs).map(|r| std::fs::read(root.join(format!("runs/r{r}/metrics.jsonl"))).unwrap()).collect()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]
    /// Interleaved concurrent ingests (one producer per run, random yields)
    /// produce byte-identical per-run files to sequential ingest, with each
    /// run's records in step order.
    #[test]
    fn concurrent_equals_sequential(steps in proptest::collection::vec(1usize..15, 1..4), yields in proptest::collection::vec(0u8..4, 64), max_items in 1usize..8) {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap();
        let nruns = steps.len();
        let cfg = IngestConfig { batch: BatchConfig { max_items, max_latency: Duration::from_millis(2), queue: 64 }, ..IngestConfig::default() };
        let (seq, con) = rt.block_on(async {
            let a = tmp("seq");
            let (ingest, _r) = open(&a, cfg).unwrap();
            for r in 0..nruns {
                ingest.clone().oneshot(IngestRequest::CreateRun { name: format!("r{r}"), project: None, config: json!({}), meta: None }).await.unwrap();
            }
            for (r, &n) in steps.iter().enumerate() {
                for s in 0..n as u64 {
                    ingest.clone().oneshot(log(&format!("r{r}"), s, (r * 100) as f64 + s as f64)).await.unwrap();
                }
            }
            let b = tmp("con");
            let (ingest, _r) = open(&b, cfg).unwrap();
            for r in 0..nruns {
                ingest.clone().oneshot(IngestRequest::CreateRun { name: format!("r{r}"), project: None, config: json!({}), meta: None }).await.unwrap();
            }
            let mut hs = Vec::new();
            for (r, &n) in steps.iter().enumerate() {
                let ingest = ingest.clone();
                let ys = yields.clone();
                hs.push(tokio::spawn(async move {
                    for s in 0..n as u64 {
                        for _ in 0..ys[(r * 17 + s as usize) % ys.len()] {
                            tokio::task::yield_now().await;
                        }
                        ingest.clone().oneshot(log(&format!("r{r}"), s, (r * 100) as f64 + s as f64)).await.unwrap();
                    }
                }));
            }
            for h in hs {
                h.await.unwrap();
            }
            (store_bytes(&a, nruns), store_bytes(&b, nruns))
        });
        prop_assert_eq!(&seq, &con);
        for (r, bytes) in con.iter().enumerate() {
            let steps_seen: Vec<u64> = String::from_utf8_lossy(bytes).lines().map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["step"].as_u64().unwrap()).collect();
            prop_assert_eq!(steps_seen, (0..steps[r] as u64).collect::<Vec<_>>());
        }
    }
}
