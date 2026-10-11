//! Synthetic LLM-style runs for the dashboard's LLM views: a pretraining run
//! with per-layer grad norms and activation RMS (24 layers), a loss spike and
//! a NaN; an SFT run with evals and generations; an RL run with reward, kl,
//! entropy, response length and samples.
//!
//! ```sh
//! cargo run -p modgrad-obs --example llm_synthetic -- /tmp/obs-llm-data
//! obs-server --root /tmp/obs-llm-data --static obs/web/dist --dev
//! ```

use modgrad_obs::{Run, Sample};
use serde_json::json;
use std::path::PathBuf;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn n(&mut self) -> f64 {
        // sum of uniforms: roughly normal, sd 1
        (0..6).map(|_| self.f()).sum::<f64>() - 3.0
    }
    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[(self.f() * xs.len() as f64) as usize % xs.len()]
    }
}

fn pretrain(root: &std::path::Path, name: &str, seed: u64) {
    let mut r = Lcg(seed);
    let mut run = Run::init(root, "pt", &format!("pt/{name}"), &json!({"model": "gpt-24l", "layers": 24, "d_model": 1024, "lr": 3e-4, "seed": seed, "total_steps": 3000, "batch_tokens": 524288})).unwrap();
    let n = 3000u64;
    for s in 0..n {
        let x = s as f64;
        let warm = (x / 200.0).min(1.0);
        let lr = 3e-4 * warm * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * x / n as f64).cos()));
        let spike = if (1200..1206).contains(&s) { 1.8 * (-((x - 1200.0) / 2.0)).exp() } else { 0.0 };
        let loss = 2.4 + 8.0 * (x + 60.0).powf(-0.42) + 0.02 * r.n() + spike;
        let gn = 0.9 + 0.6 * (-x / 400.0).exp() + 0.05 * r.n() + if (1199..1203).contains(&s) { 9.0 } else { 0.0 };
        let mut row: Vec<(String, f64)> = vec![
            ("loss".into(), loss),
            ("lr".into(), lr),
            ("grad_norm".into(), if s == 2400 { f64::NAN } else { gn }),
            ("tokens_seen".into(), (s + 1) as f64 * 524288.0),
            ("tokens_per_s".into(), 182000.0 + 3000.0 * r.n()),
        ];
        if s % 10 == 0 {
            for l in 0..24u64 {
                let lf = l as f64;
                // deeper layers have larger grads early; layers 18-23 spike at 1200
                let base = (0.02 + 0.01 * lf) * (1.0 + 2.0 * (-x / 500.0).exp());
                let sp = if s == 1200 && l >= 18 { 25.0 * base } else if s == 2400 && l == 7 { 12.0 * base } else { 0.0 };
                row.push((format!("layers/{l}/grad_norm"), base * (1.0 + 0.06 * r.n()) + sp));
                // activation RMS grows with depth and slowly over training; layer 23 drifts up late
                let drift = if l == 23 && s > 2000 { (x - 2000.0) / 400.0 } else { 0.0 };
                row.push((format!("layers/{l}/act_rms"), (1.0 + 0.08 * lf) * (1.0 + 0.15 * (x / n as f64)) * (1.0 + 0.02 * r.n()) + drift));
            }
        }
        if s % 500 == 0 || s == n - 1 {
            let p = x / n as f64;
            row.push(("eval/hellaswag/acc".into(), 0.28 + 0.22 * p.sqrt() + 0.005 * r.n()));
            row.push(("eval/arc_easy/acc".into(), 0.35 + 0.25 * p.powf(0.6) + 0.006 * r.n()));
            row.push(("eval/lambada/acc".into(), 0.12 + 0.30 * p.powf(0.7) + 0.006 * r.n()));
            row.push(("eval/wikitext/ppl".into(), 60.0 * (-1.6 * p).exp() + 14.0 + 0.4 * r.n()));
            row.push(("val_loss".into(), loss + 0.05));
        }
        let items: Vec<(&str, modgrad_obs::Item)> = row.iter().map(|(k, v)| (k.as_str(), (*v).into())).collect();
        run.log(s, items).unwrap();
        if s == 1200 {
            run.event(s, "loss spike: skipped 2 batches").unwrap();
        }
    }
    run.event(2400, "grad_norm NaN; batch skipped by the grad scaler").unwrap();
    run.finish().unwrap();
}

const SFT_PROMPTS: [&str; 6] = [
    "Write a haiku about gradient descent.",
    "Explain what a learning rate warmup is in one sentence.",
    "Translate 'the model is still training' into French.",
    "What is 17 * 23?",
    "Summarise the plot of Hamlet in two sentences.",
    "Give three tips for debugging a loss spike.",
];

fn sft_answer(i: usize, quality: f64, r: &mut Lcg) -> String {
    let good: [&str; 6] = [
        "Small steps downhill,\nthe loss settles in the valley,\nmorning gradients.",
        "Warmup raises the learning rate gradually over the first steps so early updates do not destabilise training.",
        "Le modele est encore en train de s'entrainer.",
        "17 * 23 = 391.",
        "Prince Hamlet seeks revenge on his uncle Claudius, who murdered his father and married his mother. His hesitation leads to a chain of deaths, including his own.",
        "1. Check the data batch at the spike step. 2. Lower the learning rate or add warmup. 3. Clip gradients and watch per-layer grad norms.",
    ];
    let mid: [&str; 6] = [
        "Gradient goes down,\nthe loss is going down too,\nthe model learns things.",
        "Learning rate warmup is when the learning rate starts small and gets bigger.",
        "Le model est toujours training.",
        "17 * 23 = 381.",
        "Hamlet is a prince whose father dies. He wants revenge and many people die at the end.",
        "1. Look at the data. 2. Lower the learning rate. 3. Try again.",
    ];
    let bad: [&str; 6] = [
        "Gradient descent is an optimisation algorithm used in machine learning.",
        "A learning rate warmup is a learning rate warmup that warms up the learning rate.",
        "the model is still training",
        "17 * 23 is a multiplication problem.",
        "Hamlet is a play by Shakespeare.",
        "Debugging is important. You should debug the loss spike.",
    ];
    let u = r.f();
    if u < quality * 0.8 { good[i].into() } else if u < quality { mid[i].into() } else { bad[i].into() }
}

fn sft(root: &std::path::Path) {
    let mut r = Lcg(11);
    let mut run = Run::init(root, "sft", "sft/llama-8b", &json!({"base": "llama-8b", "lr": 2e-5, "epochs": 3, "total_steps": 1200})).unwrap();
    for s in 0..1200u64 {
        let x = s as f64;
        let loss = 0.85 + 0.9 * (-x / 250.0).exp() + 0.04 * r.n() - if s > 400 { 0.05 } else { 0.0 } - if s > 800 { 0.04 } else { 0.0 };
        let mut row: Vec<(String, f64)> = vec![("train/loss".into(), loss), ("lr".into(), 2e-5 * (x / 50.0).min(1.0) * (1.0 - x / 1300.0)), ("grad_norm".into(), 0.6 + 0.3 * (-x / 200.0).exp() + 0.04 * r.n())];
        if s % 200 == 0 || s == 1199 {
            let p = x / 1200.0;
            row.push(("eval/loss".into(), 0.95 + 0.7 * (-x / 300.0).exp()));
            row.push(("eval/mmlu/acc".into(), 0.58 + 0.06 * p.sqrt() + 0.004 * r.n()));
            row.push(("eval/gsm8k/acc".into(), 0.31 + 0.24 * p.powf(0.7) + 0.006 * r.n()));
            row.push(("eval/ifeval/strict_acc".into(), 0.42 + 0.28 * p.sqrt() + 0.008 * r.n()));
            let items: Vec<(&str, modgrad_obs::Item)> = row.iter().map(|(k, v)| (k.as_str(), (*v).into())).collect();
            run.log(s, items).unwrap();
            for (i, prompt) in SFT_PROMPTS.iter().enumerate() {
                let completion = sft_answer(i, 0.15 + 0.85 * p, &mut r);
                let score = if completion.starts_with("17 * 23 = 391") || completion.len() > 60 { 1.0 } else { 0.0 };
                run.log_sample(s, &Sample { key: "eval/chat".into(), prompt: (*prompt).into(), completion, score: Some(score), meta: Some(json!({"temperature": 0.0, "max_tokens": 256})), ..Default::default() }).unwrap();
            }
        } else {
            let items: Vec<(&str, modgrad_obs::Item)> = row.iter().map(|(k, v)| (k.as_str(), (*v).into())).collect();
            run.log(s, items).unwrap();
        }
    }
    run.finish().unwrap();
}

fn rl(root: &std::path::Path) {
    let mut r = Lcg(23);
    let mut run = Run::init(root, "rl", "rl/grpo-math", &json!({"algo": "grpo", "base": "sft/llama-8b", "lr": 1e-6, "kl_coef": 0.04, "group_size": 8, "total_steps": 600})).unwrap();
    let problems = [
        ("What is 12 + 35?", "47"),
        ("A train travels 60 km in 1.5 hours. What is its average speed in km/h?", "40"),
        ("Solve for x: 3x - 7 = 11.", "6"),
        ("How many prime numbers are less than 20?", "8"),
        ("What is 15% of 240?", "36"),
        ("If a rectangle is 7 by 9, what is its area?", "63"),
    ];
    let fillers = ["Let me think step by step.", "First, I will write down what we know.", "We can check the answer at the end.", "Rearranging the terms gives", "So the result is", "Double-checking the arithmetic,"];
    for s in 0..600u64 {
        let x = s as f64;
        let p = x / 600.0;
        let reward = 0.22 + 0.55 * (1.0 - (-x / 180.0).exp()) + 0.05 * r.n() - if (380..400).contains(&s) { 0.25 } else { 0.0 };
        let kl = 0.002 + 0.03 * p + 0.003 * r.n().abs() + if (380..395).contains(&s) { 0.08 } else { 0.0 };
        let entropy = 1.2 - 0.6 * p + 0.03 * r.n();
        let resp_len = 180.0 + 240.0 * p + 12.0 * r.n();
        let row = [("reward", reward), ("kl", kl), ("entropy", entropy), ("response_length", resp_len), ("grad_norm", 0.3 + 0.1 * r.n().abs() + if s == 382 { 6.0 } else { 0.0 }), ("lr", 1e-6)];
        run.log_scalars(s, &row).unwrap();
        if s % 25 == 0 {
            for (pi, (q, ans)) in problems.iter().enumerate() {
                for g in 0..2 {
                    let correct = r.f() < 0.2 + 0.7 * p;
                    let steps = 1 + (p * 4.0 + r.f() * 2.0) as usize;
                    let mut text = String::new();
                    for k in 0..steps {
                        if k > 0 {
                            text.push(' ');
                        }
                        text.push_str(r.pick(&fillers));
                    }
                    let final_ans = if correct { ans.to_string() } else { format!("{}", ans.parse::<i64>().unwrap() + 1 + (r.f() * 5.0) as i64) };
                    text.push_str(&format!(" The answer is {final_ans}."));
                    let rew = if correct { 1.0 } else { 0.0 } - 0.001 * text.len() as f64 / 10.0;
                    run.log_sample(s, &Sample { key: "rollout".into(), prompt: (*q).into(), completion: text, reward: Some(rew), score: Some(if correct { 1.0 } else { 0.0 }), meta: Some(json!({"group": g, "problem": pi})) }).unwrap();
                }
            }
        }
    }
    run.event(380, "reward collapse: kl spike, lr halved").unwrap();
}

fn main() {
    let root = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "/tmp/obs-llm-data".into()));
    let _ = std::fs::remove_dir_all(root.join("runs"));
    pretrain(&root, "gpt-24l", 1);
    pretrain(&root, "gpt-24l-s2", 2);
    sft(&root);
    rl(&root);
    println!("wrote runs under {}", root.join("runs").display());
}
