//! Golden outputs of `forecast::forecast` on the dashboard-shaped cases of
//! tests/common/forecast_cases.rs, recorded from the original (unoptimised)
//! implementation before the speed-up. The selected configuration, skill
//! badge, family bests and backtest horizon must match exactly; MAEs,
//! forecast values, band edges and the ETA readout to a relative 1e-6
//! (float reassociation in the Gram/rollout algebra, not a model change).
//! Regenerate only on a deliberate model change:
//!     cargo run --release -p modgrad-obs-analysis --example forecast_bench -- --golden

mod common;
use common::forecast_cases;

type Fam = Option<(usize, &'static str, bool, f64, f64)>;

#[allow(dead_code)]
struct G {
    name: &'static str,
    skill: bool,
    lookback: usize,
    scaler: &'static str,
    pooled: bool,
    position: bool,
    lambda: f64,
    noise: f64,
    n_configs: usize,
    mae_val: f64,
    mae_test: f64,
    mae_test_naive: f64,
    len: usize,
    eta: [f64; 3],
    single: Fam,
    pooled_best: Fam,
    max_backtest_h: usize,
    n_pool: usize,
    mean: &'static [(usize, f64)],
    lo: &'static [(usize, f64)],
    hi: &'static [(usize, f64)],
}

const REL: f64 = 1e-6;

fn close(what: &str, got: f64, want: f64) {
    if want.is_nan() {
        assert!(got.is_nan(), "{what}: got {got}, want NaN");
        return;
    }
    let tol = REL * want.abs().max(1e-3);
    assert!((got - want).abs() <= tol, "{what}: got {got:e}, want {want:e} (|d| = {:e} > {tol:e})", (got - want).abs());
}

fn fam(what: &str, got: Option<modgrad_obs_analysis::forecast::FamilyBest>, want: Fam) {
    match (got, want) {
        (None, None) => {}
        (Some(g), Some((l, s, p, v, t))) => {
            assert_eq!((g.config.lookback, g.config.scaler.name(), g.config.position), (l, s, p), "{what} config");
            close(&format!("{what} mae_val"), g.mae_val, v);
            close(&format!("{what} mae_test"), g.mae_test, t);
        }
        (g, w) => panic!("{what}: got {g:?}, want {w:?}"),
    }
}

const GOLDEN: &[G] = &[
    // single-1500
    G { name: "single-1500", skill: true, lookback: 8, scaler: "robust", pooled: false, position: true, lambda: 1e-3, noise: 2e-1, n_configs: 720, mae_val: 1.9991584423004714e-2, mae_test: 1.9302477524528884e-2, mae_test_naive: 2.1558138777444225e-2, len: 750, eta: [1.7179407808314914e4, 1.501e4, 1.758924886447321e4],
    single: Some((8, "robust", true, 1.9991584423004714e-2, 1.9302477524528884e-2)), pooled_best: None, max_backtest_h: 715, n_pool: 0,
    mean: &[(0, 2.318137586662518e0), (83, 2.3119460427419516e0), (166, 2.3044222426809124e0), (249, 2.297284452183356e0), (332, 2.290481064259221e0), (416, 2.283905396638718e0), (499, 2.2776877098157327e0), (582, 2.271725134016854e0), (665, 2.2659975555968495e0), (749, 2.260422016792973e0)],
    lo: &[(0, 2.2929912787250335e0), (83, 2.285296369096148e0), (166, 2.270566903414284e0), (249, 2.2628826933691357e0), (332, 2.2635276386739e0), (416, 2.283905396638718e0), (499, 2.273451323790568e0), (582, 2.2436311745794293e0), (665, 2.2447777917189393e0), (749, 2.260422016792973e0)],
    hi: &[(0, 2.370177375058336e0), (83, 2.3509697166574006e0), (166, 2.3239059407310902e0), (249, 2.3080118518317714e0), (332, 2.314755127480906e0), (416, 2.322532382747321e0), (499, 2.3236123516989298e0), (582, 2.31957191255983e0), (665, 2.3015297461281525e0), (749, 2.3098864703118815e0)] },
    // pooled-1500
    G { name: "pooled-1500", skill: true, lookback: 8, scaler: "robust", pooled: false, position: true, lambda: 1e-3, noise: 2e-1, n_configs: 1620, mae_val: 2.03462781980419e-2, mae_test: 1.89193244667179e-2, mae_test_naive: 1.9474571875437656e-2, len: 750, eta: [1.731856611932669e4, 1.501e4, 1.757905343175367e4],
    single: Some((8, "robust", true, 2.03462781980419e-2, 1.89193244667179e-2)), pooled_best: Some((32, "global", true, 2.211058569627414e-2, 2.015848540282375e-2)), max_backtest_h: 715, n_pool: 3,
    mean: &[(0, 2.313946269760239e0), (83, 2.312781166332581e0), (166, 2.305500668894006e0), (249, 2.2984981459967595e0), (332, 2.291822406028768e0), (416, 2.2853702155613522e0), (499, 2.279269389712403e0), (582, 2.2734189728566934e0), (665, 2.267799214919028e0), (749, 2.2623287085174475e0)],
    lo: &[(0, 2.289385300458902e0), (83, 2.2931290729582665e0), (166, 2.286153200392061e0), (249, 2.284775393202316e0), (332, 2.2776732783689604e0), (416, 2.2853702155613522e0), (499, 2.258877563694877e0), (582, 2.262683036423946e0), (665, 2.267799214919028e0), (749, 2.2623287085174475e0)],
    hi: &[(0, 2.3250111767444084e0), (83, 2.3351061866168457e0), (166, 2.332571810736021e0), (249, 2.324050285925505e0), (332, 2.3290219835613675e0), (416, 2.2994100943364693e0), (499, 2.297651015098065e0), (582, 2.330180749459359e0), (665, 2.3054772252732927e0), (749, 2.2708283667598854e0)] },
    // single-300
    G { name: "single-300", skill: true, lookback: 16, scaler: "robust", pooled: false, position: true, lambda: 1e-4, noise: 2e-1, n_configs: 720, mae_val: 2.4777688755804533e-2, mae_test: 2.6192548805038574e-2, mae_test_naive: 4.1998287895708e-2, len: 150, eta: [1.868515791099214e4, 1.5530608358562195e4, 1.9469824908437997e4],
    single: Some((16, "robust", true, 2.4777688755804533e-2, 2.6192548805038574e-2)), pooled_best: None, max_backtest_h: 143, n_pool: 0,
    mean: &[(0, 2.565860241804941e0), (16, 2.546241230839446e0), (33, 2.529951621846249e0), (49, 2.517111460978487e0), (66, 2.504677754976264e0), (82, 2.4936792925534323e0), (99, 2.4825589768139884e0), (115, 2.4725469837568803e0), (132, 2.4623409369004845e0), (149, 2.452541259440015e0)],
    lo: &[(0, 2.529606830655193e0), (16, 2.523836429285171e0), (33, 2.5087506325180824e0), (49, 2.5005707342972805e0), (66, 2.4722556309270276e0), (82, 2.470368779901055e0), (99, 2.4352073111534147e0), (115, 2.4494470380066122e0), (132, 2.459764023133898e0), (149, 2.429550058092062e0)],
    hi: &[(0, 2.5857332298309625e0), (16, 2.584199027188758e0), (33, 2.566564095641059e0), (49, 2.531300057190771e0), (66, 2.561369041265701e0), (82, 2.5225474174543496e0), (99, 2.5328289434971727e0), (115, 2.5246731976727452e0), (132, 2.4979187094472093e0), (149, 2.507830541330876e0)] },
    // pooled-eval-9
    G { name: "pooled-eval-9", skill: true, lookback: 2, scaler: "global", pooled: true, position: true, lambda: 1e-3, noise: 0e0, n_configs: 420, mae_val: 4.4908154930338284e-2, mae_test: 8.891993519082024e-2, mae_test_naive: 1.4677715210807799e-1, len: 5, eta: [6.105311780154323e3, 6.105311780154323e3, f64::NAN],
    single: Some((1, "standard", true, 3.146353246644811e-1, 5.506287085078062e-2)), pooled_best: Some((2, "global", true, 4.4908154930338284e-2, 8.891993519082024e-2)), max_backtest_h: 4, n_pool: 3,
    mean: &[(0, 4.171562976294339e0), (1, 4.036728689606617e0), (2, 3.920761266538217e0), (3, 3.822190781319025e0), (4, 3.738310650258649e0)],
    lo: &[(0, 4.1269345488639955e0), (1, 3.9788362529761754e0), (2, 3.920761266538217e0), (3, 3.822190781319025e0), (4, 3.738310650258649e0)],
    hi: &[(0, 4.279565356648176e0), (1, 4.143422743428512e0), (2, 4.020732462240073e0), (3, 3.971620281257589e0), (4, 3.9053779101118637e0)] },
    // single-eval-12
    G { name: "single-eval-12", skill: true, lookback: 1, scaler: "standard", pooled: false, position: true, lambda: 1e-4, noise: 2e-1, n_configs: 240, mae_val: 3.800926997027159e-2, mae_test: 3.417910876612584e-2, mae_test_naive: 1.522916973542047e-1, len: 6, eta: [8.30534254935944e3, 8.068826705425441e3, f64::NAN],
    single: Some((1, "standard", true, 3.800926997027159e-2, 3.417910876612584e-2)), pooled_best: None, max_backtest_h: 6, n_pool: 0,
    mean: &[(0, 3.9639325952098634e0), (1, 3.8835481027137218e0), (2, 3.809141689951433e0), (3, 3.7397379831200337e0), (4, 3.674666830731679e0), (5, 3.613407937416312e0)],
    lo: &[(0, 3.9269123554458005e0), (1, 3.8344816353857785e0), (2, 3.789683437465253e0), (3, 3.7089345640054288e0), (4, 3.644028333566234e0), (5, 3.613407937416312e0)],
    hi: &[(0, 4.0114902583085685e0), (1, 3.9332784251829134e0), (2, 3.865261776640791e0), (3, 3.8126495228840467e0), (4, 3.777399936664221e0), (5, 3.7046819864248457e0)] },
];

#[test]
fn forecast_matches_golden_outputs() {
    let cases = forecast_cases::cases();
    assert_eq!(cases.len(), GOLDEN.len());
    for (c, g) in cases.iter().zip(GOLDEN) {
        assert_eq!(c.name, g.name);
        let f = forecast_cases::run(c).unwrap_or_else(|| panic!("{}: no forecast", g.name));
        let n = g.name;
        let cfg = f.config;
        assert_eq!(f.skill, g.skill, "{n} skill");
        assert_eq!((cfg.lookback, cfg.scaler.name(), cfg.pooled, cfg.position), (g.lookback, g.scaler, g.pooled, g.position), "{n} config");
        assert_eq!((cfg.lambda, cfg.noise), (g.lambda, g.noise), "{n} lambda/noise");
        assert_eq!(f.n_configs, g.n_configs, "{n} n_configs");
        assert_eq!((f.mean.len(), f.max_backtest_h, f.n_pool), (g.len, g.max_backtest_h, g.n_pool), "{n} sizes");
        close(&format!("{n} mae_val"), f.mae_val, g.mae_val);
        close(&format!("{n} mae_test"), f.mae_test, g.mae_test);
        close(&format!("{n} mae_test_naive"), f.mae_test_naive, g.mae_test_naive);
        fam(&format!("{n} single"), f.single, g.single);
        fam(&format!("{n} pooled"), f.pooled, g.pooled_best);
        for (what, got, want) in [("mean", &f.mean, g.mean), ("lo", &f.lo, g.lo), ("hi", &f.hi, g.hi)] {
            for &(i, v) in want {
                close(&format!("{n} {what}[{i}]"), got[i], v);
            }
        }
        let eta = modgrad_obs_analysis::curve::eta(&f.steps, &f.mean, &f.lo, &f.hi, c.target);
        for k in 0..3 {
            close(&format!("{n} eta[{k}]"), eta[k], g.eta[k]);
        }
    }
}
