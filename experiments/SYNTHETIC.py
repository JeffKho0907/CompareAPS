import numpy as np
import argparse
import time
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '..')
from arc.models import Model_Ex2
import torch.nn.functional as F
from arc.methods import SplitConformal, CVPlus
from arc.coverage import wsc_unbiased
from conformal import *
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import pandas as pd
from torchcp.classification import SplitPredictor
from torchcp.classification import SAPS, RAPS

# -----------------------
# Logging helper
# -----------------------
def log(msg: str, verbose: bool):
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Base MLP
# -----------------------
class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.2)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        return self.output(x)

# -----------------------
# Black-box wrapper for APS
# -----------------------
class TorchBlackBox:
    def __init__(self, model_ctor, device, train_cfg):
        self.model_ctor = model_ctor
        self.device = device
        self.train_cfg = train_cfg
        self.model = model_ctor().to(device)

    def _train_from_scratch(self, X, y):
        model = self.model_ctor().to(self.device)
        ds = tdata.TensorDataset(torch.tensor(X, dtype=torch.float32, device=self.device),
                                 torch.tensor(y, dtype=torch.long, device=self.device))
        dl = tdata.DataLoader(ds, batch_size=self.train_cfg.get("batch_size",128), shuffle=True)
        opt = self.train_cfg.get("optimizer_ctor", torch.optim.Adam)(model.parameters(),
                                                                     **self.train_cfg.get("opt_kwargs", {}))
        loss = nn.CrossEntropyLoss()
        for _ in range(self.train_cfg.get("epochs", 50)):
            model.train()
            for xb, yb in dl:
                opt.zero_grad()
                out = model(xb)
                loss(out, yb).backward()
                opt.step()
        return model.eval()

    def fit(self, X, y):
        fitted = TorchBlackBox(self.model_ctor, self.device, self.train_cfg)
        fitted.model = self._train_from_scratch(X, y)
        return fitted

    @torch.no_grad()
    def predict_proba(self, X):
        logits = self.model(torch.tensor(X, dtype=torch.float32, device=self.device))
        return torch.softmax(logits, dim=1).cpu().numpy()

# -----------------------
# Local conditional coverage util
# -----------------------
@torch.no_grad()
def local_conditional_coverage(
    X, y, S, *,
    scaler=None, n_ref=5000, k=50, bandwidth=None,
    random_state=0, batch_size=8192, device=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(y)
    covered = np.fromiter((float(y[i] in S[i]) for i in range(n)), dtype=np.float32, count=n)
    if scaler is None:
        scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    rng = np.random.default_rng(random_state)
    n_ref = min(n_ref, n)
    ref_idx = rng.choice(n, size=n_ref, replace=False)
    X_ref = Xs[ref_idx]
    covered_ref = covered[ref_idx]
    X_ref_t = torch.from_numpy(X_ref).to(device)
    covered_ref_t = torch.from_numpy(covered_ref).to(device)
    Xs_t = torch.from_numpy(Xs).to(device)
    k = int(min(k, n_ref))
    ref_sq = (X_ref_t ** 2).sum(dim=1)
    all_k_dists = torch.empty((n, k), dtype=torch.float32, device=device)
    all_k_idx   = torch.empty((n, k), dtype=torch.int64,   device=device)
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        Xb = Xs_t[start:end]
        x_sq = (Xb ** 2).sum(dim=1, keepdim=True)
        d2 = x_sq + ref_sq.unsqueeze(0) - 2.0 * (Xb @ X_ref_t.T)
        dists, idxs = torch.topk(d2, k=k, dim=1, largest=False, sorted=True)
        all_k_dists[start:end] = torch.sqrt(torch.clamp(dists, min=0.0))
        all_k_idx[start:end]   = idxs
        start = end
    if (bandwidth is None) or not np.isfinite(bandwidth) or (bandwidth <= 0):
        kth = all_k_dists[:, -1]
        kth_nonzero = kth[kth > 0]
        fallback = kth_nonzero.median().item() if kth_nonzero.numel() > 0 else 1.0
        kth = torch.where(kth > 0, kth, torch.tensor(fallback, device=device))
        bandwidth = torch.median(kth).item()
        if (not np.isfinite(bandwidth)) or (bandwidth <= 0):
            bandwidth = 1.0
    W = torch.exp(-(all_k_dists ** 2) / (2.0 * (bandwidth ** 2)))
    denom = W.sum(dim=1, keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    neigh_cov = covered_ref_t[all_k_idx]
    local_cov = (W * neigh_cov).sum(dim=1) / denom.squeeze(1)
    return local_cov.detach().cpu().numpy(), float(bandwidth)

# -----------------------
# Single-K runner with logs
# -----------------------
def run_experiment(K: int, p: int = 200, rng_seed: int = 42, device=None, verbose: bool=False) -> pd.DataFrame:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ----
    log(f"Generating synthetic data with Model_Ex2 (K={K}, p={p}) ...", verbose)
    data_model = Model_Ex2(K, p, magnitude=0.2)
    X = data_model.sample_X(200000)
    y = data_model.sample_Y(X) % K
    log(f"Total samples: {X.shape[0]} | Unique classes: {len(np.unique(y))}", verbose)

    # Splits
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.55, random_state=rng_seed, stratify=y)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.1818181818, random_state=rng_seed, stratify=y_temp)
    X_tune, X_calibtune, y_tune, y_calibtune = train_test_split(X_calib, y_calib, test_size=0.777777778, random_state=rng_seed, stratify=y_calib)
    X_tune_cal, X_tune_eval, y_tune_cal, y_tune_eval = train_test_split(X_tune, y_tune, test_size=0.5, random_state=rng_seed, stratify=y_tune)

    log(f"Split sizes → Train:{len(X_train)}  Calib:{len(X_calib)}  TuneCal:{len(X_tune_cal)}  "
        f"TuneEval:{len(X_tune_eval)}  Test:{len(X_test)}", verbose)

    # ---- model ----
    model = DeepMLP(input_dim=p, num_classes=K).to(device)
    train_dataset = tdata.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32, device=device),
        torch.tensor(y_train, dtype=torch.long, device=device)
    )
    train_loader = tdata.DataLoader(train_dataset, batch_size=256, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    log("Training base MLP ...", verbose)
    t0 = time.time()
    for epoch in range(50):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        if verbose and (epoch+1) in {1, 5, 10, 20, 30, 40, 50}:
            log(f"Epoch {epoch+1}/50 done", True)
    model.eval()
    log(f"Finished base training in {time.time()-t0:.1f}s", verbose)

    # ---- black-box wrapper for APS ----
    black_box = TorchBlackBox(
        model_ctor=lambda: DeepMLP(input_dim=p, num_classes=K),
        device=device,
        train_cfg={"epochs": 50, "batch_size": 256, "optimizer_ctor": torch.optim.Adam}
    )

    # ---- methods ----
    methods = {
        'Split': SplitConformal,
        'CV+': CVPlus
    }
    alpha = 0.1 #----------------change here to get results on different alpha-----------------
    results = []

    # ---- APS (Split/CV+) ----
    for name, Method in methods.items():
        log(f"[APS/{name}] Calibrating on X_calib ...", verbose)
        t0 = time.time()
        method = Method(X_calib, y_calib, black_box, alpha)
        log(f"[APS/{name}] Calibration finished in {time.time()-t0:.2f}s", verbose)

        log(f"[APS/{name}] Predicting on X_test ...", verbose)
        t0 = time.time()
        S_test = method.predict(X_test)
        log(f"[APS/{name}] Prediction finished in {time.time()-t0:.2f}s", verbose)

        coverage = np.mean([y_test[i] in S_test[i] for i in range(len(y_test))])
        avg_size = np.mean([len(S_test[i]) for i in range(len(y_test))])
        size_cov = np.mean([len(S_test[i]) for i in range(len(y_test)) if y_test[i] in S_test[i]])
        log(f"[APS/{name}] cov={coverage:.4f}, size={avg_size:.2f}, size|cov={size_cov:.2f}", verbose)

        log(f"[APS/{name}] Computing Worst-Slice Coverage (M=500) ...", verbose)
        t0 = time.time()
        ws_cov = wsc_unbiased(X_test, y_test, S_test, M=500)
        log(f"[APS/{name}] WSC={ws_cov:.4f} (time {time.time()-t0:.1f}s)", verbose)

        log(f"[APS/{name}] Computing Local Conditional Coverage ...", verbose)
        t0 = time.time()
        local_cov, _ = local_conditional_coverage(
            X_test, y_test, S_test, n_ref=5000, k=50, batch_size=8192, device=device
        )
        lc5 = np.quantile(local_cov, 0.05)
        log(f"[APS/{name}] LC5={lc5:.4f} (time {time.time()-t0:.1f}s)", verbose)

        results.append([name, coverage, ws_cov, avg_size, size_cov, 1-alpha, len(y_test), lc5])

    # ---- RAPS ----
    log("[RAPS] Preparing loaders ...", verbose)
    tune_dataset = tdata.TensorDataset(
        torch.tensor(X_tune_cal, dtype=torch.float32, device=device),
        torch.tensor(y_tune_cal, dtype=torch.long, device=device)
    )
    tune_loader = tdata.DataLoader(tune_dataset, batch_size=128, shuffle=False)
    calib_dataset = tdata.TensorDataset(
        torch.tensor(X_calib, dtype=torch.float32, device=device),
        torch.tensor(y_calib, dtype=torch.long, device=device)
    )
    calib_loader = tdata.DataLoader(calib_dataset, batch_size=128, shuffle=False)
    test_dataset = tdata.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32, device=device),
        torch.tensor(y_test, dtype=torch.long, device=device)
    )
    test_loader = tdata.DataLoader(test_dataset, batch_size=128, shuffle=False)

    best_score = -np.inf
    best_cfg = None
    lamdas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    kregs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,60,80,100,200,250,300,350,400,450,500,600]

    log("[RAPS] Tuning (grid over λ x kreg) ...", verbose)
    t0 = time.time()
    for lam in lamdas:
        for kreg in kregs:
            raps_tune = ConformalModel(
                model, tune_loader, alpha=alpha,
                kreg=kreg, lamda=lam,
                randomized=True, allow_zero_sets=False
            )
            _, _, coverage_tune, avg_size_tune = validate(tune_loader, raps_tune, print_bool=False)
            score = -avg_size_tune if coverage_tune >= 1 - alpha - 0.03 else -np.inf
            if score > best_score:
                best_score = score
                best_cfg = (lam, kreg)
    log(f"[RAPS] Best: λ={best_cfg[0]:.3f}, kreg={best_cfg[1]} (score={best_score:.3f}) "
        f"in {time.time()-t0:.1f}s", verbose)

    log("[RAPS] Calibrating on X_calib and evaluating on X_test ...", verbose)
    raps_cal = ConformalModel(
        model, calib_loader, alpha=alpha,
        kreg=best_cfg[1], lamda=best_cfg[0],
        randomized=True, allow_zero_sets=False
    )
    with torch.no_grad():
        _, S_test_raps = raps_cal(torch.tensor(X_test, dtype=torch.float32, device=device))
    _, _, coverage_raps, avg_size_raps = validate(test_loader, raps_cal, print_bool=False)
    size_cov_raps = np.mean([len(S_test_raps[i]) for i in range(len(y_test)) if y_test[i] in S_test_raps[i]])
    ws_cov_raps = wsc_unbiased(X_test, y_test, S_test_raps, M=500)
    local_cov_raps, _ = local_conditional_coverage(X_test, y_test, S_test_raps, n_ref=5000, k=50, batch_size=8192, device=device)
    lc5_raps = np.quantile(local_cov_raps, 0.05)
    log(f"[RAPS] cov={coverage_raps:.4f}, size={avg_size_raps:.2f}, size|cov={size_cov_raps:.2f}, "
        f"WSC={ws_cov_raps:.4f}, LC5={lc5_raps:.4f}", verbose)
    results.append(["RAPS", coverage_raps, ws_cov_raps, avg_size_raps, size_cov_raps, 1-alpha, len(y_test), lc5_raps])

    # ---- SAPS ----
    log("[SAPS] Preparing logits and loaders ...", verbose)
    with torch.no_grad():
        tune_cal_logits  = model(torch.tensor(X_tune_cal,  dtype=torch.float32, device=device))
        tune_eval_logits = model(torch.tensor(X_tune_eval, dtype=torch.float32, device=device))
        calib_logits     = model(torch.tensor(X_calib,     dtype=torch.float32, device=device))
        test_logits      = model(torch.tensor(X_test,      dtype=torch.float32, device=device))

    best_rw = None
    best_size_saps = np.inf
    rws = [0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.07,0.08,0.09,0.1,0.2,0.3]

    log("[SAPS] Tuning ranking_weight ...", verbose)
    t0 = time.time()
    for rw in rws:
        sap = SAPS("softmax", True, rw)
        sp_tune = SplitPredictor(sap, model)
        sp_tune.calibrate(tune_loader, alpha)
        pred_sets_tune = sp_tune.predict_with_logits(tune_eval_logits)
        s_tune = [np.where(np.array(s.cpu(), dtype=int) == 1)[0] for s in pred_sets_tune]
        coverage_tune = np.mean([y_tune_eval[i] in s_tune[i] for i in range(len(y_tune_eval))])
        avg_size_tune = np.mean([len(s_tune[i]) for i in range(len(y_tune_eval))])
        score = avg_size_tune if coverage_tune >= 1 - alpha - 0.03 else np.inf
        if score < best_size_saps:
            best_size_saps = avg_size_tune
            best_rw = rw
    log(f"[SAPS] Best rw={best_rw:.3f} (tuned in {time.time()-t0:.1f}s)", verbose)

    log("[SAPS] Calibrating on X_calib and evaluating on X_test ...", verbose)
    sap_final = SAPS("softmax", True, best_rw)
    sp_cal = SplitPredictor(sap_final, model)
    sp_cal.calibrate(calib_loader, alpha)
    pred_sets_test = sp_cal.predict_with_logits(test_logits)
    S_test_saps = [np.where(np.array(s.cpu(), dtype=int) == 1)[0] for s in pred_sets_test]
    coverage_saps = np.mean([y_test[i] in S_test_saps[i] for i in range(len(y_test))])
    avg_size_saps = np.mean([len(S_test_saps[i]) for i in range(len(y_test))])
    size_cov_saps = np.mean([len(S_test_saps[i]) for i in range(len(y_test)) if y_test[i] in S_test_saps[i]])
    ws_cov_saps = wsc_unbiased(X_test, y_test, S_test_saps, M=500)
    local_cov_saps, _ = local_conditional_coverage(X_test, y_test, S_test_saps, n_ref=5000, k=50, batch_size=8192, device=device)
    lc5_saps = np.quantile(local_cov_saps, 0.05)
    log(f"[SAPS] cov={coverage_saps:.4f}, size={avg_size_saps:.2f}, size|cov={size_cov_saps:.2f}, "
        f"WSC={ws_cov_saps:.4f}, LC5={lc5_saps:.4f}", verbose)
    results.append(["SAPS", coverage_saps, ws_cov_saps, avg_size_saps, size_cov_saps, 1-alpha, len(y_test), lc5_saps])

    # ---- Collate results ----
    df_results = pd.DataFrame(results, columns=[
        'Method', 'Coverage', 'Conditional coverage', 'Length',
        'Length cover', 'Nominal', 'n_test', 'local_conditional'
    ])
    log("Run complete. Results table prepared.", verbose)
    return df_results

# -----------------------
# CLI and multi-K + pretty printing
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=None, help="Number of classes K (if omitted, runs K in {10,100,1000})")
    parser.add_argument("--p", type=int, default=200, help="Number of features p")
    parser.add_argument("--verbose", action="store_true", help="Print progress logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--decimals", type=int, default=6, help="Decimals to print")
    args = parser.parse_args()

    # Determinism
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Ks = [args.K] if args.K is not None else [10, 100, 1000]

    all_results = []
    for K in Ks:
        log(f"==== Running experiment for K={K}, p={args.p} ====", True)
        dfK = run_experiment(K=K, p=args.p, device=device, verbose=args.verbose)
        dfK.insert(0, "K", K)
        all_results.append(dfK)

    df_all = pd.concat(all_results, ignore_index=True)

    # Pretty, grouped print by K with separators
    cols = ["K", "Method", "Coverage", "Conditional coverage", "Length", "Length cover", "local_conditional"]
    method_order = ["CV+", "Split", "RAPS", "SAPS"]  
    dec = max(0, int(args.decimals))

    for Kval in sorted(df_all["K"].unique()):
        block = df_all[df_all["K"] == Kval].copy()
        block["Method"] = pd.Categorical(block["Method"], categories=method_order, ordered=True)
        block = block.sort_values("Method")
        sep = "=" * 30
        print(f"\n{sep}  K = {Kval}  {sep}")
        print(block[cols].to_string(index=False,float_format=lambda x: f"{x:.{dec}f}"))

    print("\n" + "=" * 72)

    # Optional summary table (kept)
    summary = (df_all.groupby(["K", "Method"])
                     [["Coverage", "Conditional coverage", "Length", "Length cover", "local_conditional"]]
                     .mean()
                     .reset_index())
    print("\n==== Summary (means) by K x Method ====")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.{dec}f}"))

    # Optional: save
    df_all.to_csv("all_results_by_K.csv", index=False)
    summary.to_csv("summary_by_K.csv", index=False)
