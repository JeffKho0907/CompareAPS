import numpy as np
import time
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '..')
import torch.nn.functional as F
from arc.methods import SplitConformal, CVPlus, JackknifePlus
from arc.coverage import wsc_unbiased
from conformal import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import pandas as pd
from torchcp.classification import SplitPredictor
from torchcp.classification import SAPS
from sklearn.datasets import fetch_openml
import inspect
from sklearn.datasets  import fetch_openml
import torchcp



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==== 1. Synthetic Data ====
X_df, y_series = fetch_openml(data_id=42396, as_frame=True, return_X_y=True)
X = X_df.to_numpy(dtype = np.float64)
print("Shape of X:", X.shape)
y, classes = pd.factorize(y_series, sort=True)
K = len(classes)
p = X.shape[1]
print("Max label in y:", np.max(y))
print("Unique labels:", np.unique(y))
sigma = SAPS("softmax",False,0.2)
beta = SplitPredictor(sigma)
print(inspect.signature(SplitPredictor))

rng_seed = 42
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.55, random_state=rng_seed, stratify=y)
X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size = 0.1818181818, random_state=rng_seed, stratify=y_temp)
X_tune, X_calibtune, y_tune, y_calibtune = train_test_split(X_calib, y_calib, test_size = 0.777777778, random_state=rng_seed, stratify=y_calib) #specifically for SAPS
X_tune_cal, X_tune_eval, y_tune_cal, y_tune_eval = train_test_split(X_tune, y_tune, test_size = 0.5, random_state=rng_seed, stratify=y_tune)
X_trainARC = np.concatenate([X_train, X_calib], axis = 0)
y_trainARC = np.concatenate([y_train,y_calib], axis = 0)


print(f"shapes -> train: {X_train.shape},calib:{X_calib.shape}, calibtune:{X_calibtune.shape},tune: {X_tune.shape}, tunecal:{X_tune_cal.shape}, tuneeval:{X_tune_eval.shape} , test:{X_test.shape}, XtrainARC:{X_trainARC.shape}")
# ==== 2. MLP Neural Network Model ====
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

model = DeepMLP(input_dim=p, num_classes =K).to(device)

# ==== 3. Mini-batch Training ====
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
train_dataset = tdata.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = tdata.DataLoader(train_dataset, batch_size=128, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(50):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

model.eval()
# ==== 4. APS Black-Box Wrapper (still works with MLP) ====
class TorchBlackBox:
    def __init__(self, model_ctor, device, train_cfg):
        """
        model_ctor: callable () -> uninitialized model (same arch)
        train_cfg: dict with epochs, batch_size, optimizer_ctor, etc.
        """
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
        for _ in range(self.train_cfg.get("epochs", 20)):
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
        return fitted  # a NEW black box trained on (X,y)

    @torch.no_grad()
    def predict_proba(self, X):
        logits = self.model(torch.tensor(X, dtype=torch.float32, device=self.device))
        return torch.softmax(logits, dim=1).cpu().numpy()


black_box = TorchBlackBox(model_ctor=lambda: DeepMLP(input_dim=p, num_classes=K),
    device=device,
    train_cfg={"epochs": 20, "batch_size": 256, "optimizer_ctor": torch.optim.Adam})
probs = black_box.predict_proba(X_test)
print("Average Top-1 predicted probability:", np.max(probs, axis =1).mean())
print("Sample softmax probs:")
for i in range(5):
    print(np.round(probs[i], 3), "Top-1 prob:", np.max(probs[i]))

# ==== local conditional coverage ====
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

@torch.no_grad()
def local_conditional_coverage(
    X, y, S, *,
    scaler=None,
    n_ref=5000,        # size of reference subset
    k=50,              # neighbors used for smoothing
    bandwidth=None,
    random_state=0,
    batch_size=8192,   
    device=None
):
    """
    Local coverage via kNN kernel smoothing against a reference subset.
    All kNN + kernel math runs on GPU (if available). Memory O(n_ref * batch_size).

    Returns (local_cov: np.ndarray [n,], bandwidth: float).
    """
    # ---------- prep ----------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(y)
    # covered[i] = 1 if true label in set S[i], else 0
    covered = np.fromiter((float(y[i] in S[i]) for i in range(n)), dtype=np.float32, count=n)

    # scale once on CPU
    if scaler is None:
        scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    # choose reference subset
    rng = np.random.default_rng(random_state)
    n_ref = min(n_ref, n)
    ref_idx = rng.choice(n, size=n_ref, replace=False)
    X_ref = Xs[ref_idx]
    covered_ref = covered[ref_idx]

    # torch tensors on device
    X_ref_t = torch.from_numpy(X_ref).to(device)                  # [n_ref, p]
    covered_ref_t = torch.from_numpy(covered_ref).to(device)      # [n_ref]
    Xs_t = torch.from_numpy(Xs).to(device)                        # [n, p]

    k = int(min(k, n_ref))

    # Precompute squared norms of reference to speed up distance calc
    ref_sq = (X_ref_t ** 2).sum(dim=1)    # [n_ref]

    # storage for kNN results
    all_k_dists = torch.empty((n, k), dtype=torch.float32, device=device)
    all_k_idx   = torch.empty((n, k), dtype=torch.int64,   device=device)

    # ---------- batched kNN to reference set ----------
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        Xb = Xs_t[start:end]                        # [b, p]
        x_sq = (Xb ** 2).sum(dim=1, keepdim=True)   # [b, 1]
        d2 = x_sq + ref_sq.unsqueeze(0) - 2.0 * (Xb @ X_ref_t.T)   # [b, n_ref]
        # topk smallest distances
        dists, idxs = torch.topk(d2, k=k, dim=1, largest=False, sorted=True)
        all_k_dists[start:end] = torch.sqrt(torch.clamp(dists, min=0.0))  # [b, k] Euclidean
        all_k_idx[start:end]   = idxs
        start = end

    # ---------- bandwidth (if not given): median k-th neighbor distance ----------
    if (bandwidth is None) or not np.isfinite(bandwidth) or (bandwidth <= 0):
        kth = all_k_dists[:, -1]
        kth_nonzero = kth[kth > 0]
        fallback = kth_nonzero.median().item() if kth_nonzero.numel() > 0 else 1.0
        kth = torch.where(kth > 0, kth, torch.tensor(fallback, device=device))
        bandwidth = torch.median(kth).item()
        if (not np.isfinite(bandwidth)) or (bandwidth <= 0):
            bandwidth = 1.0

    # ---------- kernel weights + local smoothing ----------
    W = torch.exp(-(all_k_dists ** 2) / (2.0 * (bandwidth ** 2)))    # [n, k]
    denom = W.sum(dim=1, keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))

    neigh_cov = covered_ref_t[all_k_idx]                              # [n, k]
    local_cov = (W * neigh_cov).sum(dim=1) / denom.squeeze(1)         # [n]

    return local_cov.detach().cpu().numpy(), float(bandwidth)
def softmax_np(x): return torch.softmax(x, dim=1).cpu().numpy()

# ==== 5. Run APS Methods ====
methods = {
    'Split': SplitConformal,
    'CV+': CVPlus
    #'Jackknife+': JackknifePlus
}
def to_indices(s):
    import numpy as np, torch
    # -> numpy 1D array
    if torch.is_tensor(s):
        arr = s.detach().cpu().numpy()
    else:
        arr = np.asarray(s)

    # boolean mask
    if arr.dtype == np.bool_:
        return np.flatnonzero(arr)

    # integer mask (0/1) -> treat as mask
    if np.issubdtype(arr.dtype, np.integer):
        if arr.ndim == 1 and np.all((arr == 0) | (arr == 1)):
            return np.flatnonzero(arr)
        return arr.astype(int)

    return arr.astype(int)


alphas = [0.1] #----------------add values here to get results on different alpha-----------------

for alpha in alphas: 
    results = []
    for name, Method in methods.items():
        method = Method(X_calib, y_calib, black_box, alpha)
        S_test = method.predict(X_test)
        print('size and marginal for ' + name  +'--------------------------')
        coverage = np.mean([y_test[i] in S_test[i] for i in range(len(y_test))])
        avg_size = np.mean([len(S_test[i]) for i in range(len(y_test))])
        size_cov = np.mean([len(S_test[i]) for i in range(len(y_test)) if y_test[i] in S_test[i]])
        print(coverage)
        print(avg_size)
        print("Computing worst-slice coverage for APS...")
        start_time = time.time()
        ws_cov = wsc_unbiased(X_test, y_test, S_test, M= 500)
        print(f"Split WSC took {time.time() - start_time:.1f} seconds")
        print("Computing local conditional coverage for Split...")
        start_time = time.time()
        local_cov, _ = local_conditional_coverage(X_test, y_test, S_test, n_ref=5000,k=50,batch_size=8192)
        print(f"Split Local CC took {time.time() - start_time:.1f} seconds")
        results.append([name, coverage, ws_cov, avg_size, size_cov, 1-alpha, len(y_test), np.quantile(local_cov, 0.05)])

    # ==== 6. Run RAPS Method (with DataLoader for calibration) ====
    print('RAPS------------------------------------')
    tune_dataset = tdata.TensorDataset(torch.tensor(X_tune_cal, dtype=torch.float32, device=device),
                                       torch.tensor(y_tune_cal,dtype=torch.long, device=device))
    tune_loader = tdata.DataLoader(tune_dataset, batch_size=128, shuffle=False)

    calib_dataset = tdata.TensorDataset(torch.tensor(X_calib, dtype=torch.float32, device=device),
                                    torch.tensor(y_calib, dtype=torch.long, device=device))
    calib_loader = tdata.DataLoader(calib_dataset, batch_size=128, shuffle=False)

    test_dataset =tdata.TensorDataset(torch.tensor(X_test, dtype=torch.float32, device=device),
                                    torch.tensor(y_test, dtype=torch.long, device=device))
    test_loader = tdata.DataLoader(test_dataset, batch_size=128, shuffle=False)


    best_score = -np.inf
    best_cfg = None
    lamdas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5,0.6,0.7,0.8,0.9,1]
    kregs = [1,2,3,4,5,6,7,8,9,10]
    best_ws_cov = 0.0
    print('tuning-----------------------------')
    for lam in lamdas:
        for kreg in kregs:

            raps_tune = ConformalModel(
                model, tune_loader, alpha=alpha,     # Fixed alpha, like 0.1
                kreg=kreg, lamda=lam,
                randomized=False, allow_zero_sets=False
            )
            with torch.no_grad():
                tune_sets = raps_tune(torch.tensor(X_tune_eval, dtype=torch.float32, device=device))[1]
            for i in range(3):
                print(f"Raw RAPS set {i}:", tune_sets[i])
                print(f"Type: {type(tune_sets[i])}")
                
                # Handle both numpy arrays and tensors
                if isinstance(tune_sets[i], torch.Tensor):
                    raps_indices = tune_sets[i].cpu().numpy()
                else:
                    # It's a numpy array
                    raps_indices = tune_sets[i]
                
            print(f"After where: {raps_indices}")
            print(f"True label: {y_tune_eval[i]}")
            top1, top5, coverage_tune, avg_size_tune = validate(tune_loader, raps_tune, print_bool=True)

            if coverage_tune >= 1 - alpha:
                score = -avg_size_tune   # smaller sets = better
            else:
                score = -np.inf

            if score > best_score:
                best_score = score
                best_cfg = (lam, kreg)
            print(f"lam={lam:.3f}, kreg={kreg}, cov={coverage_tune:.3f}, size={avg_size_tune:.2f}, score={score:.2f}")
    
    print('tuning-done')
    print(f"lamda={best_cfg[0]:.3f}")
    if best_cfg is None:
        lam_best, kreg_best = 0.1, 1   # some safe defaults
    else:
        lam_best, kreg_best = best_cfg
    print(f"[RAPS] alpha={alpha} tuned lam={lam_best}, kreg={kreg_best} on TUNE (score={best_score:.4f})")
    
    raps_cal = ConformalModel(model, calib_loader, alpha = alpha,
                              kreg = kreg_best, lamda = lam_best,
                               randomized=False, allow_zero_sets=False )
    
    with torch.no_grad():
        _, S_test_raps = raps_cal(torch.tensor(X_test,dtype=torch.float32, device=device))
    print('size and marginal for RAPS--------------------------')
    top1, top5, coverage_raps, avg_size_raps = validate(test_loader, raps_cal, print_bool=True)
    size_cov_raps   = np.mean([len(S_test_raps[i]) for i in range(len(y_test)) if y_test[i] in S_test_raps[i]])
    print("Computing worst-slice coverage for RAPS...")
    start_time = time.time()
    ws_cov_raps     = wsc_unbiased(X_test, y_test, S_test_raps, M=500)
    print(f"RAPS WSC took {time.time() - start_time:.1f} seconds")
    print("Computing local conditional coverage for RAPS...")
    start_time = time.time()
    local_cov_raps, _ = local_conditional_coverage(X_test, y_test, S_test_raps,n_ref=5000,k=50,batch_size=8192)
    print(f"RAPS Local CC took {time.time() - start_time:.1f} seconds")
    print(f"[RAPS] alpha={alpha}: cov={coverage_raps:.4f}, size={avg_size_raps:.2f}, WSC={ws_cov_raps:.4f}, LC5={np.quantile(local_cov_raps,0.05):.4f}")
    results.append(["RAPS", coverage_raps, ws_cov_raps, avg_size_raps, size_cov_raps, 1-alpha, len(y_test), np.quantile(local_cov_raps, 0.05)])
    # ==== 7. Run SAPS Method (with DataLoader for logits if you want) ====
    print('SAPS----------------------------------------')
    with torch.no_grad():
        tune_cal_logits  = model(torch.tensor(X_tune_cal,  dtype=torch.float32, device=device))
        tune_eval_logits = model(torch.tensor(X_tune_eval, dtype=torch.float32, device=device))
        calib_logits = model(torch.tensor(X_calib, dtype=torch.float32, device=device))
        test_logits = model(torch.tensor(X_test, dtype=torch.float32, device=device))
    tune_dataset = tdata.TensorDataset(torch.tensor(X_tune, dtype=torch.float32, device=device),
                                       torch.tensor(y_tune,dtype=torch.long, device=device))
    tune_loader = tdata.DataLoader(tune_dataset, batch_size=128, shuffle=False)
    calib_dataset = tdata.TensorDataset(torch.tensor(X_calib, dtype=torch.float32, device=device),
                                    torch.tensor(y_calib, dtype=torch.long, device=device))
    calib_loader = tdata.DataLoader(calib_dataset, batch_size=128, shuffle=False)
    
    best_rw = None
    best_size_saps = np.inf
    print('tuning---------------------------')
    for rw in [0.00001,1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2,0.001, 0.0025, 0.005,0.006,0.007,0.008,0.009,0.01, 0.015, 0.02, 0.025, 0.03, 0.035,0.04,0.045,0.05,0.055,0.06,0.07,0.08,0.09,0.1, 0.2, 0.3,0.4]:
        sap = SAPS("softmax", False, rw)
        sp_tune = SplitPredictor(sap,model)
        sp_tune.calibrate(tune_loader,alpha)
        pred_sets_tune = sp_tune.predict_with_logits(tune_eval_logits, 1.1)
        s_tune = [to_indices(s) for s in pred_sets_tune]
        for i in range(3):  # first 3 examples
            print("Raw SAPS set:", pred_sets_tune[i])
            print("After np.where:", s_tune[i])
            print("True label:", y_tune_eval[i])

        coverage_tune = np.mean([y_tune_eval[i] in s_tune[i] for i in range(len(y_tune_eval))])
        avg_size_tune = np.mean([len(S) for S in s_tune])
        if coverage_tune >= 1 - alpha:
                score = avg_size_tune   # smaller sets = better
        else:
            score = np.inf

        
        if score < best_size_saps:
            best_size_saps = avg_size_tune
            best_rw = rw
        print(f"rw={rw:.3f}, coverage={coverage_tune:.3f}, size={avg_size_tune:.2f}")
    print('tuning-done')
    print(f"[SAPS] alpha={alpha} tuned ranking_weight={best_rw} on TUNE (score={best_size_saps:.4f})")
    if best_rw is None:
        best_rw = 0.1

    sap_final = SAPS("softmax", False, best_rw)
    sp_cal = SplitPredictor(sap_final, model)
    sp_cal.calibrate(calib_loader, alpha)
    pred_sets_test = sp_cal.predict_with_logits(test_logits, 1.1)
    print('size and marginal for SAPS--------------------------')
    S_test_saps = [to_indices(s) for s in pred_sets_test]
    coverage_saps = np.mean([y_test[i] in S_test_saps[i] for i in range(len(y_test))])
    avg_size_saps = np.mean([len(S) for S in S_test_saps])
    size_cov_saps   = np.mean([len(S_test_saps[i]) for i in range(len(y_test)) if y_test[i] in S_test_saps[i]])
    print("Computing worst-slice coverage for SAPS...")
    start_time = time.time()
    ws_cov_saps     = wsc_unbiased(X_test, y_test, S_test_saps, M= 500)
    print(f"SAPS WSC took {time.time() - start_time:.1f} seconds")
    print("Computing local conditional coverage for SAPS...")
    start_time = time.time()
    local_cov_saps, _ = local_conditional_coverage(X_test, y_test, S_test_saps,n_ref=5000,k=50,batch_size=8192)
    print(f"SAPS Local CC took {time.time() - start_time:.1f} seconds")
    print(f"[SAPS] alpha={alpha}: cov={coverage_saps:.4f}, size={avg_size_saps:.2f}, WSC={ws_cov_saps:.4f}, LC5={np.quantile(local_cov_saps,0.05):.4f}")
    results.append(["SAPS", coverage_saps, ws_cov_saps, avg_size_saps, size_cov_saps, 1-alpha, len(y_test), np.quantile(local_cov_saps,0.05)])
    # ==== 8. Display Results ====
    df_results = pd.DataFrame(results, columns=['Method', 'Coverage', 'Conditional coverage', 'Length', 'Length cover', 'Nominal', 'n_test', 'local_conditional'])
    print(df_results)
   