import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch
import requests
import os
import json
import time
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
sys.path.insert(0, '..')
from arc.coverage import wsc_unbiased
from conformal import *
from arc.methodsImg import SplitConformal, CVPlus, JackknifePlus

from torchcp.classification import SplitPredictor, SAPS

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[INFO] Using device:", device)
    print("[INFO] CUDA available:", torch.cuda.is_available())
    
    try:
        x = torch.rand(10000,10000, device=device)
        y = x @ x
        print(f"[DEBUG] Test matrix mean (should be float): {y.mean().item():.4f}")
    except Exception as e:
        print(f"[WARNING] Could not run CUDA matrix multiply: {e}")

    print("[INFO] ---- Stage 1: DATASET LOADING ----")
    USE_CIFAR10 = False
    BATCH_SIZE = 4
    N_CALIB = 2000  
    N_TEST = 1000   
    N_TUNE_CAL = 1000
    N_TUNE_EVAL = 1000

    if USE_CIFAR10:
        print("[INFO] Using CIFAR-10 as test dataset (for debugging)")
        transform = transforms.Compose([
            transforms.Resize((112, 112)),  # To match ResNet
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        indices = np.arange(len(dataset))
        labels = np.array([sample[1] for sample in dataset])
        rest_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=N_TEST,
        stratify=labels,
        random_state=42
        )
        calib_labels_rest = labels[rest_idx]
        calib_idx, test_idx = train_test_split(indices, test_size=N_TEST, random_state=42, stratify=labels)

        calib_idx = calib_idx[:N_CALIB]
        calib_dataset = Subset(dataset, calib_idx)
        test_dataset = Subset(dataset, test_idx)
        print(f"[DEBUG] Calib set size: {len(calib_dataset)} | Test set size: {len(test_dataset)}")
    else:
        IMAGENET_ROOT = os.path.normpath('./imagenet_val')
        print("[INFO] Using ImageNet from", IMAGENET_ROOT)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(str(IMAGENET_ROOT), transform=transform)
        print(dataset.class_to_idx)
        print(dataset[0], "hi")
        indices = np.arange(len(dataset))
        temp_idx, test_idx = train_test_split(indices, test_size=N_TEST, random_state=42)
        tune_idx, calib_idx = train_test_split(temp_idx, test_size=N_CALIB, random_state=42)
        tune_cal_idx, tune_eval_idx = train_test_split(tune_idx, test_size=N_TUNE_EVAL, random_state=42)
        tune_cal_idx = tune_cal_idx[:N_TUNE_CAL]
        tune_caldataset = Subset(dataset, tune_cal_idx)
        tune_eval_dataset = Subset(dataset, tune_eval_idx)
        calib_dataset = Subset(dataset, calib_idx)
        test_dataset = Subset(dataset, test_idx)
        print(f"Tune-cal: {len(tune_caldataset)}, Tune-eval: {len(tune_eval_dataset)}")
        print(f"Calib: {len(calib_dataset)}, Test: {len(test_dataset)}")
        print(f"[DEBUG] Calib set size: {len(calib_dataset)} | Test set size: {len(test_dataset)}")
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        model_classes = [line.strip() for line in response.text.strip().split('\n')]
        
        
        folder_to_idx = dataset.class_to_idx
        synset_list_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        model_synsets = list(folder_to_idx.keys())
        idx_remap = {v: i for i, (k, v) in enumerate(sorted(folder_to_idx.items(), key=lambda x: x[0]))}

    
    print("[INFO] Creating dataloaders...")
    calib_loader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    tune_cal_loader = DataLoader(tune_caldataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    tune_eval_loader = DataLoader(tune_eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    # -------- 2. MODEL --------
    print("\n[INFO] ---- Stage 2: MODEL LOADING ----")
    if USE_CIFAR10:
        num_classes = 10
    else:
        num_classes = 1000  # ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if not USE_CIFAR10 else None)
    model = model.to(device)
    model.eval()
    print(f"[DEBUG] Loaded ResNet50 (num_classes={num_classes})")

    # -------- 3. GET IMAGES & LABELS AS ARRAYS --------
    def get_images_labels(loader):
        images_list, labels_list = [], []
        for batch_idx, (x, y) in enumerate(loader):
            images_list.append(x)
            labels_list.append(y)
            if batch_idx == 0:
                print(f"[DEBUG] First batch x shape: {x.shape}, y shape: {y.shape}")
        images = torch.cat(images_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        print(f"[DEBUG] Total images shape: {images.shape}, dtype: {images.dtype}")
        print(f"[DEBUG] Total labels shape: {labels.shape}, dtype: {labels.dtype}")
        return images, labels

    print("\n[INFO] Collecting calibration and test images/labels...")
    calib_images, calib_labels = get_images_labels(calib_loader)
    tune_cal_images, tune_cal_labels = get_images_labels(tune_cal_loader)
    tune_eval_images, tune_eval_labels = get_images_labels(tune_eval_loader)
    test_images, test_labels = get_images_labels(test_loader)
    with torch.no_grad():
        sample_imgs = test_images[:2].to(device)
        logits = model(sample_imgs)
        print(f"[DEBUG] Model logits shape: {logits.shape}")
        print(f"[DEBUG] Top 5 preds for first sample: {torch.topk(logits[0], 5).indices.cpu().numpy()}")
    
    calib_images_np = calib_images.cpu().numpy()
    test_images_np  = test_images.cpu().numpy()
    calib_labels_np = calib_labels.cpu().numpy()
    test_labels_np  = test_labels.cpu().numpy()

    with torch.no_grad():
        print('tesing model predictions alignment-----------------------------')
        sample_logits = model(test_images[:5].to(device))
        predicted_classes = sample_logits.argmax(dim=1).cpu().numpy()
        print("Model predictions:", predicted_classes)
        print("Your test labels:", test_labels_np[:5])
    with torch.no_grad():
        print("testing model miscalibration ===================")
        correct = 0
        confidences = []
        for i in range(min(200, len(test_images))):
            logits = model(test_images[i:i+1].to(device))
            pred = logits.argmax().item()
            conf = torch.softmax(logits, dim=1).max().item()
            
            if pred == test_labels_np[i]:
                correct += 1
            confidences.append(conf)
        
        accuracy = correct / len(range(min(200, len(test_images))))
        avg_confidence = np.mean(confidences)
        print(f"Model accuracy: {accuracy:.3f}")
        print(f"Average model confidence: {avg_confidence:.3f}")
        print(f"Overconfidence gap: {avg_confidence - accuracy:.3f}")
    calib_labels_np = np.array([idx_remap[int(lbl)] for lbl in calib_labels_np])
    test_labels_np = np.array([idx_remap[int(lbl)] for lbl in test_labels_np])

    print(f"[DEBUG] Calibration labels min/max: {calib_labels_np.min()}, {calib_labels_np.max()}")
    print(f"[DEBUG] Test labels min/max: {test_labels_np.min()}, {test_labels_np.max()}")
    print(f"[DEBUG] Unique calib labels: {np.unique(calib_labels_np)[:20]} ...")
    print(f"[DEBUG] Unique test labels: {np.unique(test_labels_np)[:20]} ...")

    print(f"[DEBUG] Unique test labels: {np.unique(test_labels_np)[:20]}...")
    print(f"[DEBUG] Min label: {np.min(test_labels_np)}, Max label: {np.max(test_labels_np)}")

    # -------- 4. TORCHBLACKBOX FOR ARC --------
    print("\n[INFO] ---- Stage 3: BlackBox Wrapper ----")
    class TorchBlackBox:
        def __init__(self, model):
            self.model = model.eval()
            self.device = next(model.parameters()).device
        def fit(self, X, y):
            return self  # Already trained
        def predict_proba(self, X, batch_size=128):
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            if X.ndimension() == 3:
                X = X.unsqueeze(0)
            X = X.to(self.device)
            self.model.eval()
            probs = []
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    batch = X[i:i+batch_size]
                    logits = self.model(batch)
                    batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
                    probs.append(batch_probs)
            return np.vstack(probs)

    black_box = TorchBlackBox(model)
    print(f"[DEBUG] BlackBox device: {black_box.device}")
    test_probs = black_box.predict_proba(test_images_np)
    print(f"[DEBUG] BlacBox prob sum (should be close to 1): {test_probs.sum(axis=1)}")
    print("Top-1 confidence:", test_probs.max(axis=1))
    print(f"[DEBUG] BlackBox probs for sample 0 (top 5): {np.sort(test_probs[0])[-5:]}")

    # -------- 5. GET LOGITS FOR SAPS --------
    print("\n[INFO] Getting logits for SAPS/RAPS...")
    def get_logits_labels(model, loader, device):
        import psutil
        import os
        import gc
        
        model.eval()
        logits_list, labels_list = [], []
        
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            
            with torch.no_grad():  # Prevent gradient accumulation
                logits = model(x).cpu().detach()  # .detach() prevents graph retention
            
            logits_list.append(logits)
            labels_list.append(y.detach())  # detach labels too
            
            # Clear GPU memory immediately after each batch
            del x, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if batch_idx == 0:
                print(f"[DEBUG] SAPS logits first batch shape: {logits_list[0].shape}")
            
            # Memory monitoring every 10 batches
            if batch_idx % 10 == 0:
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                print(f"[DEBUG] Batch {batch_idx}, Memory: {memory_mb:.1f} MB")
                
                # Force garbage collection
                gc.collect()
        
        print(f"[DEBUG] Concatenating {len(logits_list)} batches...")
        result_logits = torch.cat(logits_list, dim=0)
        result_labels = torch.cat(labels_list, dim=0)
        
        # Clear intermediate lists
        del logits_list, labels_list
        gc.collect()
        
        return result_logits, result_labels

    calib_logits, calib_labels_saps = get_logits_labels(model, calib_loader, device)
    test_logits, test_labels_saps = get_logits_labels(model, test_loader, device)
    tune_eval_logits, tune_eval_labels_saps = get_logits_labels(model, tune_eval_loader, device)
    def remap(arr_torch):
        a = arr_torch.cpu().numpy()
        return np.array([idx_remap[int(lbl)] for lbl in a], dtype = np.int64)
    tune_eval_labels_saps = remap(tune_eval_labels_saps)
    print(f"[DEBUG] Calib logits shape: {calib_logits.shape} | Test logits shape: {test_logits.shape}")

    # -------- 6. EXPERIMENT LOOP --------
    alpha = 0.05 #----------------change here to get results on different alpha-----------------
    results = []

    def eval_prediction_sets(S_test, y_test):
        coverage = np.mean([y_test[i] in S_test[i] for i in range(len(y_test))])
        avg_size = np.mean([len(S_test[i]) for i in range(len(y_test))])
        size_cov = np.mean([len(S_test[i]) for i in range(len(y_test)) if y_test[i] in S_test[i]])
        return coverage, avg_size, size_cov

    print("\n[INFO] ---- Stage 4: ARC Methods ----")
    for name, Method in zip(
        ['Split', 'CV+'
         #'Jackknife+'
         ],
        [SplitConformal, CVPlus
         #JackknifePlus
         ]
    ):
        print("Label range in Y:", np.min(calib_labels_np), np.max(calib_labels_np))
        print("Unique labels (calib):", np.unique(calib_labels_np)[:20])
        print("Unique labels (test):", np.unique(test_labels_np)[:20])
        print("calib_labels_np shape:", calib_labels_np.shape)
        print("test_labels_np shape:", test_labels_np.shape)
        print(f"[INFO] Running ARC method: {name}")
        if(name == 'CV+'):
            method = Method(calib_images_np, calib_labels_np, black_box,alpha,n_folds=10,allow_empty=True)
        else:
            method = Method(calib_images_np, calib_labels_np, black_box, alpha, allow_empty=True)
        S_test = method.predict(test_images_np)
        for i in range(10):
            print(f"[DEBUG] Sample {i} | Set: {S_test[i]} | Label: {test_labels_np[i]} | Hit: {test_labels_np[i] in S_test[i]}")
        set_lens = [len(s) for s in S_test]
        print(f"[DEBUG] S_test for {name}: len={len(S_test)}, first set: {S_test[0]}")
        print(f"[DEBUG] {name} | Prediction set sizes (min/max/mean): {min(set_lens)}, {max(set_lens)}, {np.mean(set_lens):.2f}")
        print(f"[DEBUG] {name} | First 5 sets: {[s.tolist() for s in S_test[:5]]}")
        # Optionally: Check if true label is EVER in set
        print(f"[DEBUG] {name} first 10 test labels: {test_labels_np[:10]}")
        print(f"[DEBUG] {name} set coverage in first 100: {[test_labels_np[i] in S_test[i] for i in range(100)]}")
        coverage, avg_size, size_cov = eval_prediction_sets(S_test, test_labels_np)
        X_feat = np.array([len(s) for s in S_test]).reshape(-1,1)
        ws_cov = wsc_unbiased(X_feat, test_labels_np, S_test)
        print(f"[DEBUG] {name} | Coverage: {coverage:.3f} | Avg Size: {avg_size:.2f} | WSC: {ws_cov:.2f}")
        results.append([name, coverage, ws_cov, avg_size, size_cov, 1-alpha, len(test_labels_np)])

    print("\n[INFO] ---- Stage 5: RAPS ----")
    lamdas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    kregs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,60,80,100,200]
    best_score = -np.inf
    best_cfg = None
    print('tuning====================================')
    for lam in lamdas:
        for kreg in kregs:

            raps_tune = ConformalModel(
                model, tune_cal_loader, alpha=0.1, kreg = kreg, lamda = lam,
                randomized = False, allow_zero_sets=False
            )
            S_test_raps = []
            with torch.no_grad():
                for i, (x,_) in enumerate(tune_eval_loader):
                    x = x.to(device)
                    _, sets = raps_tune(x)
                    S_test_raps.extend(sets)
                    if i ==0:
                        print(f"[DEBUG] RAPS: first set: {sets[0]}")
            top1, top5, coverage_tune, avg_size_tune = validate(tune_eval_loader, raps_tune, print_bool = True)
            if coverage_tune >= 1-alpha - 0.02:
                score = -avg_size_tune
            else:
                score = -np.inf
            if score > best_score:
                best_score = score
                best_cfg = (lam,kreg)
            print(f"lam={lam:.3f}, kreg={kreg}, cov={coverage_tune:.3f}, size={avg_size_tune:.2f}, score={score:.2f}")
    print('tuning-done')
    print(f"lamda={best_cfg[0]:.3f}")
    if best_cfg is None:
        lam_best, kreg_best = 0.1, 1
    else:
        lam_best, kreg_best = best_cfg
    print(f"[RAPS] alpha={alpha} tuned lam={lam_best}, kreg={kreg_best} on TUNE (score={best_score:.4f})")
    raps_model = ConformalModel(model, calib_loader, alpha=alpha, kreg=kreg_best, lamda=lam_best, randomized=False, allow_zero_sets=False)
    S_test_raps = []
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            x = x.to(device)
            _, sets = raps_model(x)
            S_test_raps.extend(sets)
            if i == 0:
                print(f"[DEBUG] RAPS: first set: {sets[0]}")
    top1, top5, coverage_raps, avg_size_raps = validate(test_loader, raps_model, print_bool = True)
    size_cov_raps = 1
    X_feat_raps = np.array([len(s) for s in S_test_raps]).reshape(-1,1)
    ws_cov_raps = wsc_unbiased(X_feat_raps, test_labels_np, S_test_raps)
    print(f"[DEBUG] RAPS | Coverage: {coverage_raps:.3f} | Avg Size: {avg_size_raps:.2f} | WSC: {ws_cov_raps:.2f}")
    results.append(['RAPS', coverage_raps, ws_cov_raps, avg_size_raps, size_cov_raps, 1-alpha, len(test_labels_np)])

    print("\n[INFO] ---- Stage 6: SAPS ----")
    best_rw = None
    best_size_saps = np.inf
    print('tuning')
    for rw in [0.001, 0.01, 0.05, 0.1, 0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]:
        predictor_saps = SplitPredictor(SAPS("softmax", False, rw), model)
        predictor_saps.calibrate(tune_cal_loader, alpha)
        prediction_sets_saps = predictor_saps.predict_with_logits(tune_eval_logits)
        S_test_saps = [np.where(np.array(s) == 1)[0] for s in prediction_sets_saps]
        
        print(f"[DEBUG] SAPS: first set: {S_test_saps[0]}")
        coverage_saps_tune, avg_size_saps_tune, size_cov_saps_tune = eval_prediction_sets(S_test_saps, tune_eval_labels_saps)
        if coverage_saps_tune >= 1- alpha - 0.02:
            score = avg_size_saps_tune
        else:
            score = np.inf
        if score < best_size_saps :
            best_size_saps = avg_size_saps_tune
            best_rw = rw
        print(f"rw={rw:.3f}, coverage={coverage_saps_tune:.3f}, size={avg_size_saps_tune:.2f}")
    print('tuning done')
    #print(f"rw={best_rw:.3f}")
    print(f"[SAPS] alpha={alpha} tuned ranking_weight={best_rw} on TUNE (score={best_size_saps:.4f})")
    if best_rw is None:
        best_rw = 0.1

    sap_final = SAPS("softmax", False, best_rw)
    sp_cal = SplitPredictor(sap_final,model)
    sp_cal.calibrate(calib_loader, alpha)
    pred_sets_test = sp_cal.predict_with_logits(test_logits)
    S_testr_saps = [np.where(np.array(s) == 1)[0] for s in pred_sets_test]
    coverage_saps, avg_size_saps, size_cov_saps = eval_prediction_sets(S_testr_saps, test_labels_np)
    X_feat_saps = np.array([len(s) for s in S_testr_saps]).reshape(-1,1)
    ws_cov_saps = wsc_unbiased(X_feat_saps, test_labels_np, S_testr_saps)
    print(f"[DEBUG] SAPS | Coverage: {coverage_saps:.3f} | Avg Size: {avg_size_saps:.2f} | WSC: {ws_cov_saps:.2f}")
    results.append(['SAPS', coverage_saps, ws_cov_saps, avg_size_saps, size_cov_saps, 1-alpha, len(test_labels_np)])
    


    print("\n[INFO] ---- Stage 7: Results ----")
    df_results = pd.DataFrame(results, columns=['Method', 'Coverage', 'WSC', 'Length', 'Length cover', 'Nominal', 'n_test'])
    print(df_results)
    df_results.to_csv("conformal_results_ALL.csv", index = False)
    print("[INFO] Results saved to conformal_results_ALL.csv")


if __name__ == "__main__":
    main()