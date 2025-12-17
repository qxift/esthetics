
import os, sys, math
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# ===================== CONFIG  =====================
AESTH_DIR = "../Data/museums/aesthetic"      
UNA_DIR   = "../Data/museums/unaesthetic"    
OUT_DIR   = "../code/plots/museums_dino_clip_binary_strict"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_PER_CULTURE = None   # None = all
KNN_K = 10
TSNE_PERP = 30
RANDOM_SEED = 42
# =================================================================
# ---------------- cache directory ----------------
CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
# =================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
sns.set(style="whitegrid", context="talk")
print("Device:", device)

# ---------------- imports used for models ----------------
# We will load DINOv2 via torch.hub (facebookresearch/dinov2) and OpenCLIP from open_clip.
from torchvision import transforms

# ---------------- preprocessing ----------------

# Standard preprocessing used for DINOv2 / ViT-style models:
preprocess_torch = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---------------- load DINOv2 model (strict) ----------------
# This uses torch.hub to fetch the official DINOv2 implementation from GitHub.
# Requirements: internet access and PyTorch installed.
print("Loading DINOv2 (dinov2_vitb14) via torch.hub ... (this will download weights if needed)")
try:
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dino = dino.to(device).eval()
except Exception as e:
    raise SystemExit("Failed to load DINOv2 via torch.hub. Make sure you have internet and PyTorch. Error:\n" + str(e))

def embed_dino(path):
    """#explanations: produce a DINOv2 embedding (1D numpy vector) for a single image path."""
    img = Image.open(path).convert("RGB")
    x = preprocess_torch(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = dino(x)
    # dino may return tuple; ensure vector
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.cpu().numpy().reshape(-1)

# ---------------- load OpenCLIP (strict) ----------------
# We require open_clip_torch to be installed (pip install open-clip-torch).
try:
    import open_clip
except Exception as e:
    raise SystemExit("open_clip (open-clip-torch) not installed. Install with:\n\npip install open-clip-torch\n\nError: " + str(e))

print("Loading OpenCLIP ViT-B/32 (laion2b_s34b_b79k) ...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.to(device).eval()

def embed_clip(path):
    """#explanations: produce a CLIP embedding (1D numpy vector) for a single image path using OpenCLIP preprocess."""
    img = Image.open(path).convert("RGB")
    x = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        v = clip_model.encode_image(x)
    return v.cpu().numpy().reshape(-1)

# ---------------- helper to embed a folder and extract culture from filename ----------------
def embed_folder(folder, embed_fn, max_per_culture=None):
    """
    Walk `folder`, embed each image using `embed_fn(path)`.
    Culture is inferred as text before first underscore in filename (e.g., 'albanian_2.jpg' -> 'albanian').
    Returns: embeddings (N,D), cultures (list), filenames (list)
    """
    embs = []
    cultures = []
    files = []
    for fn in sorted(os.listdir(folder)):
        path = os.path.join(folder, fn)
        if os.path.isdir(path): continue
        culture = fn.split("_")[0]
        if max_per_culture is not None and sum(1 for c in cultures if c==culture) >= max_per_culture:
            continue
        try:
            vec = embed_fn(path)
        except Exception as e:
            print("Embed failed:", path, e)
            continue
        if vec is None:
            continue
        embs.append(vec)
        cultures.append(culture)
        files.append(fn)
    if len(embs)==0:
        return np.zeros((0,1)), [], []
    return np.vstack(embs), cultures, files

# ---------------- embed museums dataset with DINO (this is required) ----------------
AESTH_CACHE = os.path.join(CACHE_DIR, "aesthetic_dino.npz")

if os.path.exists(AESTH_CACHE):
    print("Loading cached aesthetic DINO embeddings")
    data = np.load(AESTH_CACHE, allow_pickle=True)
    aesth_emb = data["emb"]
    aesth_cultures = data["cult"].tolist()
    aesth_files = data["files"].tolist()
else:
    print("Embedding aesthetic images with DINO (this can take time)...")
    aesth_emb, aesth_cultures, aesth_files = embed_folder(
        AESTH_DIR, embed_dino, max_per_culture=MAX_PER_CULTURE
    )
    np.savez(
        AESTH_CACHE,
        emb=aesth_emb,
        cult=np.array(aesth_cultures, dtype=object),
        files=np.array(aesth_files, dtype=object)
    )

UNA_CACHE = os.path.join(CACHE_DIR, "unaesthetic_dino.npz")

if os.path.exists(UNA_CACHE):
    print("Loading cached unaesthetic DINO embeddings")
    data = np.load(UNA_CACHE, allow_pickle=True)
    una_emb = data["emb"]
    una_cultures = data["cult"].tolist()
    una_files = data["files"].tolist()
else:
    print("Embedding unaesthetic images with DINO (this can take time)...")
    una_emb, una_cultures, una_files = embed_folder(
        UNA_DIR, embed_dino, max_per_culture=MAX_PER_CULTURE
    )
    np.savez(
        UNA_CACHE,
        emb=una_emb,
        cult=np.array(una_cultures, dtype=object),
        files=np.array(una_files, dtype=object)
    )

# Combine
X_dino = np.vstack([aesth_emb, una_emb])         # (N, D_dino)
labels = np.array([1]*len(aesth_emb) + [0]*len(una_emb))
cultures = aesth_cultures + una_cultures
files = aesth_files + una_files
N = X_dino.shape[0]
print("DINO combined shape:", X_dino.shape, "n images:", N)

idx_good = np.where(labels == 1)[0]   # aesthetic
idx_bad  = np.where(labels == 0)[0]   # unaesthetic

if N == 0:
    raise SystemExit("No images embedded with DINO. Check folders and filenames.")

# normalize DINO features for cosine sims
Xd_n = X_dino / (np.linalg.norm(X_dino, axis=1, keepdims=True) + 1e-12)

# ----------------embed the SAME images with CLIP (must match order) ----------------
# We'll embed aesthetic files first, then unaesthetic files, in the same order used for DINO.
print("Embedding the SAME images with OpenCLIP (ViT-B/32)...")

CLIP_CACHE = os.path.join(CACHE_DIR, "clip_embeddings.npy")

if os.path.exists(CLIP_CACHE):
    print("Loading cached CLIP embeddings")
    X_clip = np.load(CLIP_CACHE)
else:
    print("Embedding the SAME images with OpenCLIP...")
    X_clip = np.vstack([
        np.vstack([embed_clip(os.path.join(AESTH_DIR, fn)) for fn in tqdm(aesth_files)]),
        np.vstack([embed_clip(os.path.join(UNA_DIR, fn)) for fn in tqdm(una_files)])
    ])
    np.save(CLIP_CACHE, X_clip)
print("CLIP combined shape:", X_clip.shape)

# normalize CLIP features
Xc_n = X_clip / (np.linalg.norm(X_clip, axis=1, keepdims=True) + 1e-12)

# ---------------- compute similarity matrices (cosine) ----------------
sim_dino = Xd_n @ Xd_n.T    # NxN
sim_clip = Xc_n @ Xc_n.T    # NxN
# enforce symmetry numerically
sim_dino = (sim_dino + sim_dino.T) / 2.0
sim_clip = (sim_clip + sim_clip.T) / 2.0

# ---------------- per-image representational alignment ----------------
def per_image_alignment(simA, simB):
    """
    For each image i, compute Pearson r between simA[:,i] and simB[:,i].
    Returns array (N,) of r values (np.nan for degenerate cases).
    """
    N = simA.shape[0]
    rs = np.full(N, np.nan, dtype=float)
    for i in range(N):
        a = simA[:, i]
        b = simB[:, i]
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            rs[i] = np.nan
        else:
            rs[i] = pearsonr(a, b)[0]
    return rs

print("Computing per-image DINO<->CLIP alignment (Pearson r of sim-columns)...")
r_clip_to_dino = per_image_alignment(sim_dino, sim_clip)

# ---------------- DINO self-alignment  ----------------
print("Computing per-image DINO<->DINO alignment (baseline)...")
r_dino_to_dino = per_image_alignment(sim_dino, sim_dino)

mean_align_good_dino = np.nanmean(r_dino_to_dino[idx_good]) if idx_good.size > 0 else np.nan
mean_align_bad_dino  = np.nanmean(r_dino_to_dino[idx_bad])  if idx_bad.size > 0 else np.nan

# Aggregate by label
idx_good = np.where(labels==1)[0]
idx_bad  = np.where(labels==0)[0]
mean_align_good = np.nanmean(r_clip_to_dino[idx_good]) if idx_good.size>0 else np.nan
mean_align_bad  = np.nanmean(r_clip_to_dino[idx_bad])  if idx_bad.size>0 else np.nan
print("Mean alignment (CLIP -> DINO): aesthetic:", mean_align_good, "unaesthetic:", mean_align_bad)

# ---------------- make alignment plot (models on x axis, alignment to DINO on y axis) ----------------
# We only have one external model (CLIP). We'll plot a single x tick "CLIP (ViT-B/32)"
models_list = ["DINOv2 (ViT-B/14)", "OpenCLIP ViT-B/32"]

good_vals = [
    mean_align_good_dino,   # DINO → DINO
    mean_align_good         # CLIP → DINO
]

bad_vals = [
    mean_align_bad_dino,    # DINO → DINO
    mean_align_bad          # CLIP → DINO
]

plt.figure(figsize=(7,5))
xs = np.arange(len(models_list))
plt.plot(xs, good_vals, marker='o', label="Aesthetic", linewidth=2)
plt.plot(xs, bad_vals,  marker='s', label="Unaesthetic", linewidth=2)
plt.xticks(xs, models_list, rotation=20)
plt.ylabel("Alignment to DINO (mean Pearson r of sim-columns)")
plt.title("Representational alignment to DINO — Aesthetic vs Unaesthetic (CLIP)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_alignment_to_dino_clip_only.png"), dpi=200)
plt.close()
print("Saved model_alignment_to_dino_clip_only.png")

# ---------------- numeric CSV for per-image and aggregated alignment ----------------
pd.DataFrame({
    "filename": files,
    "culture": cultures,
    "label": labels,
    "r_clip_to_dino": r_clip_to_dino
}).to_csv(os.path.join(OUT_DIR, "per_image_alignment_clip_to_dino.csv"), index=False)

pd.DataFrame([{
    "mean_align_clip_good": float(mean_align_good) if not np.isnan(mean_align_good) else None,
    "mean_align_clip_bad": float(mean_align_bad) if not np.isnan(mean_align_bad) else None,
    "n_images": int(N),
    "n_aesthetic": int(len(idx_good)),
    "n_unaesthetic": int(len(idx_bad))
}]).to_csv(os.path.join(OUT_DIR, "alignment_summary_clip_to_dino.csv"), index=False)
print("Saved per-image and summary CSVs")

# ---------------- neighbor-based diagnostics (same-class neighbor rate) ----------------
nbrs = NearestNeighbors(n_neighbors=min(KNN_K+1, N), metric="cosine").fit(Xd_n)
dist, inds = nbrs.kneighbors(Xd_n)
neighbor_indices = inds[:, 1:KNN_K+1]
same_rate = np.array([ (labels[neighbor_indices[i]] == labels[i]).sum() / max(1, neighbor_indices.shape[1]) for i in range(N) ])
mean_same_good = np.nanmean(same_rate[labels==1]) if np.any(labels==1) else np.nan
mean_same_bad  = np.nanmean(same_rate[labels==0]) if np.any(labels==0) else np.nan

# plot neighbor same-class rate by k
k_range = list(range(1, min(KNN_K, N-1) + 1))
means_good = []
means_bad = []
for k in k_range:
    neighs_k = inds[:, 1:k+1]
    rates = np.array([ (labels[neighs_k[i]] == labels[i]).sum() / k for i in range(N) ])
    means_good.append(np.nanmean(rates[labels==1]) if np.any(labels==1) else np.nan)
    means_bad.append(np.nanmean(rates[labels==0]) if np.any(labels==0) else np.nan)

plt.figure(figsize=(8,5))
plt.plot(k_range, means_good, marker='o', label="Aesthetic")
plt.plot(k_range, means_bad, marker='s', label="Unaesthetic")
plt.xlabel("k (nearest neighbors)")
plt.ylabel("Mean same-class neighbor proportion")
plt.title("Neighbor-based coherence (DINO features)")
plt.legend(); plt.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "neighbor_same_class_rate_by_k_dino.png"), dpi=200); plt.close()
print("Saved neighbor_same_class_rate_by_k_dino.png")

# ---------------- centroid similarity heatmaps across cultures ----------------
cultures_set = sorted(set(cultures))
good_centroids = []
bad_centroids = []
for cult in cultures_set:
    idx_c = [i for i,c in enumerate(cultures) if c==cult and labels[i]==1]
    idx_b = [i for i,c in enumerate(cultures) if c==cult and labels[i]==0]
    if len(idx_c)>0:
        good_centroids.append(Xd_n[idx_c].mean(axis=0))
    else:
        good_centroids.append(np.zeros((Xd_n.shape[1],), dtype=float))
    if len(idx_b)>0:
        bad_centroids.append(Xd_n[idx_b].mean(axis=0))
    else:
        bad_centroids.append(np.zeros((Xd_n.shape[1],), dtype=float))

good_centroids = np.vstack(good_centroids)
bad_centroids = np.vstack(bad_centroids)

def rownorm(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True)+1e-12)
G = rownorm(good_centroids) @ rownorm(good_centroids).T
B = rownorm(bad_centroids)  @ rownorm(bad_centroids).T

plt.figure(figsize=(10,8))
sns.heatmap(G, vmin=-1, vmax=1, cmap="coolwarm", xticklabels=cultures_set, yticklabels=cultures_set)
plt.xticks(rotation=90); plt.title("Centroid similarity: Aesthetic (DINO)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "centroid_similarity_aesthetic_dino.png"), dpi=200); plt.close()

plt.figure(figsize=(10,8))
sns.heatmap(B, vmin=-1, vmax=1, cmap="coolwarm", xticklabels=cultures_set, yticklabels=cultures_set)
plt.xticks(rotation=90); plt.title("Centroid similarity: Unaesthetic (DINO)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "centroid_similarity_unaesthetic_dino.png"), dpi=200); plt.close()

print("Saved centroid heatmaps.")

# ---------------- t-SNE for quick visual inspection ----------------
Z = TSNE(n_components=2, perplexity=min(TSNE_PERP, max(5, N//3)), random_state=RANDOM_SEED).fit_transform(Xd_n)
plt.figure(figsize=(8,6)); plt.scatter(Z[labels==1,0], Z[labels==1,1], s=8, label="Aesthetic")
plt.scatter(Z[labels==0,0], Z[labels==0,1], s=8, label="Unaesthetic")
plt.legend(); plt.title("t-SNE (DINO features)"); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"tsne_dino_binary.png"), dpi=200); plt.close()

# ---------------- save a summary CSV ----------------
summary = {
    "mean_align_clip_good": float(mean_align_good) if not np.isnan(mean_align_good) else None,
    "mean_align_clip_bad": float(mean_align_bad) if not np.isnan(mean_align_bad) else None,
    "mean_neighbor_same_good": float(mean_same_good) if not np.isnan(mean_same_good) else None,
    "mean_neighbor_same_bad": float(mean_same_bad) if not np.isnan(mean_same_bad) else None,
    "n_images": int(N),
    "n_cultures": int(len(cultures_set))
}
pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, "alignment_summary_strict.csv"), index=False)
print("ALL DONE. Outputs in:", OUT_DIR)
