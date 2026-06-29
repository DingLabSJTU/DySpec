# -*- coding: utf-8 -*-
"""
DySpec pixel-wise KMeans clustering script
Output file: cluster_result.pkl
"""
import os
import numpy as np
import cv2
import pickle
from sklearn.cluster import KMeans

# ====================== Multithreading settings to suppress runtime warnings ======================
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

# ====================== File path configuration ======================
def main():
    input_folder = r"D:\3"
    save_cluster_path = r"D:\3\cluster_result.pkl"

    # ====================== Load 18 x 16-bit TIFF images ======================
    print("🔹 Loading 18 TIFF images...")
    imgs = []
    for i in range(1, 19):
        file_path = os.path.join(input_folder, f"{i}.tif")
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        imgs.append(img)

    H, W = imgs[0].shape
    stack = np.stack(imgs, axis=-1)
    flat = stack.reshape(-1, 18)

    # ====================== Background pixel filtering ======================
    intensity_threshold = 50
    is_background = (flat < intensity_threshold).all(axis=1)
    is_valid = ~is_background
    valid_pixels = flat[is_valid]

    print(f"📊 Total valid pixels: {valid_pixels.shape[0]:,}")

    # ====================== KMeans clustering ======================
    n_clusters = min(1000, valid_pixels.shape[0])
    print(f"🧠 Starting KMeans clustering with {n_clusters} clusters")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    labels = kmeans.fit_predict(valid_pixels)
    centers = kmeans.cluster_centers_

    # ====================== Package and save clustering results ======================
    cluster_result = {
        "H": H,
        "W": W,
        "is_valid": is_valid,
        "labels": labels,
        "centers": centers,
    }

    with open(save_cluster_path, "wb") as f:
        pickle.dump(cluster_result, f)

    print(f"✅ Clustering finished. Result saved to: {save_cluster_path}")


if __name__ == "__main__":
    main()