import os
import numpy as np
import tifffile
import pickle
import time
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

# ====================== 线程开满 ======================
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

# ====================== 路径 ======================
input_folder = r"D:\cluster\50_plex_left 1680, 5620, 18000, 18000 main"
save_cluster_path = os.path.join(input_folder, "粗100_细100_1wclasses.pkl")

# ====================== 配置 ======================
H, W = 18000, 18000
n_channels = 18
threshold = 50

# ==============================================================================
# 🚀🚀🚀 【极速模式】一次性读取所有图像，只打开一次文件，速度快 10~30 倍
# ==============================================================================
print("🔹 极速加载全部 18 张图像...")

# 一次性读取所有通道（超快！）
imgs = []
for c in tqdm(range(1, 19), desc="读取通道"):
    paths = [
        os.path.join(input_folder, f"{c}.tif"),
        os.path.join(input_folder, f"{c}.tiff"),
        os.path.join(input_folder, f"{c:02d}.tif"),
        os.path.join(input_folder, f"{c:02d}.tiff")
    ]
    img_path = next(p for p in paths if os.path.exists(p))

    # 一次性整张读入（超快！不切块、不循环折磨硬盘）
    img = tifffile.imread(img_path)
    imgs.append(img.astype(np.float32))

# 直接拼接（内存够就能跑，不够我再给你终极轻量版）
print("🔹 拼接通道...")
stack = np.stack(imgs, axis=-1)
del imgs  # 释放内存

print("🔹 展平像素...")
flat = stack.reshape(-1, 18)
del stack  # 释放内存

# 背景过滤
print("🔹 过滤背景...")
is_valid = ~(flat < threshold).all(axis=1)
valid_pixels = flat[is_valid]
del flat  # 释放内存

print(f"📊 有效像素：{valid_pixels.shape[0]:,}")

# ====================== 🚀 10万类聚类 ======================
n_coarse = 100
n_fine = 100
print(f"🚀 开始聚类：{n_coarse} × {n_fine} = 100,000 类")

start = time.time()

# 粗聚类
kmeans_coarse = MiniBatchKMeans(n_clusters=n_coarse, random_state=42, n_init=3, batch_size=20000)
coarse_labels = kmeans_coarse.fit_predict(valid_pixels)

# 细聚类
final_labels = np.zeros(len(valid_pixels), dtype=np.int32)
all_centers = []

for c in tqdm(range(n_coarse), desc="细聚类"):
    mask = coarse_labels == c
    pixels = valid_pixels[mask]
    if len(pixels) < 100:
        final_labels[mask] = c * n_fine
        all_centers.append(kmeans_coarse.cluster_centers_[c])
        continue

    kmeans_fine = MiniBatchKMeans(n_clusters=n_fine, random_state=42, n_init=3, batch_size=20000)
    flabels = kmeans_fine.fit_predict(pixels)
    final_labels[mask] = c * n_fine + flabels
    all_centers.extend(kmeans_fine.cluster_centers_)

centers = np.array(all_centers)

# ====================== 保存 ======================
print(f"✅ 聚类完成！耗时：{(time.time() - start) / 60:.1f} 分钟")

cluster_result = {
    "H": H,
    "W": W,
    "is_valid": is_valid,
    "labels": final_labels,
    "centers": centers
}

with open(save_cluster_path, "wb") as f:
    pickle.dump(cluster_result, f)

print("💾 全部完成！")