import os

# ====================== 【关键】强制线程开满 ======================
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

import numpy as np
import cv2
import pickle
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import nnls
from skimage.measure import label, regionprops

# ====================== 路径 ======================
save_cluster_path = r"D:\cluster\20_plex_left 2976, 3504, 18000, 18000 main\粗100_细100_1wclasses.pkl"
output_folder = r"D:\Decode_results\20p"
DAPI_PATH = r"D:\cluster\20_plex_left 2976, 3504, 18000, 18000 main\DAPI.tiff"
os.makedirs(output_folder, exist_ok=True)

# ====================== 参数 ======================
BACKGROUND_SUBTRACT = 5000
MAX_CELL_DIAMETER = 150
INTENSITY_THRESHOLD = 1000
NUCLEUS_RATIO_THRESHOLD = 0.001

# ====================== 输出模式选择 ======================
# True = 输出全部图片
# False = 只输出下面 custom_order 里指定的图片
EXPORT_ALL = True

# 👇 在这里写你想要输出的【序号】，想输出什么顺序就什么顺序
# 大肿：[2,19,28,41] B滤泡：[9,14,21,22,24,31]
custom_order = [9,20]

# ====================== 只保留前20个蛋白参考库 ======================
ref_library = {
    1: [ # 1_T-bet，同50p
        np.array([367, 216, 319, 338, 1322, 430, 500, 319, 213, 2019, 574, 282, 163, 98, 504, 631, 2173, 244]),*[np.zeros(18)]*9],
    2: [ # 2_HER2
        np.array([498, 169, 769, 1980, 1852, 507, 996, 220, 341, 1385, 944, 271, 170, 83, 743, 1971, 6274, 590]),*[np.zeros(18)]*9],
    3: [ # 3_CD45RA
        # V4 np.array([762, 355, 440, 491, 943, 569, 645, 224, 215, 469, 877, 189, 6397, 334, 662, 2806, 6261, 925]),
        # V5
        np.array([1200,325,673,491,614,920,1769,564,363,1073,2115,546,483,170,370,2427,5167,1201]),
        *[np.zeros(18)]*9],
    4: [ # 4_CD4,同50p
        np.array([973, 413, 2120, 525, 1130, 728, 1258, 1524, 448, 1119, 404, 672, 147, 85, 2131, 1377, 1131, 706]),*[np.zeros(18)]*9],
    5: [ # 5_FOxp3,同50p
        np.array([933, 240, 5332, 939, 1087, 857, 1381, 2134, 678, 2362, 795, 619, 206, 120, 1082, 5404, 2652, 744]),*[np.zeros(18)]*9],
    6: [ # 6_CD68,同50p
        np.array([500, 127, 146, 102, 225, 321, 865, 547, 189, 489, 213, 266, 211, 4686, 2565, 874, 5130, 1270]),*[np.zeros(18)]*9],
    7: [ # 7_MPO,同50p
        np.array([1250, 394, 450, 1533, 977, 807, 1396, 497, 5186, 1524, 730, 660, 188, 75, 739, 1304, 1918, 634]),*[np.zeros(18)]*9],
    8: [ # 8_CD8a,同50p
        np.array([2054, 520, 704, 706, 3304, 1042, 951, 1629, 530, 1465, 679, 607, 399, 168, 1681, 3304, 4321, 3227]),*[np.zeros(18)]*9],
    9: [ # 9_CD23
        # V3: 20-明虹全局挑出来的在滤泡的区域
        np.array([2438, 325, 368, 812, 1082, 5315, 474, 278, 171, 494, 319, 246, 47, 112, 124, 156, 427, 136]),*[np.zeros(18)]*9],
    10: [ # 10_CD27
        np.array([1464, 454, 486, 555, 841, 1229, 2091, 3720, 1409, 3901, 625, 841, 333, 103, 1437, 1239, 1833, 835]),*[np.zeros(18)]*9],
    11: [ # 11_Tryptase
        np.array([1460, 8641, 7764, 8645, 2008, 3377, 707, 503, 261, 732, 332, 216, 239, 108, 1223, 941, 1356, 448]),*[np.zeros(18)]*9],
    12: [ #12_CTLA-4 V2:明虹全局
        np.array([1485, 693, 10339, 1293, 3424, 1182, 840, 754, 361, 1056, 1368, 334, 240, 91, 468, 3613, 3031, 657]),*[np.zeros(18)]*9],
    13: [ # 13_LAG-3
        np.array([1800, 494, 1206, 16362, 925, 2996, 1070, 1115, 569, 1464, 354, 354, 220, 71, 1878, 986, 3524, 955]),*[np.zeros(18)]*9],
    14: [ # 14_PD-1
        np.array([11706, 438, 838, 577, 1168, 3851, 493, 155, 111, 372, 157, 200, 150, 143, 176, 328, 549, 194]),*[np.zeros(18)]*9],
    15: [ # 15_CD31
        np.array([995, 645, 848, 1035, 770, 790, 1033, 689, 559, 1707, 2284, 366, 314, 131, 501, 4875, 4349, 1113]),*[np.zeros(18)]*9],
    16: [ # 16_CD163
        np.array([1507, 430, 641, 2149, 16372, 1356, 1025, 293, 266, 1522, 573, 202, 107, 76, 675, 1597, 1667, 339]),*[np.zeros(18)]*9],
    17: [ # 17_ER
        np.array([359, 129, 269, 274, 580, 342, 1326, 356, 589, 1003, 1075, 1307, 143, 276, 357, 1381, 2490, 547]),*[np.zeros(18)]*9],
    18: [ #18_SMA
        np.array([1224, 151, 424, 572, 770, 1219, 4436, 1576, 1912, 4971, 3174, 720, 287, 80, 234, 779, 1745, 386]),*[np.zeros(18)]*9],
    19: [ # 19_CK19
        np.array([1462, 259, 1326, 1009, 1980, 652, 614, 490, 198, 1361, 5746, 566, 203, 129, 1222, 3379, 3747, 737]),*[np.zeros(18)]*9],
    20: [ # 20_ki67
        np.array([1478, 328, 949, 1126, 2757, 5495, 1034, 444, 475, 1477, 937, 304, 210, 58, 151, 1632, 2217, 525]),*[np.zeros(18)]*9],
}

# 只保留前20个蛋白名称
protein_names = [
    "T-bet","HER2","CD45RA","CD4","Foxp3","CD68","MPO","CD8a","CD23","CD27",
    "Tryptase","CTLA-4","LAG-3","PD-1","CD31","CD163","ER","SMA","CK19","Ki67"
]

# 只解码 1~20
ALL_PROT_IDS = list(range(1, 21))

# 构建参考库
all_refs = []
ref_to_protein = []
for prot_id in ALL_PROT_IDS:
    for spec in ref_library[prot_id]:
        all_refs.append(spec)
        ref_to_protein.append(prot_id)
all_refs = np.array(all_refs)

# 粗筛
def coarse_screen(target, topk=20):
    sim = cosine_similarity(target.reshape(1, -1), all_refs)[0]
    top_ids = np.argsort(sim)[-topk:]
    cand_p = np.unique([ref_to_protein[i] for i in top_ids])
    return cand_p

# 组合
def generate_valid_combos(candidate_p):
    combos = []
    lst = list(candidate_p)
    for k in range(1, 4):
        for c in combinations(lst, k):
            mats = [ref_library[i][0] for i in c]
            combos.append((c, np.array(mats).T))
    return combos

# 解码
def fine_decode(target, combos):
    best_loss = np.inf
    best_idx = None
    best_w = None
    for idx_tuple, A in combos:
        try:
            w, loss = nnls(A, target)
            if loss < best_loss:
                best_loss = loss
                best_idx = idx_tuple
                best_w = w
        except:
            continue
    return best_idx, best_w

# 加载聚类
print("🔸 加载聚类结果...")
with open(save_cluster_path, "rb") as f:
    cluster_result = pickle.load(f)
H = cluster_result["H"]
W = cluster_result["W"]
is_valid = cluster_result["is_valid"]
labels = cluster_result["labels"]
centers = cluster_result["centers"]

# 解码
print("🚀 开始解码...")
center_results = []
for c in tqdm(centers):
    cand = coarse_screen(c)
    combos = generate_valid_combos(cand)
    idx, w = fine_decode(c, combos)
    res = np.zeros(20, dtype=np.float32)  # 只保留20个
    if idx is not None and w is not None:
        for i, co in zip(idx, w):
            res[i-1] = co
    center_results.append(res)
center_results = np.array(center_results)

valid_pos = np.nonzero(is_valid)[0]
valid_lab = labels[valid_pos]

# 导出图像
print("💾 导出图像...")

# ====================== 智能导出：全部 / 自定义 ======================
if EXPORT_ALL:
    export_list = [(i, protein_names[i]) for i in range(20)]
else:
    export_list = [(idx-1, protein_names[idx-1]) for idx in custom_order if 1<=idx<=20]

skip_idx = {8, 19}
for ch_idx, name in export_list:
    if ch_idx in skip_idx:
        continue
    flat_arr = np.zeros(H * W, dtype=np.float32)
    flat_arr[valid_pos] = center_results[valid_lab, ch_idx]
    img = flat_arr.reshape(H, W)

    maxv = img.max()
    if maxv > 1e-6:
        out_img = (img / maxv * 65535).astype(np.uint16)
    else:
        out_img = np.zeros_like(img, dtype=np.uint16)

    save_path = os.path.join(output_folder, f"{ch_idx+1:02d}_{name}.tif")
    cv2.imwrite(save_path, out_img)
    del flat_arr, img, out_img

# CD23 & Ki67 拆分（一定会输出）
print("🔬 拆分 CD23 & Ki67...")
cd23_flat = np.zeros(H*W, np.float32)
cd23_flat[valid_pos] = center_results[valid_lab, 8]
cd23_img = cd23_flat.reshape(H,W)

ki67_flat = np.zeros(H*W, np.float32)
ki67_flat[valid_pos] = center_results[valid_lab, 19]
ki67_img = ki67_flat.reshape(H,W)

merge = cd23_img + ki67_img
if merge.max() > 1e-6:
    merge = (merge / merge.max()) * 65535.0

dapi = cv2.imread(DAPI_PATH, -1)
dapi8 = cv2.normalize(dapi, None, 0,255,cv2.NORM_MINMAX,cv2.CV_8U)
_, mask = cv2.threshold(dapi8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
mask = mask.astype(bool)

sub = np.clip(merge - BACKGROUND_SUBTRACT, 0, None)
cell_labels = label(sub > 0)
ki67_final = np.zeros_like(merge)
cd23_final = np.zeros_like(merge)

for reg in regionprops(cell_labels, intensity_image=merge):
    y, x = reg.coords[:,0], reg.coords[:,1]
    ok_size = reg.equivalent_diameter <= MAX_CELL_DIAMETER
    bright = merge[y,x] > INTENSITY_THRESHOLD
    ratio = np.mean(mask[y[bright], x[bright]]) if np.any(bright) else 0
    if ok_size and ratio >= NUCLEUS_RATIO_THRESHOLD:
        ki67_final[y,x] = merge[y,x]
    else:
        cd23_final[y,x] = merge[y,x]

cv2.imwrite(os.path.join(output_folder, "09_CD23.tif"), cd23_final.astype(np.uint16))
cv2.imwrite(os.path.join(output_folder, "20_Ki67.tif"), ki67_final.astype(np.uint16))

print("🎉 全部完成！只解码前20个蛋白")