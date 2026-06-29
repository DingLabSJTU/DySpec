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
save_cluster_path = r"D:\cluster\50_plex_left 1680, 5620, 18000, 18000 main\粗100_细10_1kclasses.pkl"
output_folder = r"D:\Decode_results"
DAPI_PATH = r"D:\cluster\50_plex_left 1680, 5620, 18000, 18000 main\DAPI.tiff"
os.makedirs(output_folder, exist_ok=True)

# ====================== 参数 ======================
BACKGROUND_SUBTRACT = 500
MAX_CELL_DIAMETER = 50
INTENSITY_THRESHOLD = 1000
NUCLEUS_RATIO_THRESHOLD = 0.001

# ====================== 输出模式选择 ======================
# True = 输出全部图片
# False = 只输出下面 custom_order 里指定的图片
EXPORT_ALL = False

# 👇 在这里写你想要输出的【序号】，想输出什么顺序就什么顺序
# 大肿：[2,19,28,41] B滤泡：[9,14,21,22,24,31]
custom_order = [17,39]

# ====================== 50蛋白参考库 ======================
ref_library = {
    1: [np.array([367, 216, 319, 338, 1322, 430, 500, 319, 213, 2019, 574, 282, 163, 98, 504, 631, 2173, 244]),*[np.zeros(18)]*9],
    2: [ # 2_HER2
        # V1从20-plex捞的，解不出来；下为从50-plex重新捞的V2,可解
        np.array([717, 872, 1307, 1856, 2779, 1083, 1035, 700, 820, 2080, 1486, 564, 608, 281, 538, 881, 552, 570]),*[np.zeros(18)]*9],
    3: [np.array([762, 355, 440, 491, 943, 569, 645, 224, 215, 469, 877, 189, 6397, 334, 662, 2806, 6261, 925]),*[np.zeros(18)]*9],
    4: [np.array([973, 413, 2120, 525, 1130, 728, 1258, 1524, 448, 1119, 404, 672, 147, 85, 2131, 1377, 1131, 706]),*[np.zeros(18)]*9],
    5: [np.array([933, 240, 5332, 939, 1087, 857, 1381, 2134, 678, 2362, 795, 619, 206, 120, 1082, 5404, 2652, 744]),*[np.zeros(18)]*9],
    6: [np.array([500, 127, 146, 102, 225, 321, 865, 547, 189, 489, 213, 266, 211, 4686, 2565, 874, 5130, 1270]),*[np.zeros(18)]*9],
    7: [np.array([1250, 394, 450, 1533, 977, 807, 1396, 497, 5186, 1524, 730, 660, 188, 75, 739, 1304, 1918, 634]),*[np.zeros(18)]*9],
    8: [ # 8_CD8a
        # V1: 原本属于CD56的barcode
        np.array([985,474,770,1341,1727,684,2401,2104,980,2254,1507,631,484,490,862,1187,980,678]),*[np.zeros(18)]*9],
    9: [ # 9_CD23
        # V3: 20-明虹全局挑出来的在滤泡的区域
        np.array([2438, 325, 368, 812, 1082, 5315, 474, 278, 171, 494, 319, 246, 47, 112, 124, 156, 427, 136]),*[np.zeros(18)]*9],
    10: [np.array([3157, 1313, 971, 1949, 1945, 777, 2031, 3575, 1077, 2116, 1940, 1098, 712, 897, 1647, 1784, 1933, 986]),*[np.zeros(18)]*9],
    11: [ # 11_try 目前V3解码出来的很干净
        np.array([1134,4672,1062,1143,1073,943,1012,629,642,3171,994,526,746,379,830,733,4871,784])],
    12: [np.array([1485, 693, 10339, 1293, 3424, 1182, 840, 754, 361, 1056, 1368, 334, 240, 91, 468, 3613, 3031, 657]),*[np.zeros(18)]*9],
    13: [np.array([1800, 494, 1206, 16362, 925, 2996, 1070, 1115, 569, 1464, 354, 354, 220, 71, 1878, 986, 3524, 955]),*[np.zeros(18)]*9],
    14: [ # 14_PD-1
        # V2: 用的20-plex扒拉出来的全局光谱，暗，但是有特征B区域
        np.array([11706, 438, 838, 577, 1168, 3851, 493, 155, 111, 372, 157, 200, 150, 143, 176, 328, 549, 194]),*[np.zeros(18)]*9],
    15: [np.array([995, 645, 848, 1035, 770, 790, 1033, 689, 559, 1707, 2284, 366, 314, 131, 501, 4875, 4349, 1113]),*[np.zeros(18)]*9],
    16: [np.array([1507, 430, 641, 2149, 16372, 1356, 1025, 293, 266, 1522, 573, 202, 107, 76, 675, 1597, 1667, 339]),*[np.zeros(18)]*9],
    17: [ #ER
        np.array([764,568,1318,2049,1874,932,2801,818,690,2250,1565,762,494,636,518,1117,813,814]),
        *[np.zeros(18)]*9],
    18: [ #18_SMA
        # 这条都与t-bet共定位了不要了，而且这条decode出来效果确实也不好，一堆杂点
        # np.array([1224, 151, 424, 572, 770, 1219, 4436, 1576, 1912, 4971, 3174, 720, 287, 80, 234, 779, 1745, 386])
        # V2-从50-plex重新扒拉的，这条明显干净一点
        np.array([1088,320,614,1418,1620,741,11682,2834,1179,3766,6356,1144,717,375,625,922,806,1015]),*[np.zeros(18)]*9],
    19: [np.array([1462, 259, 1326, 1009, 1980, 652, 614, 490, 198, 1361, 5746, 566, 203, 129, 1222, 3379, 3747, 737]),*[np.zeros(18)]*9],
    20: [ # 20_ki67
        # V2:后来重新扒拉的光谱
        np.array([1478, 328, 949, 1126, 2757, 5495, 1034, 444, 475, 1477, 937, 304, 210, 58, 151, 1632, 2217, 525]),*[np.zeros(18)]*9],
    21: [np.array([1.0000, 0.1829, 0.1133, 0.1710, 0.2462, 0.5829, 0.2846, 0.0972, 0.0966, 0.2824, 0.3826, 0.0743, 0.1129, 0.0739, 0.0897, 0.1337, 0.1260, 0.0871]),*[np.zeros(18)]*9],
    22: [np.array([1.0000, 0.2588, 0.1824, 0.2853, 0.3401, 0.8898, 0.3823, 0.1847, 0.1165, 0.3172, 0.2892, 0.0898, 0.1337, 0.0976, 0.1337, 0.1882, 0.1411, 0.1013]),*[np.zeros(18)]*9],
    23: [np.array([0.5000, 0.1848, 0.2640, 0.4313, 0.4848, 0.6994, 0.2074, 0.1332, 0.0736, 0.2787, 1.0, 0.1327, 0.0551, 0.0339, 0.1792, 0.7720, 0.9236, 0.1819]),*[np.zeros(18)]*9],
    24: [np.array([0.8124, 0.2083, 0.2033, 0.4410, 0.2935, 1.0000, 0.6534, 0.1679, 0.1485, 0.3560, 0.2954, 0.0897, 0.1439, 0.1249, 0.1832, 0.2660, 0.1974, 0.1549]),*[np.zeros(18)]*9],
    25: [np.array([0.7104, 0.2237, 0.8633, 0.7488, 1.0000, 0.3619, 0.4975, 0.3457, 0.2622, 0.8351, 0.6078, 0.2044, 0.2132, 0.1698, 0.3436, 0.4514, 0.2604, 0.2385]),*[np.zeros(18)]*9],
    26: [np.array([1.0000, 0.1839, 0.1367, 0.2374, 0.6327, 0.2600, 0.2098, 0.1514, 0.0795, 0.2840, 0.1946, 0.1171, 0.1367, 0.1043, 0.1480, 0.2126, 0.1636, 0.1329]),*[np.zeros(18)]*9],
    27: [np.array([1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),*[np.zeros(18)]*9],
    28: [ # 28_CK6
        # V1乱得很，下面V2结果不错
        np.array([1768,699,1918,3331,2898,1002,3281,1017,734,3279,3624,839,704,687,735,2447,841,827]),*[np.zeros(18)]*9],
    29: [np.array([0.1000, 0.0361, 0.0556, 0.0873, 0.0822, 0.0578, 0.0925, 0.0508, 0.0299, 0.0601, 0.0481, 0.0511, 0.0371, 0.0218, 0.0371, 0.0559, 0.0348, 0.0299]),*[np.zeros(18)]*9],
    30: [np.array([4236, 1500, 1774, 1901, 3054, 931, 2035, 2353, 1169, 2942, 1185, 838, 1863, 1155, 1397, 2011, 2525, 1714]),*[np.zeros(18)]*9],
    31: [ # 31_CD19
        # V1: 解不出来
        # np.array([1.0000, 0.2549, 0.1217, 0.1808, 0.3270, 0.6202, 0.2118, 0.1006, 0.0945, 0.2400, 0.2153, 0.0709, 0.1074, 0.0901, 0.1220, 0.1685, 0.1370, 0.1218]),
        # V2: 重新从50p全局挖的
        np.array([5101,2113,1527,1947,1591,3518,2207,917,650,2136,2415,504,891,645,833,1259,882,702]),
        *[np.zeros(18)]*9],
    32: [np.array([0.4080, 0.4937, 0.5248, 0.7183, 1.0000, 0.3635, 0.5778, 0.6422, 0.3811, 0.8664, 0.4111, 0.2395, 0.2777, 0.1821, 0.2718, 0.3749, 0.3044, 0.2201]),*[np.zeros(18)]*9],
    33: [np.array([1.0000, 0.2696, 0.3138, 0.5049, 0.6071, 0.5419, 0.8263, 0.3235, 0.1967, 0.6111, 0.4779, 0.2010, 0.3024, 0.2153, 0.2987, 0.7008, 0.3820, 0.4543]),*[np.zeros(18)]*9],
    34: [np.array([0.7956, 0.1022, 0.1291, 0.1832, 0.2345, 0.0980, 0.2243, 0.5038, 0.1476, 0.2710, 0.2997, 0.1457, 1.0000, 0.1175, 0.1174, 0.1925, 0.1146, 0.1218]),*[np.zeros(18)]*9],
    35: [ # 35_PCNA
        # V3: 后来用的V3
        np.array([604,981,1524,1966,2467,4485,2826,2578,813,838,934,854,696,1132,961,1422,876,743]),*[np.zeros(18)]*9],
    36: [ # 36_Cyclin D1
        # V1:后面扒拉的两条光谱还不如第一条
        np.array([1624, 491, 1193, 1687, 2385, 4808, 3801, 2141, 1107, 2343, 1142, 895, 647, 731, 700, 1568, 1287, 922]),*[np.zeros(18)]*9],
    37: [np.array([0.7158, 0.2506, 0.1666, 0.2832, 0.2529, 0.6598, 1.0000, 0.2015, 0.1200, 0.3402, 0.2537, 0.1032, 0.1134, 0.1027, 0.1404, 0.2001, 0.1415, 0.1183]),*[np.zeros(18)]*9],
    38: [np.array([2792, 1221, 1403, 2528, 1487, 647, 3257, 2001, 1177, 3291, 1201, 732, 989, 943, 1024, 2049, 3166, 1307]),*[np.zeros(18)]*9],
    39: [ # sox9
        # v2: np.array([4130, 1292, 1742, 2495, 2020, 1001, 2233, 1619, 913, 2333, 1324, 611, 1187, 840, 884, 1850, 1735, 1075]),
        # 80p-V4 np.array([2313,1061,1413,1924,2507,1158,5400,1070,666,4197,839,788,1020,1508,1029,3283,670,838]),
        np.array([972,497,1439,3263,2149,1308,2014,982,491,1876,1363,799,780,753,846,1942,819,1131]),
        *[np.zeros(18)]*7],
    40: [np.array([9845, 1300, 821, 1830, 2480, 809, 1531, 5011, 977, 1702, 1967, 1119, 8695, 1263, 645, 1444, 1005, 848]),*[np.zeros(18)]*9],
    41: [ # 41_VDAC1
        # V1 结果乱，又长得不像
        # np.array([0.5816, 0.5089, 0.4725, 0.7235, 1.0000, 0.4564, 0.6325, 0.4315, 0.2656, 0.7710, 0.5889, 0.2596, 0.3004, 0.2333, 0.6996, 0.4527, 0.2942, 0.2497]),
        # 重新扒拉一个V2看看
        np.array([1940,574,1550,2016,2939,1114,2428,970,667,2454,1400,685,1214,407,1889,2236,891,1101]),
        *[np.zeros(18)]*9],
    42: [np.array([0.5683, 0.1375, 0.2096, 0.2917, 0.3404, 0.1389, 0.1973, 0.1779, 0.1211, 1.0000, 0.5109, 0.2105, 0.1184, 0.0902, 0.1830, 0.2770, 0.1807, 0.1523]),*[np.zeros(18)]*9],
    43: [np.array([1.0000, 0.1515, 0.1170, 0.1840, 0.2523, 0.1124, 0.2782, 0.0787, 0.0604, 0.1831, 0.1924, 0.0681, 0.1273, 0.0616, 0.0807, 0.1647, 0.0908, 0.0753]),*[np.zeros(18)]*9],
    44: [np.array([0.3024, 0.0997, 0.2800, 0.4112, 0.4317, 0.2080, 0.6764, 0.3143, 0.1884, 0.5375, 1.0000, 0.2302, 0.1649, 0.0924, 0.1259, 0.2951, 0.1312, 0.1434]),*[np.zeros(18)]*9],
    45: [np.array([0.1000, 0.03755, 0.05356, 0.08843, 0.08095, 0.06221, 0.09346, 0.03448, 0.02400, 0.05633, 0.05762, 0.04044, 0.02937, 0.01975, 0.02918, 0.04940, 0.03105, 0.03494]),*[np.zeros(18)]*9],
    46: [np.array([1.0000, 0.3027, 0.3796, 0.5107, 0.7825, 0.2471, 0.6133, 0.1855, 0.1622, 0.4430, 0.2609, 0.1209, 0.1844, 0.1047, 0.1869, 0.2579, 0.2051, 0.1445]),*[np.zeros(18)]*9],
    47: [ # 47_CD57
        # np.array([0.3844, 0.1178, 0.2826, 0.6680, 0.7428, 0.2364, 1.0000, 0.3738, 0.2453, 0.8184, 0.1924, 0.2050, 0.2458, 0.2399, 0.2880, 0.3901, 0.3602, 0.3137]),
        np.array([1556,1477,1144,2704,3007,957,2048,1513,993,3313,779,830,995,2971,1166,1579,1458,1270]),
        *[np.zeros(18)]*9],
    48: [ # 48_CD56
        np.array([4689,1013,971,1983,3106,4723,1628,1545,740,3720,2125,691,878,3291,1436,1983,1176,1314]),*[np.zeros(18)]*9],
    49: [np.array([0.07444, 0.02748, 0.04370, 0.05904, 0.1000, 0.04052, 0.03911, 0.03067, 0.02859, 0.06696, 0.03511, 0.02370, 0.03785, 0.01830, 0.05230, 0.05726, 0.05407, 0.04111]),*[np.zeros(18)]*9],
    50: [np.array([0.5596, 0.3127, 0.3807, 0.7082, 1.0000, 0.2549, 0.4888, 0.3355, 0.2504, 0.5300, 0.4324, 0.1807, 0.1650, 0.1208, 0.1883, 0.2961, 0.2280, 0.1705]),*[np.zeros(18)]*9],
}

protein_names = [
    "T-bet","HER2","CD45RA","CD4","Foxp3","CD68","MPO","CD8a","CD23","CD27",
    "Tryptase","CTLA-4","LAG-3","PD-1","CD31","CD163","ER","SMA","CK19","Ki67",
    "CD21","CD20","OX40","CD79a","CD25","ICOS","HLA-DR","CK6","Caspase-1","AR",
    "CD19","vWF","PDPN","PD-L1","PCNA","Cyclin D1","CD74","CD38","SOX9","IDO-1",
    "VDAC1","GCDFP-15","PDGFR-β","VISTA","TIGIT","p27","CD57","CD56","CD206","γ-H2A"
]
ALL_PROT_IDS = list(range(1, 51))

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
    res = np.zeros(50, dtype=np.float32)
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
    # 导出全部
    export_list = [(i, protein_names[i]) for i in range(50)]
else:
    # 导出自定义顺序
    export_list = [(idx-1, protein_names[idx-1]) for idx in custom_order]

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

print("🎉 全部完成！")