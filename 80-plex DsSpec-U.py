# 改最大组合数只有一个要变动的地方：range(1,4) → range(1,3)

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
save_cluster_path = r"D:\cluster\80_plex_left 100, 4464, 18000, 18000 main\粗100_细10_1kclasses.pkl"
output_folder = r"D:\Decode_results\80p"
DAPI_PATH = r"D:\cluster\80_plex_left 100, 4464, 18000, 18000 main\DAPI.tiff"
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
# 大肿：[2,19,28,41,52,71]
# B滤泡：[9,14,21,22,24,31] 只要T1-480（PD-1）、T1-780(CD23)和T2-690(CD21)有峰，就会解出B信号，T2-570(CD20)和T1-520(CD19)不太支持循环染，T1-620(CD79a)本身染不是很好，因此贡献就相对较少了
# 阴性对照：[77,78,79,80]
custom_order = [40]

# ====================== 80蛋白参考库（已改为从1开始） ======================
ref_library = {
    1: [  # 01_T-bet
        np.array([1120,1002,1420,2471,2605,991,1961,1286,835,3123,1130,612,1382,1357,1060,1896,880,908]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    2: [  # 02_HER2
        # V1解不出来；V2解出来全是自发荧光纤维；下是V3，解出来明显更像，但没有明显左右深浅的差别
        np.array([1290, 1120, 1621, 2664, 2900, 1087, 1668, 1306, 806, 2208, 1245, 622, 473, 560, 865, 910, 926, 568]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    3: [  # 03_CD45RA
        np.array([8721,1197,1032,1289,2282,1032,750,1066,562,615,506,458,3621,843,728,1945,583,643]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    4: [  # 04_CD4
        np.array([3139,1291,1336,2444,2302,849,846,838,732,1129,916,578,1380,1636,1995,2066,829,1098]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    5: [  # 05_Foxp3
        np.array([3654,883,862,1725,1372,754,1700,959,575,1958,445,508,982,1000,829,2998,633,795]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    6: [  # 06_CD68
        np.array([0.2544, 0.1354, 0.1578, 0.1930, 0.2199, 0.2088, 0.2258, 0.4050, 0.2081, 0.3840, 0.3304, 0.1404, 0.0838, 0.2134, 0.1677, 0.4924, 1.0, 0.2186]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    7: [  # 07_MPO
        # np.array([6805,2993,1548,2317,2445,1261,1648,992,1522,3028,991,609,587,883,796,3709,474,739]),
        np.array([6805,2993,1548,2317,2445,1261,1648,992,1522,3028,991,609,587,883,796,3709,3474,739]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    8: [  # 08_CD8a
        np.array([7561, 1784, 1728, 1648, 2934, 924, 2699, 865, 1071, 1340, 973, 859, 733, 884, 729, 2159, 1252, 2653]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    9: [  # 09_CD23
        # V1: 解出来很亮，但是有背景
        np.array([6340,1278,1036,1318,1803,2740,687,1016,786,1647,1093,560,750,833,1044,1242,657,595]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    10: [  # 10_CD27
        np.array([3949, 1414, 1564, 1717, 2457, 876, 1321, 2843, 810, 1952, 1417, 684, 586, 931, 1202, 1434, 1030, 943]),
        np.array([3493,1378,1522,2022,2974,1088,1732,4286,1068,1756,853,960,601,1360,1383,1631,807,878]),
        np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    11: [  # 11_Tryptase
        # V1 np.array([2570,3236,2022,2641,3199,4612,797,1221,795,650,884,1422,2484,459,403,664,512,910]),
        # V2: 微调
        np.array([2623,3432,2270,2421,2248,4472,692,1217,746,823,1276,1310,895,403,362,1128,592,1019]),
        np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    12: [  # 12_CTLA-4
        np.array([3693,2199,4431,5064,5765,1437,1020,1523,1275,2220,1691,759,919,968,1225,1279,1116,1007]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    13: [  # 13_LAG-3
        np.array([2015,1768,2719,3307,4221,1282,1909,1513,575,2771,955,833,1531,1082,812,956,566,732]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    14: [  # 14_PD-1
        # V1：解出来一堆杂信号
        # np.array([1.0, 0.0255, 0.3552, 0.0783, 0.0837, 0.3229, 0.0440, 0.0216, 0.0150, 0.0474, 0.0340, 0.0152, 0.0092, 0.0079, 0.0081, 0.0253, 0.0431, 0.0140]),
        # CD79A取奇数，不行
        # np.array([0.8124, 0, 0.2033, 0, 0.2935, 0, 0.6534, 0, 0.1485, 0, 0.2954, 0, 0.1439, 0, 0.1832, 0, 0.1974, 0]),
        # V2: 从80p全局找
        np.array([0.3691, 0.0509, 0.1543, 0.1244, 0.1755, 1.0, 0.1545, 0.0710, 0.0601, 0.1717, 0.1260, 0.0371, 0.0171, 0.0118, 0.0184, 0.0356, 0.0597, 0.0209]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    15: [  # 15_CD31
        np.array([0.1553, 0.1110, 0.2994, 0.1423, 0.1840, 0.1409, 0.0873, 0.0419, 0.0336, 0.0844, 0.1550, 0.0449, 0.0329, 0.0138, 0.0454, 0.5902, 1.0, 0.1634]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    16: [  # 16_CD163
        np.array([5471,1490,1744,2033,4562,1508,1185,669,640,1281,883,439,559,736,1157,4337,683,737]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    17: [  # 17_ER
        # V0: np.array([0.4467, 0.1224, 0.1382, 0.2776, 1.0, 0.3054, 0.8404, 0.3028, 0.1527, 0.3262, 0.2006, 0.1495, 0.1161, 0.0637, 0.7369, 0.7363, 0.9937, 0.2997]),
        np.array([1653,1572,1350,2357,2663,1125,2764,2040,853,1756,869,798,1228,870,864,1285,518,795]),
        np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    18: [  # 18_SMA
        np.array([1182, 759, 1211, 2056, 3840, 2693, 8387, 2380, 1368, 2213, 1955, 818, 824, 578, 1034, 1118, 571, 693]),
        np.array([1679, 1086, 1307, 1448, 2395, 4116, 7174, 1687, 864, 1362, 934, 550, 779, 572, 764, 1207, 616, 598]),
        np.array([2116, 760, 1306, 2369, 2438, 1338, 8808, 2427, 1030, 1216, 881, 655, 1394, 641, 1026, 1942, 777, 643]),
        np.array([2033, 1126, 1491, 2595, 2758, 1837, 4636, 2024, 907, 1561, 765, 555, 792, 639, 1055, 1819, 718, 609]),
        np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    19: [  # 19_CK19
        np.array([2010, 1729, 1504, 2671, 3981, 1304, 983, 1558, 884, 1485, 1317, 527, 901, 1032, 1301, 1411, 628, 689]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    20: [  # 20_Ki67
        # 下面这三条解码出来都不是特别多
        np.array([11194,1569,1518,2554,4120,8743,2874,1228,685,2594,1258,633,584,1029,789,2339,569,683]),
        np.array([6780,1111,942,1815,2556,7921,702,565,601,1478,1538,421,623,754,811,1454,425,562]),
        np.array([8862,1831,1246,2324,2475,4765,1344,1424,874,1515,802,415,661,1003,1222,2407,630,870]),
        # 下面这条是跟50p一样的barcode,解码出来还是很少
        np.array([1478, 328, 949, 1126, 2757, 5495, 1034, 444, 475, 1477, 937, 304, 210, 58, 151, 1632, 2217, 525]),
        np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    21: [  # 21_CD21
        np.array([4044,3398,2867,2324,2918,10655,794,546,840,592,752,1114,811,364,336,751,628,1319]),
        # V1: 从80p全局重新找的
        # np.array([6672,1278,1043,1149,2116,4047,2162,646,802,2747,1314,639,609,915,858,1135,914,912]),
        # V2：80p全局
        # np.array([3927,1660,1697,2730,3730,1366,1360,2074,1008,2107,2024,841,1542,677,1035,2059,775,806]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    22: [  # 22_CD20
        np.array([1.0000, 0.2588, 0.1824, 0.2853, 0.3401, 0.8898, 0.3823, 0.1847, 0.1165, 0.3172, 0.2892, 0.0898, 0.1337, 0.0976, 0.1337, 0.1882, 0.1411, 0.1013]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    23: [  # 23_OX40
        np.array([1803, 1831, 1355, 1850, 1823, 1686, 911, 665, 900, 1520, 747, 641, 724, 613, 446, 532, 699, 1017]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    24: [  # 24_CD79a
        np.array([0.8124, 0.2083, 0.2033, 0.4410, 0.2935, 1.0000, 0.6534, 0.1679, 0.1485, 0.3560, 0.2954, 0.0897, 0.1439, 0.1249, 0.1832, 0.2660, 0.1974, 0.1549]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    25: [  # 25_CD25
        np.array([0.7104, 0.2237, 0.8633, 0.7488, 1.0000, 0.3619, 0.4975, 0.3457, 0.2622, 0.8351, 0.6078, 0.2044, 0.2132, 0.1698, 0.3436, 0.4514, 0.2604, 0.2385]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    26: [  # 26_ICOS
        # V1: np.array([1.0000, 0.1839, 0.1367, 0.2374, 0.6327, 0.2600, 0.2098, 0.1514, 0.0795, 0.2840, 0.1946, 0.1171, 0.1367, 0.1043, 0.1480, 0.2126, 0.1636, 0.1329]),
        np.array([6956,1573,1620,2148,6102,1364,2044,646,702,1174,749,572,398,894,1644,6415,678,925]),
        np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    27: [  # 27_HLA-DR
        np.array([1.0000, 0.1218, 0.1469, 0.2075, 0.2702, 0.1464, 0.1783, 0.1136, 0.0824, 0.2398, 0.2437, 0.0714, 0.0847, 0.0582, 0.0979, 0.1627, 0.1022, 0.0748]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    28: [  # 28_CK6
        # V2用的50-plex的，扒拉出来效果不好
        # np.array([1768,699,1918,3331,2898,1002,3281,1017,734,3279,3624,839,704,687,735,2447,841,827]),
        # V3重新从80全局扒拉的
        np.array([2623,1709,1512,2688,3439,1460,980,1331,727,2201,1021,790,922,928,1478,1965,723,769]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    29: [ # Caspase-1
        np.array([430, 244, 217, 262, 321, 913, 338, 289, 188, 287, 251, 533, 319, 225, 283, 311, 279, 632]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    30: [  # 30_AR
        np.array([1714,1006,1610,1566,2478,911,2294,1377,933,2320,1228,710,424,755,732,978,447,478]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    31: [  # 31_CD19
        np.array([1.0000, 0.2549, 0.1217, 0.1808, 0.3270, 0.6202, 0.2118, 0.1006, 0.0945, 0.2400, 0.2153, 0.0709, 0.1074, 0.0901, 0.1220, 0.1685, 0.1370, 0.1218]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    32: [  # 32_vWF
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    33: [  # 33_PDPN
        np.array([3209,2166,1098,651,1011,987,2983,1244,988,821,644,398,521,677,371,464,1908,489]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    34: [  # 34_PD-L1
        np.array([0.7956, 0.1022, 0.1291, 0.1832, 0.2345, 0.0980, 0.2243, 0.5038, 0.1476, 0.2710, 0.2997, 0.1457, 1.0000, 0.1175, 0.1174, 0.1925, 0.1146, 0.1218]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    35: [  # 35_PCNA
        # V1 这用的还是50p光谱
        np.array([604,981,1524,1966,2467,4485,2826,2578,813,838,934,854,696,1132,961,1422,876,743]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    36: [  # 36_Cyclin D1
        # 这用的还是50p光谱
        np.array([1624, 491, 1193, 1687, 2385, 4808, 3801, 2141, 1107, 2343, 1142, 895, 647, 731, 700, 1568, 1287, 922]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    37: [  # 37_CD74
        np.array([3783,1042,1716,2026,3024,1104,4233,1263,659,1678,815,567,1077,1472,891,1165,665,787]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    38: [  # 38_CD38
        np.array([1.0000, 0.2258, 0.1855, 0.1913, 0.2125, 0.1126, 0.1125, 0.2122, 0.1011, 0.2075, 0.1709, 0.0966, 0.2333, 0.0987, 0.1139, 0.1900, 0.2368, 0.1384]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    39: [  # 39_SOX9
        np.array([2313,1061,1413,1924,2507,1158,5400,1070,666,4197,839,788,1020,1508,1029,3283,670,838]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    40: [  # 40_IDO-1
        # V1 np.array([0.3585, 0.0974, 0.2935, 0.5357, 0.5365, 0.1917, 0.4217, 1.0000, 0.2467, 0.5918, 0.2369, 0.2533, 0.2533, 0.1453, 0.2304, 0.3482, 0.2351, 0.2055]),
        # V2
        np.array([1843,1587,1296,2035,3333,1335,906,2602,1034,1246,1112,677,4066,703,1061,1372,559,805]),
        np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    41: [  # 41_VDAC1
        # V1：弱，放大还是有点像的，那还不如用原本这根呢，但是这根还是数，之后记得切换一下
        np.array([0.5816, 0.5089, 0.4725, 0.7235, 1.0000, 0.4564, 0.6325, 0.4315, 0.2656, 0.7710, 0.5889, 0.2596, 0.3004, 0.2333, 0.6996, 0.4527, 0.2942, 0.2497]),
        # V2: 从80-plex重新扒拉了一根，感觉效果更差了
        # np.array([1409,1057,1104,1693,2362,1061,669,1348,771,1142,993,526,635,847,1631,985,808,674]),
        np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    42: [  # 42_GCDFP-15
        np.array([1889,1065,1131,3183,2724,842,869,1392,604,1252,816,482,459,646,1139,1137,575,721]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    43: [  # 43_PDGFR-β
        np.array([1.0000, 0.1515, 0.1170, 0.1840, 0.2523, 0.1124, 0.2782, 0.0787, 0.0604, 0.1831, 0.1924, 0.0681, 0.1273, 0.0616, 0.0807, 0.1647, 0.0908, 0.0753]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    44: [  # 44_VISTA
        np.array([0.3024, 0.0997, 0.2800, 0.4112, 0.4317, 0.2080, 0.6764, 0.3143, 0.1884, 0.5375, 1.0000, 0.2302, 0.1649, 0.0924, 0.1259, 0.2951, 0.1312, 0.1434]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    45: [ # TIGIT
        np.array([2027,1088,1023,4416,1057,787,821,522,892,1024,2839,1211,1944,1088,946,872,667,435]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    46: [  # 46_p27
        np.array([2903,1314,1477,2255,2792,1014,4056,1747,651,3603,1260,735,1636,1258,919,1825,597,678]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    47: [  # 47_CD57
        # V1: np.array([3060,1868,2782,1877,2335,1139,3412,1668,853,3218,1180,835,798,2132,1063,1389,908,956]),
        np.array([1060,868,782,877,2335,1139,1412,1668,853,3218,1180,835,798,3132,1063,1389,908,956]),
        np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    48: [  # 48_CD56
        np.array([2299,1005,1033,2110,2084,742,3602,1495,1020,2774,1087,625,570,2380,947,1492,703,983]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    49: [ # CD206
        np.array([1388,1346,4421,2341,1564,1347,1091,877,1655,1436,1734,3876,1422,1011,1251,2567,821,491]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    50: [  # 50_γ-H2A
        # V0: np.array([0.5596, 0.3127, 0.3807, 0.7082, 1.0000, 0.2549, 0.4888, 0.3355, 0.2504, 0.5300, 0.4324, 0.1807, 0.1650, 0.1208, 0.1883, 0.2961, 0.2280, 0.1705]),
        np.array([5193,1707,1965,1827,5281,1190,1805,695,546,2445,875,453,340,732,559,1878,592,526]),
        np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    51: [  # 51_SOX2
        np.array([3898,1815,1401,3148,3012,1250,1041,829,759,3507,1245,611,494,525,507,722,434,637]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    52: [  # 52_SOD2
        # V0 np.array([1.0000, 0.1942, 0.1669, 0.4462, 0.4341, 0.1547, 0.2324, 0.2090, 0.0967, 0.1972, 0.1131, 0.0821, 0.0838, 0.1601, 0.1469, 0.3437, 0.0909, 0.1225]),
        # V3 np.array([2979,2093,2618,4350,5725,1373,854,2331,1133,2334,1770,710,2191,603,1029,1616,749,759]),
        np.array([3308,1926,1477,2228,3028,1048,729,1679,811,2052,1004,657,2611,796,1377,1541,588,668]),
        np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    53: [ # MMP-9
        np.array([3521,2351,1171,983,1061,1887,3321,1621,1324,841,772,354,441,654,335,531,2541,351]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    54: [  # 54_PR
        # V1 np.array([3310,1767,1544,3029,2988,1139,1861,1504,1348,2123,1122,584,551,870,976,1218,912,853]),
        np.array([4939,2275,2304,4622,4007,1486,1424,1319,1776,2890,1447,694,620,1003,1099,1750,994,997]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    55: [  # 55_S100A9
        np.array([1745,1133,1427,1891,2886,1325,1094,2815,3552,2482,1842,977,1621,875,898,1491,577,3619]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    56: [  # 56_FAP
        # V1: np.array([1892,1021,981,2092,6471,3219,1921,2215,1982,3387,1092,987,567,2981,779,478,732,887]),
        np.array([1507,430,641,2149,16372,1356,1025,293,266,1522,573,202,107,76,675,1597,1667,339]),
        np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    57: [  # 57_CD209
        # np.array([1.0000, 0.2721, 0.3273, 0.5723, 0.5381, 0.2503, 0.3174, 0.9541, 0.3083, 0.4262, 0.3337, 0.2315, 0.1437, 0.2505, 0.1944, 0.3834, 0.1706, 0.1574]),
        np.array([4531,1233,1483,1593,1438,1134,1438,4323,1397,931,1512,1049,651,1135,881,7737,773,713]),
        np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    58: [  # 58_CD123
        # V1: np.array([0.7293, 0.2405, 0.4069, 0.7852, 0.8590, 0.3153, 0.4113, 0.5748, 0.2323, 0.4663, 0.2828, 0.2361, 1.0000, 0.4311, 0.5041, 0.8097, 0.3070, 0.3218]),
        # V2: np.array([1514,682,542,721,1257,619,1007,404,632,659,585,317,2491,536,647,1000,566,594]),
        np.array([7697,1186,1431,1334,2735,1369,3232,2801,715,791,651,684,9850,876,703,1340,532,593]),
        np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    59: [  # 59_CD11c
        np.array([1.0000, 0.2481, 0.2661, 0.4225, 0.6322, 0.2023, 0.3486, 0.2674, 0.2422, 0.4463, 0.4717, 0.1365, 0.1754, 0.2088, 0.2168, 0.3936, 0.1304, 0.1559]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    60: [  # 60_CD34
        np.array([1433,1879,1857,2169,2710,6078,4349,1600,757,1674,911,521,756,528,648,856,584,539]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    61: [  # 61_GZMB
        np.array([1.0000, 0.2541, 0.2744, 0.5453, 0.7256, 0.2139, 0.4161, 0.3164, 0.1278, 0.4368, 0.2695, 0.1284, 0.1274, 0.1799, 0.3041, 0.3475, 0.1765, 0.2261]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    62: [  # 62_CD11b
        np.array([0.7327, 0.2351, 0.2278, 0.4092, 0.4590, 0.1970, 0.3430, 0.1819, 0.1468, 0.3529, 0.2025, 0.1314, 0.1411, 0.2010, 0.2229, 1.0000, 0.1584, 0.2299]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    63: [  # 63_CD45RO
        np.array([1.0000, 0.3599, 0.3873, 0.7341, 0.9263, 0.3205, 0.5936, 0.6554, 0.2656, 0.7989, 0.3918, 0.2321, 0.2531, 0.3405, 0.5807, 0.8945, 0.2357, 0.3004]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    64: [  # 64_CD66b
        np.array([2142,1450,1684,1676,6695,1160,6487,1143,872,3012,897,673,751,974,1037,3732,726,860]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    65: [  # 65_CD117
        # V1 np.array([2211,1430,1904,2191,2862,2797,1918,1109,605,1989,917,431,729,619,847,1345,432,530]),
        # np.array([1430,2211,1904,2191,2862,2797,1918,1109,605,1989,917,431,729,619,847,1345,432,530]),
        np.array([1915,3807,1452,2207,2176,1788,2136,1385,991,1780,925,484,481,489,509,989,708,514]),
        np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    66: [ # CD80
        np.array([1092,2011,7386,1874,1011,973,834,1291,1377,5377,2109,1877,1476,1137,1096,897,1044,2431]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    67: [ # CD86
        np.array([1334,1587,1043,7371,1436,987,1221,5543,1342,1164,1052,1247,1338,1642,3346,1985,744,678]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    68: [  # 68_p21
        # V2: np.array([3674,1173,1237,1804,4160,1127,3442,1358,732,2392,933,614,1128,828,803,2224,640,600]),
        np.array([5336,1185,1249,1877,2249,873,2539,1305,688,3045,1094,669,2599,641,1010,1474,623,758]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    69: [  # 69_Caspase-3
        np.array([3577,1813,1644,2249,3463,1179,2751,918,724,4359,1643,596,651,1131,997,3590,589,790]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    70: [  # 70_IL-1β
        np.array([3131,1510,3504,2209,2923,1085,3093,3852,1041,2227,1070,586,2858,839,1065,1561,566,602]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    71: [  # 71_NOX2
        # V1: np.array([1.0000, 0.3478, 0.2246, 0.3071, 0.3785, 0.1724, 0.1488, 0.1566, 0.0996, 0.1541, 0.1041, 0.0797, 0.1034, 0.1090, 0.1764, 0.7571, 0.0895, 0.1057]),
        np.array([5593,1897,1318,1912,1893,1077,755,892,572,722,561,450,519,634,1161,4424,553,576]),
        np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    72: [  # 72_HIF-1α
        np.array([3148, 1228, 2008, 2083, 1490, 783, 1147, 1182, 606, 2526, 713, 462, 2693, 950, 876, 1283, 681, 744]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    73: [  # 73_COUP-TF1√
        np.array([0.5782,0.5327,0.55,0.6543,0.8322,0.2943,1,0.4566,0.221,0.8351,0.345,0.2075,0.2599,0.8542,0.249,0.4627,0.2315,0.2519]),
        # V2:np.array([3669, 1201, 1207, 1513, 3077, 962, 1636, 775, 972, 1514, 723, 463, 703, 2111, 883, 1353, 701, 1193]),
        # V3:np.array([1951, 808, 895, 1143, 2013, 949, 1309, 891, 579, 1666, 617, 368, 665, 1578, 855, 1121, 568, 763]),
        # V4:np.array([4019,1159,1934,1647,2021,1065,3505,1232,714,2638,904,1105,927,2029,791,1397,728,1048]),
        np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    74: [  # 74_ATG-7
        # V0: 小区域挖出来的3条均值 np.array([1.0000, 0.8453, 0.4025, 0.5436, 0.7096, 0.3204, 0.2381, 0.3014, 0.3226, 0.4324, 0.2432, 0.1501, 0.2129, 0.2247, 0.2200, 0.5261, 0.1983, 0.3462]),
        # V2: np.array([2436,4479,2361,2677,3293,7951,758,1022,776,754,835,1118,897,439,405,933,735,1097]),
        np.array([2737,2775,1374,1819,2539,1099,1112,1186,1100,1250,798,604,560,755,819,1877,693,2061]),
        np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    75: [  # 75_HMGB1√
        # np.array([0.7329, 0.2452, 0.2849, 0.5343, 0.4940, 0.1990, 0.4478, 0.5418, 0.2303, 1.0000, 0.3649, 0.1939, 0.1724, 0.3021, 0.2943, 0.5328, 0.1847, 0.2529]),
        np.array([2592,968,1211,2197,2357,851,1697,2647,962,4761,1484,860,555,1287,1598,2998,727,1127]),
        np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    76: [ # GSDMD(-)
        np.array([2299,787,4279,1095,1907,756,588,395,2572,878,500,350,2838,524,611,1380,424,485]),
        # V2: np.array([2299,787,673,1095,1907,756,588,395,411,878,500,350,288,524,611,1380,424,485]),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
        np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18),
    ],
    77: [ # MsIgG1
        np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
    ],
    78: [ # MsIgG2a
        np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]),
    ],
    79: [ # MsIgG2b
        np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
    ],
    80: [ # RbIgG
        np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    ],
}

protein_names = [
    "T-bet","HER2","CD45RA","CD4","Foxp3","CD68","MPO","CD8a","CD23","CD27",
    "Tryptase","CTLA-4","LAG-3","PD-1","CD31","CD163","ER","SMA","CK19","Ki67",
    "CD21","CD20","OX40","CD79a","CD25","ICOS","HLA-DR","CK6","Caspase-1","AR",
    "CD19","vWF","PDPN","PD-L1","PCNA","Cyclin D1","CD74","CD38","SOX9","IDO-1",
    "VDAC1","GCDFP-15","PDGFR-β","VISTA","TIGIT","p27","CD57","CD56","CD206","γ-H2A",
    "SOX2","SOD2","MMP-9","PR","S100A9","FAP","CD209","CD123","CD11c","CD34",
    "GZMB","CD11b","CD45RO","CD66b","CD117","CD80","CD86","p21","Caspase-3","IL-1β",
    "NOX2","HIF-1α","COUP-TF1","ATG-7","HMGB1","GSDMD(-)","MsIgG1","MsIgG2a","MsIgG2b","RbIgG"
]
ALL_PROT_IDS = list(range(1, 81))  # 1~80

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
    res = np.zeros(80, dtype=np.float32)
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
    export_list = [(i, protein_names[i]) for i in range(80)]
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

print("🎉 全部完成！80重蛋白解码成功！")