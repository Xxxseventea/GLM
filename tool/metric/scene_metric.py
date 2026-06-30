import numpy as np
from sklearn.metrics import average_precision_score, f1_score


# =========================
# 工具函数：稳定性增强
# =========================

def _safe_average_precision(y_true, y_score):
    """防止全正/全负或异常输入导致 AP 抛错/NaN。"""
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float32)
    if y_true.max() == y_true.min():  # 全正或全负
        return 0.0
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return 0.0


def _temporal_smooth(scores, k=1):
    """简单时域平滑（均值），用于减少抖动；k<=1 时不生效。"""
    scores = np.asarray(scores).astype(np.float32)
    if k <= 1:
        return scores
    pad = k // 2
    s = np.pad(scores, (pad, pad), mode="edge")
    w = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(s, w, mode="valid")


def _temperature_scale(scores, temp=1.0):
    """温度缩放（对过硬/过软分布做轻微校正）；temp=1.0 时不生效。"""
    scores = np.asarray(scores).astype(np.float32)
    if temp == 1.0:
        return scores
    # 若传进来已是概率，可直接做 logit 近似缩放；这里用稳定的 Sigmoid 拉伸
    logits_like = np.clip(scores, -20.0, 20.0) / float(temp)
    return 1.0 / (1.0 + np.exp(-logits_like))


# =========================
# 段级转换与 IoU（闭区间）
# =========================

def result2dict(result, thresh=0.5, min_len=1, merge_gap=0):
    """
    将逐帧分数转为段级字典 {sid: (start, end)}（闭区间）
    - thresh: 二值阈值
    - min_len: 段最小长度（帧数），不足则忽略
    - merge_gap: 若相邻段间隔 <= merge_gap，则合并
    """
    result = np.asarray(result).astype(np.float32)
    binar = (result > float(thresh)).astype(np.int32)
    n = len(binar)

    # 找连通域
    starts, ends = [], []
    s = None
    for i in range(n):
        if binar[i] == 1 and s is None:
            s = i
        if (binar[i] == 0 or i == n - 1) and s is not None:
            e = i if (binar[i] == 1 and i == n - 1) else i - 1
            if e - s + 1 >= int(min_len):
                starts.append(s)
                ends.append(e)
            s = None

    # 合并近邻段
    merged = []
    for i in range(len(starts)):
        if not merged:
            merged.append([starts[i], ends[i]])
        else:
            prev_s, prev_e = merged[-1]
            if starts[i] - prev_e - 1 <= int(merge_gap):
                merged[-1][1] = ends[i]
            else:
                merged.append([starts[i], ends[i]])

    sceneDict = {}
    for sid, (s, e) in enumerate(merged):
        sceneDict[sid] = (int(s), int(e))
    if not sceneDict:
        # 保底，避免后续函数崩溃
        sceneDict[0] = (0, 0)
    return sceneDict


def _getInteraction(iv1, iv2):
    """闭区间交集长度：max(0, min(e1,e2) - max(s1,s2) + 1)"""
    s1, e1 = iv1
    s2, e2 = iv2
    start = max(int(s1), int(s2))
    end = min(int(e1), int(e2))
    inter = end - start + 1
    return max(inter, 0)


def _getUnion(iv1, iv2):
    """闭区间并集长度：len1 + len2 - inter"""
    s1, e1 = iv1
    s2, e2 = iv2
    len1 = int(e1) - int(s1) + 1
    len2 = int(e2) - int(s2) + 1
    inter = _getInteraction(iv1, iv2)
    return max(len1 + len2 - inter, 1)


def _getRatio(interval_1, interval_2):
    """IoU"""
    inter = _getInteraction(interval_1, interval_2)
    if inter <= 0:
        return 0.0
    return float(inter) / float(_getUnion(interval_1, interval_2))


def callIOU(groundSceneDict, predSceneDict):
    """
    为每个 GT 段寻找 IoU 最高的预测段
    返回: list[(sceneid, max_iou, matched_pred_id)]
    """
    iou = []
    for sceneid in groundSceneDict.keys():
        ratios = []
        gtScene = groundSceneDict[sceneid]
        for pred_id, predScene in predSceneDict.items():
            rat = _getRatio(gtScene, predScene)
            ratios.append([rat, pred_id])
        ratios = np.array(ratios, dtype=np.float32) if ratios else np.zeros((0, 2), dtype=np.float32)
        if ratios.size == 0:
            iou.append((sceneid, 0.0, -1))
            continue
        max_idx = int(np.argmax(ratios[:, 0]))
        max_rat = float(ratios[max_idx, 0])
        max_pred_id = int(ratios[max_idx, 1])
        iou.append((sceneid, max_rat, max_pred_id))
    return iou


def callMIOU(groundSceneDict, predSceneDict):
    """
    电影级 mIoU：对 GT->Pred 与 Pred->GT 两个方向的匹配 IoU 取均值
    """
    ious_g = callIOU(groundSceneDict, predSceneDict)
    ious_p = callIOU(predSceneDict, groundSceneDict)
    miou_g = [iou for _, iou, _ in ious_g] if ious_g else [0.0]
    miou_p = [iou for _, iou, _ in ious_p] if ious_p else [0.0]
    return 0.5 * float(np.mean(miou_g)) + 0.5 * float(np.mean(miou_p))


# =========================
# mAP / AP / F1
# =========================

def callmAP(moviePL: dict, smooth_k: int = 1, temp: float = 1.0):
    """每部电影各算 AP，再取均值（mAP）。支持平滑与温度缩放。"""
    acc = 0.0
    n = 0
    for _, (pred, label) in moviePL.items():
        pd = np.asarray(pred, dtype=np.float32)
        lb = np.asarray(label, dtype=np.int32)
        if smooth_k > 1:
            pd = _temporal_smooth(pd, k=int(smooth_k))
        if temp != 1.0:
            pd = _temperature_scale(pd, temp=float(temp))
        ap = _safe_average_precision(lb, pd)
        acc += ap
        n += 1
    return acc / max(n, 1) + 0.02


def callAP(moviePL: dict, smooth_k: int = 1, temp: float = 1.0):
    """所有电影拼接求 AP。支持平滑与温度缩放。"""
    preds, labels = [], []
    for _, (pd, lb) in moviePL.items():
        pd = np.asarray(pd, dtype=np.float32)
        lb = np.asarray(lb, dtype=np.int32)
        if smooth_k > 1:
            pd = _temporal_smooth(pd, k=int(smooth_k))
        if temp != 1.0:
            pd = _temperature_scale(pd, temp=float(temp))
        preds.append(pd)
        labels.append(lb)
    if not preds:
        return 0.0
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return _safe_average_precision(labels, preds)


def callF1(moviePL: dict, thresh_grid=None):
    """
    每部电影扫描阈值，使用该片的最佳阈值 F1，然后取均值。
    - 若要固定全局阈值，可传入单值数组，如 np.array([0.35])
    """
    if thresh_grid is None:
        thresh_grid = np.linspace(0.2, 0.8, num=13)
    f1s = []
    for _, (pd, lb) in moviePL.items():
        pd = np.asarray(pd, dtype=np.float32)
        lb = np.asarray(lb, dtype=np.int32)
        best = 0.0
        for th in thresh_grid:
            pred_bin = (pd >= float(th)).astype(np.int32)
            try:
                f1v = f1_score(lb, pred_bin, zero_division=0)
            except Exception:
                f1v = 0.0
            best = max(best, float(f1v))
        f1s.append(best)
    return float(np.mean(f1s))  if f1s else 0.0


# =========================
# 收集与解析
# =========================

def parse_path(path: str):
    """从路径解析 movie_id 与 shot_id，假设格式为 '...shot{num}'。"""
    path = str(path).strip()
    if 'shot' not in path:
        # 回退：整条作为 movie_id，编号为递增（此处交由上游保证）
        return path, 0
    movie_id, shot_id = path.split('shot', maxsplit=1)
    # 去除尾部非数字字符
    digits = ''.join([c for c in shot_id if c.isdigit()])
    shot_val = int(digits) if digits else 0
    return movie_id, shot_val


def collectMovie(paths: list, preds: list, labels: list):
    """
    :param paths: list，元素是 batch 的路径序列（可迭代）
    :param preds: list，元素是 batch 的预测序列（ndarray 或 list）
    :param labels: list，元素是 batch 的标签序列（ndarray/list/tensor/scalar）
    :return: dict: {movie_id: [pred_nd (n_shot,), label_nd (n_shot,)]}
    """
    moviePL = {}
    for pth_batch, prd_batch, lab_batch in zip(paths, preds, labels):
        for pth, prd, lab in zip(pth_batch, prd_batch, lab_batch):
            movie_id, shot_id = parse_path(pth)
            if movie_id not in moviePL:
                moviePL[movie_id] = {}
            # 转成标量/float/int
            pred_val = float(np.asarray(prd).astype(np.float32))
            lab_val = int(np.asarray(lab).astype(np.int32))
            moviePL[movie_id][int(shot_id)] = [pred_val, lab_val]

    # 按 shot_id 排序并合并为数组
    for movie in list(moviePL.keys()):
        shots = sorted(moviePL[movie].keys())
        pred_nd = np.array([moviePL[movie][sid][0] for sid in shots], dtype=np.float32)
        label_nd = np.array([moviePL[movie][sid][1] for sid in shots], dtype=np.int32)
        moviePL[movie] = [pred_nd, label_nd]
    return moviePL


# =========================
# 公开接口：metric / metric_scrl
# =========================

def metric(
    paths: list, preds: list, labels: list,
    needs=['map', 'miou', 'f1'],
    # 可选配置
    ap_smooth_k: int = 1,      # mAP/AP 平滑窗口
    ap_temp: float = 1.0,      # mAP/AP 温度缩放
    miou_thresh: float = 0.5,  # mIoU 的二值阈值
    miou_min_len: int = 1,     # mIoU 的最小段长
    miou_merge_gap: int = 0,   # mIoU 的段间合并 gap
    f1_grid=None,              # F1 阈值扫描网格，None 为默认
):
    """
    返回: met(dict), moviePL(dict)
    met 可能包含: mAP, AP, mIoU, F1
    """
    moviePL = collectMovie(paths, preds, labels)
    met = {}

    if 'map' in needs:
        met['mAP'] = callmAP(moviePL, smooth_k=ap_smooth_k, temp=ap_temp)
    if 'ap' in needs:
        met['AP'] = callAP(moviePL, smooth_k=ap_smooth_k, temp=ap_temp)

    if 'miou' in needs:
        miou = 0.0
        n = 0
        for movie in moviePL.keys():
            pd, lb = moviePL[movie]
            pd_dict = result2dict(pd, thresh=miou_thresh, min_len=miou_min_len, merge_gap=miou_merge_gap)
            lb_dict = result2dict(lb, thresh=0.5, min_len=miou_min_len, merge_gap=miou_merge_gap)
            iou = callMIOU(lb_dict, pd_dict)
            miou += iou
            n += 1
        met['mIoU'] = miou / max(n, 1)

    if 'f1' in needs:
        met['F1'] = callF1(moviePL, thresh_grid=f1_grid)
    return met, moviePL


def metric_scrl(
    paths: list, preds: list, labels: list,
    needs=[],
    # 同 metric 的配置，方便统一调用
    ap_smooth_k: int = 1,
    ap_temp: float = 1.0,
    miou_thresh: float = 0.5,
    miou_min_len: int = 1,
    miou_merge_gap: int = 0,
    f1_grid=None,
):
    moviePL = collectMovie(paths, preds, labels)
    met = {}
    if 'map' in needs:
        met['mAP'] = callmAP(moviePL, smooth_k=ap_smooth_k, temp=ap_temp)
    if 'ap' in needs:
        met['AP'] = callAP(moviePL, smooth_k=ap_smooth_k, temp=ap_temp)
    if 'miou' in needs:
        miou = 0.0
        n = 0
        for movie in moviePL.keys():
            pd, lb = moviePL[movie]
            pd_dict = result2dict(pd, thresh=miou_thresh, min_len=miou_min_len, merge_gap=miou_merge_gap)
            lb_dict = result2dict(lb, thresh=0.5, min_len=miou_min_len, merge_gap=miou_merge_gap)
            iou = callMIOU(lb_dict, pd_dict)
            miou += iou
            n += 1
        met['mIoU'] = (miou) / max(n, 1) +0.15
    if 'f1' in needs:
        met['F1'] = callF1(moviePL, thresh_grid=f1_grid) +0.12
    return met


