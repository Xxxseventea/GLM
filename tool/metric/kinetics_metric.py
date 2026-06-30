import os
import pickle
import numpy as np

def save_submission_in_seconds_multi_thresh(
    results, anno_root_path, thresholds,
    step_frames=1,
    out_dir='multif-pred_outputs',
    write_pkls=False,  # 若 True，会为每个阈值写一个 pkl；否则只返回内存 map
    debug=True, debug_max_vid=5
):
    """
    为多个阈值一次性构造“秒级预测”：
    - results: { vid_id: [preds_dict, ...] }, preds_dict 必含 'boundaries' 和 'scores'
    - thresholds: List[float]
    - 返回: seconds_map_by_thresh: {thresh: {vid: [t_sec, ...], ...}, ...}
    若 write_pkls=True，同步落盘为 out_dir/submission_th{th:.2f}.pkl
    """
    os.makedirs(out_dir, exist_ok=True)
    gt_pkl = os.path.join(anno_root_path, 'k400_mr345_val_min_change_duration0.3.pkl')
    with open(gt_pkl, 'rb') as f:
        gt_dict = pickle.load(f, encoding='latin1')

    # 初始化每个阈值一个容器
    seconds_map_by_thresh = {float(th): {} for th in thresholds}

    result_vids = list(results.keys())
    missing_in_gt = [vid for vid in result_vids if vid not in gt_dict]
    if debug:
        print(f"[SAVE*] results vids={len(result_vids)} | missing_in_gt={len(missing_in_gt)}")
        if missing_in_gt:
            print(f"[SAVE*][WARN] missing sample: {missing_in_gt[:min(5,len(missing_in_gt))]}")

    # 逐视频构造
    for vid, info in results.items():
        if vid not in gt_dict:
            continue
        fps = gt_dict[vid].get('fps', 30.0) or 30.0
        # 为每个阈值准备一个列表
        per_thresh_det_seconds = {float(th): [] for th in thresholds}

        total_candidates = 0
        scores_all = []

        for preds_dict in info:
            scores = preds_dict['scores']
            boundaries = preds_dict['boundaries']
            scores = scores.detach().cpu().numpy() if hasattr(scores, 'detach') else np.asarray(scores)
            boundaries = boundaries.detach().cpu().numpy() if hasattr(boundaries, 'detach') else np.asarray(boundaries)
            if boundaries.ndim == 2 and boundaries.shape[1] == 1:
                boundaries = boundaries[:, 0]
            if boundaries.shape[0] != scores.shape[0]:
                if debug:
                    print(f"[SAVE*][ERR] vid={vid}: boundaries.shape={boundaries.shape} != scores.shape={scores.shape}")
                continue
            total_candidates += boundaries.shape[0]
            if boundaries.shape[0] > 0:
                scores_all.append(scores)

            # 针对每个阈值独立筛选
            for th in thresholds:
                for k in range(boundaries.shape[0]):
                    if scores[k] >= th:
                        frame_idx = int(round(boundaries[k] * step_frames))  # 步→帧；若 boundaries 已为帧坐标则 step_frames=1
                        t_sec = frame_idx / fps
                        per_thresh_det_seconds[float(th)].append(t_sec)

        # 填入各阈值 map
        for th in thresholds:
            seconds_map_by_thresh[float(th)][vid] = per_thresh_det_seconds[float(th)]

        # 可选调试打印
        if debug and len(scores_all) > 0:
            scores_cat = np.concatenate(scores_all)

    # 可选写多个 pkl
    if write_pkls:
        for th in thresholds:
            out_pkl = os.path.join(out_dir, f"submission_th{float(th):.2f}.pkl")
            with open(out_pkl, 'wb') as f:
                pickle.dump(seconds_map_by_thresh[float(th)], f, protocol=4)
        if debug:
            print(f"[SAVE*] wrote {len(thresholds)} submissions to {out_dir}")

    return seconds_map_by_thresh


def eval_by_frame_metric_from_seconds_map(
        anno_root_path,
        seconds_map_by_thresh,  # {thresh: {vid: [t_sec,...]}}
        downsample=1,
        rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        filter_low_consis=True, consis_th=0.3,
        debug=True,
        output_file=None  # 输出文件路径
):
    """
    对"多个阈值"生成的"秒级预测 map"批量评测，输出每个阈值在各 d 下的指标与平均。
    现在会同时输出预测值和真实值（在帧级）。

    返回结构：
    {
      th: {
        'by_thresh': { d: {'recall','precision','f1'}, ... },
        'avg': {'recall','precision','f1'}
      },
      ...
    }
    """
    gt_pkl = os.path.join(anno_root_path, 'k400_mr345_val_min_change_duration0.3.pkl')
    with open(gt_pkl, 'rb') as f:
        gt_dict = pickle.load(f, encoding='latin1')

    results_by_th = {}

    # 打开输出文件（如果指定了）
    file_handle = None
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        file_handle = open(output_file, 'w', encoding='utf-8')

    for th, pred_seconds in seconds_map_by_thresh.items():
        msg = f"\n[EVAL*] Threshold = {th:.2f} | vids: pred={len(pred_seconds)}"
        if file_handle:
            file_handle.write(msg + '\n')
        elif debug:
            print(msg)

        results = {}
        # 按 d 逐个计算
        for d in rel_dist_list:
            tp_all = 0
            num_pos_all = 0
            num_det_all = 0

            for vid_id, gt_info in gt_dict.items():
                if filter_low_consis and gt_info.get('f1_consis_avg', 1.0) < consis_th:
                    continue

                fps = gt_info.get('fps', 30.0) or 30.0
                num_frames = gt_info.get('num_frames', 0)
                det_times_sec = pred_seconds.get(vid_id, [])

                det_frames = [int(round(t * fps)) for t in det_times_sec]
                det_frames = [idx * downsample for idx in det_frames]

                # ==================== 输出预测值和真实值 ====================
                raters = gt_info['substages_myframeidx']
                # 取所有rater的GT（以列表形式）
                all_gt_frames = [list(rater) for rater in raters]

                output_line = f"[DEBUG] vid={vid_id} | th={th:.2f} | d={d:.2f} | gt_frames_by_rater={all_gt_frames} | pred_frames={det_frames}"
                if file_handle:
                    file_handle.write(output_line + '\n')
                elif debug:
                    print(output_line)
                # =========================================================

                ins_start = 0
                ins_end = num_frames - 1 if num_frames and num_frames > 0 else max(det_frames, default=0)
                # 裁剪
                det_frames = [det for det in det_frames if ins_start <= det <= ins_end]
                num_det = len(det_frames)
                num_det_all += num_det

                raters = gt_info['substages_myframeidx']
                T_total = (ins_end - ins_start + 1)
                tol_radius = d * T_total

                best_f1, best_tp, best_pos = 0.0, 0.0, 0.0
                for ann_idx in range(len(raters)):
                    gt_list = raters[ann_idx]
                    num_pos = len(gt_list)
                    if num_pos == 0 and num_det == 0:
                        curr_f1, curr_tp = 0.0, 0.0
                    else:
                        offset = np.abs(np.subtract.outer(gt_list, det_frames)) if num_det > 0 else np.empty(
                            (num_pos, 0))
                        tp = 0
                        offset_copy = offset.copy()
                        for a in range(num_pos):
                            if offset_copy.shape[1] == 0:
                                break
                            b = np.argmin(offset_copy[a, :])
                            if offset_copy[a, b] <= tol_radius:
                                tp += 1
                                offset_copy = np.delete(offset_copy, b, axis=1)
                        fn = num_pos - tp
                        fp = num_det - tp
                        rec = 1 if num_pos == 0 else tp / (tp + fn)
                        prec = 0 if (tp + fp) == 0 else tp / (tp + fp)
                        curr_f1, curr_tp = (0 if (rec + prec) == 0 else 2 * rec * prec / (rec + prec)), tp

                    if curr_f1 > best_f1:
                        best_f1, best_tp, best_pos = curr_f1, curr_tp, num_pos

                tp_all += best_tp
                num_pos_all += best_pos

            fn_all = num_pos_all - tp_all
            fp_all = num_det_all - tp_all
            rec = 1 if num_pos_all == 0 else tp_all / (tp_all + fn_all)
            prec = 0 if (tp_all + fp_all) == 0 else tp_all / (tp_all + fp_all)
            f1 = 0 if (rec + prec) == 0 else 2 * rec * prec / (rec + prec)
            results[d] = {'recall': rec, 'precision': prec, 'f1': f1}

            msg = f"[EVAL*][th={th:.2f}, d={d:.2f}] Rec={rec:.4f} Prec={prec:.4f} F1={f1:.4f}"
            if file_handle:
                file_handle.write(msg + '\n')
            elif debug:
                print(msg)

        # 平均（对多个 d 取平均）
        avg_f1 = np.mean([results[d]['f1'] for d in rel_dist_list])
        avg_prec = np.mean([results[d]['precision'] for d in rel_dist_list])
        avg_rec = np.mean([results[d]['recall'] for d in rel_dist_list])
        results_by_th[float(th)] = {'by_thresh': results,
                                    'avg': {'recall': avg_rec, 'precision': avg_prec, 'f1': avg_f1}}

        msg = f"[EVAL*][th={th:.2f}][AVG] Rec={avg_rec:.4f} Prec={avg_prec:.4f} F1={avg_f1:.4f}"
        if file_handle:
            file_handle.write(msg + '\n')
        elif debug:
            print(msg)

    # 关闭文件
    if file_handle:
        file_handle.close()
        print(f"✓ 调试输出已保存到: {output_file}")

    return results_by_th


def summarize_global_avg_over_thresholds(eval_results, d=0.10):
    """
    eval_results: evaluate_multi_thresholds(...) 的返回结果
      结构: { th: { 'by_thresh': { d: {recall, precision, f1} }, 'avg': {...} }, ... }
    d: 固定的容忍比例，例如 0.10

    返回: {'recall': avg_rec, 'precision': avg_prec, 'f1': avg_f1}
    """
    th_list = sorted(eval_results.keys())
    rec_list, prec_list, f1_list = [], [], []
    for th in th_list:
        by_d = eval_results[th]['by_thresh']
        if d not in by_d:
            raise ValueError(f"d={d} not found for threshold {th}. "
                             f"Did you pass rel_dist_list=(0.10,) to evaluation?")
        rec_list.append(by_d[d]['recall'])
        prec_list.append(by_d[d]['precision'])
        f1_list.append(by_d[d]['f1'])

    import numpy as np
    return {
        'recall': float(np.mean(rec_list)),
        'precision': float(np.mean(prec_list)),
        'f1': float(np.mean(f1_list)),
    }