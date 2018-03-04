
from ._postprocessing import postprocessing as _postprocessing
import numpy as np
import cv2


def _softmax(x):
    x = np.exp(x)
    return x[:, :, 1:2] / (x[:, :, 0:1] + x[:, :, 1:2])


class Postprocessor:
    t_tcl = 0.4
    t_tr = 0.4
    t_delta = 0.1
    t_rad = 0.1
    fewest_tcl_ratio = 0.04
    smallest_area_ratio = 0.5
    smallest_area_ratio_tcl = 0.3
    radius_scaling = 1.30
    fewest_tcl = 3

    def process(self, maps):
        assert len(maps.shape) == 3
        assert maps.shape[2] == 5 or maps.shape[2] == 7
        assert maps.dtype == np.float32

        if maps.shape[2] == 7:
            tcl, geo, tr = np.split(maps, [2, 5], axis=2)
            tcl = _softmax(tcl)
            tr = _softmax(tr)
            maps = np.concatenate([tcl, geo, tr], axis=2)

        return _postprocessing(maps, self)


def evaluate(height, width, cnts, resultcnts, fsk=0.8, tp=0.4, tr=0.8):
    """
    (1) adapted from the official script; however, the official script is defective
    (2) When a long predicted box covers more than one GT boxes, one of which has precision>0.4,
       the official script regrads it as a one-to-one matching. This results in FalseNegatives
       for other GT boxes that are also covered.

    SketchMap

    GT           ooo oo oooooooooo
    predicted    ------------------
                           ^
                           |
                           this GT boxes is fully recalled; for the predicted box, precision is well above 0.4
                           results: the other two GT boxes are regarded as FN.

    :param height:
    :param width:
    :param cnts:
    :param resultcnts:
    :param fsk:
    :param tp:
    :param tr:
    :return:
    """
    def drawcontour(cnt):
        img = np.zeros((height, width, 1), dtype=np.uint8)
        cv2.drawContours(img, [cnt], -1, 255, -1)
        return img
    cnts_mask = [drawcontour(cnt) for cnt in cnts]
    merged_re_cnts_mask = [drawcontour(cnt) for cnt in resultcnts]
    cnts_num = len(cnts)
    re_cnts_num = len(resultcnts)

    precise = np.zeros((cnts_num, re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            precise[i, j] = np.sum(
                cnts_mask[i] & merged_re_cnts_mask[j]) / np.sum(merged_re_cnts_mask[j])

    recall = np.zeros((cnts_num, re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            recall[i, j] = np.sum(
                cnts_mask[i] & merged_re_cnts_mask[j]) / np.sum(cnts_mask[i])

    IOU = np.zeros((cnts_num, re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            IOU[i, j] = np.sum(cnts_mask[i] & merged_re_cnts_mask[j]) / \
                np.sum(cnts_mask[i] | merged_re_cnts_mask[j])

    # print(precise, recall, IOU)

    gt_score = np.zeros((cnts_num), np.float32)
    pred_score = np.zeros((re_cnts_num), np.float32)
    flag_gt = np.zeros((cnts_num), np.int32)
    flag_pred = np.zeros((re_cnts_num), np.int32)

    # one to one
    for i in range(cnts_num):
        match_r_num = np.sum(recall[i, :] >= tr)
        match_p_num = np.sum(precise[i, :] >= tp)
        if match_p_num == 1 and match_r_num == 1:
            gt_score[i] = 1.0
            flag_gt[i] = 1
            j = int(np.argwhere(precise[i, :] >= tp))
            # j_ = int(np.argwhere(recall[i,:]>tr))
            # if j != j_:
            #     print('i',i,'j',j,'j_', j_)
            #     print('precise', precise)
            #     print('recall', recall)
            pred_score[j] = 1.0
            flag_pred[j] = 1

    # one to many
    for i in range(cnts_num):
        if flag_gt[i] > 0:
            continue
        index_list = []
        for j in range(re_cnts_num):
            if precise[i, j] >= tp and flag_pred[j] == 0:
                index_list.append(j)
        r_sum = 0.0
        for j in index_list:
            r_sum += recall[i, j]
        if r_sum >= tr:
            if len(index_list) > 1:
                gt_score[i] = fsk
                flag_gt[i] = 1
                for j in index_list:
                    pred_score[j] = fsk
                    flag_pred[j] = 1
            else:
                gt_score[i] = 1.0
                flag_gt[i] = 1
                pred_score[index_list[0]] = 1.0
                flag_pred[index_list[0]] = 1

    # many to one
    for j in range(re_cnts_num):
        if flag_pred[j] > 0:
            continue
        index_list = []
        for i in range(cnts_num):
            if recall[i, j] >= tr and flag_gt[i] == 0:
                index_list.append(i)
        p_sum = 0.0
        for i in index_list:
            p_sum += precise[i, j]
        if p_sum >= tp:
            if len(index_list) > 1:
                pred_score[j] = fsk
                flag_pred[j] = 1
                for i in index_list:
                    gt_score[i] = fsk
                    flag_gt[i] = 1
            else:
                pred_score[j] = 1.0
                flag_pred[j] = 1
                gt_score[index_list[0]] = 1.0
                flag_gt[index_list[0]] = 1

    STR, STP = np.sum(gt_score), np.sum(pred_score)
    TR = np.sum(gt_score) / cnts_num
    TP = np.sum(pred_score) / re_cnts_num

    pascal_gt_score = np.zeros((cnts_num), np.float32)
    pascal_pred_score = np.zeros((re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            if IOU[i, j] >= 0.5:
                if pascal_gt_score[i] < 0.5 and pascal_pred_score[j] < 0.5:
                    pascal_pred_score[j] = 1.0
                    pascal_gt_score[i] = 1.0
    SPR, SPP = np.sum(pascal_gt_score), np.sum(pascal_pred_score)
    PR = np.sum(pascal_gt_score) / cnts_num
    PP = np.sum(pascal_pred_score) / re_cnts_num

    # print("TR: {:07.05f}, TP: {:07.05f}, PR: {:07.05f}, PP: {:07.05f}".format(
    #     TR, TP, PR, PP))
    return TR, TP, PR, PP, precise, recall


def evaluate_new(height, width, cnts, resultcnts, fsk=0.8, tp=0.4, tr=0.8):
    """
    This evaluation script amends the problems mentioned above, resulting in higher P/R/F performance.

    :param height:
    :param width:
    :param cnts:
    :param resultcnts:
    :return:
    """
    def drawcontour(cnt):
        img = np.zeros((height, width, 1), dtype=np.uint8)
        cv2.drawContours(img, [cnt], -1, 255, -1)
        return img
    cnts_mask = [drawcontour(cnt) for cnt in cnts]
    merged_re_cnts_mask = [drawcontour(cnt) for cnt in resultcnts]
    cnts_num = len(cnts)
    re_cnts_num = len(resultcnts)

    precise = np.zeros((cnts_num, re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            precise[i, j] = np.sum(
                cnts_mask[i] & merged_re_cnts_mask[j]) / np.sum(merged_re_cnts_mask[j])

    recall = np.zeros((cnts_num, re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            recall[i, j] = np.sum(
                cnts_mask[i] & merged_re_cnts_mask[j]) / np.sum(cnts_mask[i])

    IOU = np.zeros((cnts_num, re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            IOU[i, j] = np.sum(cnts_mask[i] & merged_re_cnts_mask[j]) / \
                np.sum(cnts_mask[i] | merged_re_cnts_mask[j])

    # print(precise, recall, IOU)

    gt_score = np.zeros((cnts_num), np.float32)
    pred_score = np.zeros((re_cnts_num), np.float32)
    flag_gt = np.zeros((cnts_num), np.int32)
    flag_pred = np.zeros((re_cnts_num), np.int32)

    # one to many
    for i in range(cnts_num):
        if flag_gt[i] > 0:
            continue
        index_list = []
        for j in range(re_cnts_num):
            if precise[i, j] >= tp and flag_pred[j] == 0:
                index_list.append(j)
        r_sum = 0.0
        for j in index_list:
            r_sum += recall[i, j]
        if r_sum >= tr:
            if len(index_list) > 1:
                gt_score[i] = fsk
                flag_gt[i] = 1
                for j in index_list:
                    pred_score[j] = fsk
                    flag_pred[j] = 1
            # else:
            #     gt_score[i] = 1.0
            #     flag_gt[i] = 1
            #     for j in index_list:
            #         pred_score[j] = 1.0
            #         flag_pred[j] = 1
    # many to one
    for j in range(re_cnts_num):
        if flag_pred[j] > 0:
            continue
        index_list = []
        for i in range(cnts_num):
            if recall[i, j] >= tr and flag_gt[i] == 0:
                index_list.append(i)
        p_sum = 0.0
        for i in index_list:
            p_sum += precise[i, j]
        if p_sum >= tp:
            if len(index_list) > 1:
                pred_score[j] = fsk
                flag_pred[j] = 1
                for i in index_list:
                    gt_score[i] = fsk
                    flag_gt[i] = 1
            # else:
            #     pred_score[j] = 1.0
            #     flag_pred[j] = 1
            #     for i in index_list:

    for i in range(cnts_num):
        for j in range(re_cnts_num):
            if flag_gt[i] == 0 and flag_pred[j] == 0:
                if precise[i, j] > tp and recall[i, j] > tr:
                    gt_score[i] = 1.0
                    pred_score[j] = 1.0

    STR, STP = np.sum(gt_score), np.sum(pred_score)
    TR = np.sum(gt_score) / cnts_num
    TP = np.sum(pred_score) / re_cnts_num

    pascal_gt_score = np.zeros((cnts_num), np.float32)
    pascal_pred_score = np.zeros((re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            if IOU[i, j] >= 0.5:
                if pascal_gt_score[i] < 0.5 and pascal_pred_score[j] < 0.5:
                    pascal_pred_score[j] = 1.0
                    pascal_gt_score[i] = 1.0
    SPR, SPP = np.sum(pascal_gt_score), np.sum(pascal_pred_score)
    PR = np.sum(pascal_gt_score) / cnts_num
    PP = np.sum(pascal_pred_score) / re_cnts_num

    # print("TR: {:07.05f}, TP: {:07.05f}, PR: {:07.05f}, PP: {:07.05f}".format(
    #     TR, TP, PR, PP))
    return TR, TP, PR, PP, precise, recall  # Attention: newly added outputs


__all__ = ['Postprocessor', 'evaluate', 'evaluate_new']
