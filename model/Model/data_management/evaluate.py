# This script should not be placed under data_management. to be moved

import numpy as np
import cv2
from . import postprocessing
from .utils import get_maps

VIZ_DIR = '/home/rjq/data_cleaned/viz/'
processor = postprocessing.Postprocessor()


def get_l2_dist(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def _find(Index, Sets):
    for i, Set in enumerate(Sets):
        if Index in Set:
            return i
    return -1


def _intersection(re_mask_list):
    score = {}
    for i in range(len(re_mask_list)):
        for j in range(i, len(re_mask_list)):
            score[(i, j)] = max(np.sum(re_mask_list[i] & re_mask_list[j]) / np.sum(re_mask_list[j]),
                                np.sum(re_mask_list[i] & re_mask_list[j]) / np.sum(re_mask_list[i]))
    return score


def evaluate(img, cnts, is_text_cnts, maps, is_viz,
             save_name=None, TCL_threshold=0.5, TR_threshold=0.5, fsk=0.8, tp=0.4, tr=0.8, merge_th=0.1, mode='evaluation',NewEval=False):
    '''
    :param img: ndarrray, np.uint8,
    :param cnts:
        if is_text_cnts is True: list(ndarray), ndarray: dtype np.float32, shape [n, 1, 2]
        if is_text_cnts is False: list(list(ndarray), list(ndarray)), for [char_cnts, text_cnts]
    :param is_text_cnts: bool
    :param maps:
        maps: [neg-TCL, TCL, radius, cos_theta, sin_theta, neg-TR, TR], all of them are 2-d array,
        TR: np.bool; TCL: np.bool; radius: np.float32; cos_theta/sin_theta: np.float32
    :param is_viz: bool
    :param save_name
           if is_viz is True, save_name is used to save viz pics
    :param mode: 'evaluation' for loss cal + p/r/f calculation
                 'prediction' for prediction output only
    :return:

    '''
    if not is_text_cnts:
        char_cnts, text_cnts = cnts
        cnts = text_cnts

    assert img.shape[:2] == maps.shape[:2]

    # pick out instance TCL from cropped_TCL map

    reconstructed_cnts = processor.process(maps)

    viz = np.zeros(img.shape, np.uint8)
    cnts = [np.array(cnt, np.int32) for cnt in cnts]
    viz = cv2.drawContours(viz, cnts, -1, (255, 255, 255), 1)
    viz = cv2.drawContours(viz, reconstructed_cnts, -1, (255, 0, 0), 1)

    totaltext_recall, totaltext_precision, \
        pascal_recall, pascal_precision, precise, \
        recall = {False:postprocessing.evaluate, True:postprocessing.evaluate_new}[NewEval](*img.shape[:2], cnts, reconstructed_cnts, fsk=fsk, tp=tp, tr=tr)

    return totaltext_recall, totaltext_precision, \
        pascal_recall, pascal_precision, viz, precise, recall


if __name__ == '__main__':
    pass
