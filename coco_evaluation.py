from __future__ import division, print_function

__author__ = 'andreasveit'
__version__ = '1.3'

# Interface for evaluating with the COCO-Text dataset.

# COCO-Text is a large dataset designed for text detection and recognition.
# This is a Python API that assists in evaluating text detection and recognition
# results on COCO-Text. The format of the COCO-Text annotations is described on
# the project website http://vision.cornell.edu/se3/coco-text/. In addition to
# this evaluation API, please download the COCO-Text tool API, both the COCO
# images and annotations.
# This dataset is based on Microsoft COCO. Please visit http://mscoco.org/
# for more information on COCO, including for the image data, object annotatins
# and caption annotations. 

# The following functions are defined:
#  getDetections - Compute TP, FN and FP
#  evaluateAttribute - Evaluates accuracy for classifying text attributes
#  evaluateTranscription - Evaluates accuracy of transcriptions
#  area, intersect, iou_score, decode, inter - small helper functions
#  printDetailedResults - Prints detailed results as reported in COCO-Text paper

# COCO-Text Evaluation Toolbox.        Version 1.3
# Data, Data API and paper available at: http://vision.cornell.edu/se3/coco-text/
# Code written by Andreas Veit, 2016
# revised by Manuel Rota, 2017
# Licensed under the Simplified BSD License [see bsd.txt]

from copy import copy
import editdistance
import re

from utils import *



# Compute detections
def getDetections(groundtruth, evaluation,
                  imgIds=None, detection_threshold=0.5):
    """
    A box is a match iff the intersection of union score is >= 0.5.
    Params
    ------
    Input dicts have the format of annotation dictionaries
    """
    #parameters

    detectRes = {}
    # results are lists of dicts {gt_id: xxx, eval_id: yyy}
    detectRes['true_positives'] = []
    detectRes['false_negatives'] = []
    detectRes['false_positives'] = []

    # the default is set to evaluate on the validation set
    if imgIds is None:
        imgIds = groundtruth.val

    imgIds = inter(imgIds, evaluation.imgToAnns.keys())\
        if imgIds is not None else imgIds

    for cocoid in imgIds:
        gt_bboxes = groundtruth.imgToAnns[cocoid]\
            if cocoid in groundtruth.imgToAnns else []
        eval_bboxes = copy(evaluation.imgToAnns[cocoid])\
            if cocoid in evaluation.imgToAnns else []

        for gt_box_id in gt_bboxes:
            gt_box = groundtruth.anns[gt_box_id]['bbox']
            max_iou = 0.0
            match = None
            for eval_box_id in eval_bboxes:
                eval_box = evaluation.anns[eval_box_id]['bbox']
                iou = iou_score(gt_box, eval_box)
                if iou >= detection_threshold and iou > max_iou:
                    max_iou = iou
                    match = eval_box_id
            if match is not None:
                detectRes['true_positives'].append(
                    {'gt_id': gt_box_id, 'eval_id': match})
                eval_bboxes.remove(match)
            else:
                detectRes['false_negatives'].append({'gt_id': gt_box_id})
        if len(eval_bboxes) > 0:
            detectRes['false_positives'].extend(
                [{'eval_id': eval_box_id} for eval_box_id in eval_bboxes])

    return detectRes


def evaluateAttribute(groundtruth, evaluation, resultDict, attributes):
    '''
    Input:
    groundtruth_Dict: dict, AnnFile format
    evalDict: dict, AnnFile format
    resultDict: dict, output from getDetections
    attributes : list of strings, attribute categories
    -----
    Output:

    '''
    assert 'utf8_string' not in attributes, 'there is a separate function for utf8_string'

    res = {}
    for attribute in attributes:
        correct = []
        incorrect = []
        for detection in resultDict['true_positives']:
            gt_val = groundtruth.anns[detection['gt_id']][attribute]
            eval_val = evaluation.anns[detection['eval_id']][attribute]
            if gt_val == eval_val:
                correct.append(detection)
            else:
                if gt_val!='na':
                    incorrect.append(detection)
        res[attribute] = {
            'attribute': attribute,
            'correct': len(correct),
            'incorrect': len(incorrect),
            'accuracy': len(correct)*1.0/len(correct+incorrect)
        }
    return res


def evaluateEndToEnd(groundtruth, evaluation,
                     imgIds = None, detection_threshold = 0.5):
    """
    A box is a match iff the intersection of union score is >= 0.5.
    Params
    ------
    Input dicts have the format of annotation dictionaries
    """
    #parameters

    detectRes = {}
    # results are lists of dicts {gt_id: xxx, eval_id: yyy}
    detectRes['true_positives'] = []
    detectRes['false_negatives'] = []
    detectRes['false_positives'] = []

    # the default is set to evaluate on the validation set
    if imgIds == None:
        imgIds = groundtruth.val

    imgIds = inter(imgIds, evaluation.imgToAnns.keys())\
        if imgIds is not None else imgIds

    for cocoid in imgIds:
        gt_bboxes = groundtruth.imgToAnns[cocoid]\
            if cocoid in groundtruth.imgToAnns else []
        eval_bboxes = copy(evaluation.imgToAnns[cocoid])\
            if cocoid in evaluation.imgToAnns else []

        for gt_box_id in gt_bboxes:
            gt_box = groundtruth.anns[gt_box_id]['bbox']
            if 'utf8_string' not in groundtruth.anns[gt_box_id]:
                continue
            gt_val = decode(groundtruth.anns[gt_box_id]['utf8_string'])

            max_iou = 0.0

            match = None
            for eval_box_id in eval_bboxes:
                eval_box = evaluation.anns[eval_box_id]['bbox']
                iou = iou_score(gt_box,eval_box)

                if iou >=detection_threshold and iou > max_iou:
                    max_iou = iou
                    match = eval_box_id
                    if 'utf8_string' in evaluation.anns[eval_box_id]:
                        eval_val = decode(evaluation.anns[eval_box_id]['utf8_string'])
                        if editdistance.eval(gt_val, eval_val) == 0:
                            break
            if match is not None:
                detectRes['true_positives'].append(
                    {'gt_id': gt_box_id, 'eval_id': match})
                eval_bboxes.remove(match)
            else:
                detectRes['false_negatives'].append({'gt_id': gt_box_id})
        if len(eval_bboxes)>0:
            detectRes['false_positives'].extend(
                [{'eval_id': eval_box_id} for eval_box_id in eval_bboxes])

    resultDict = detectRes

    res = {}
    for setting, threshold in zip(['exact', 'distance1'],[0,1]):
        correct = []
        incorrect = []
        ignore = []
        for detection in resultDict['true_positives']:
            if 'utf8_string' not in groundtruth.anns[detection['gt_id']]:
                ignore.append(detection)
                continue

            gt_val = decode(groundtruth.anns[detection['gt_id']]['utf8_string'])
            if len(gt_val)<3:
                ignore.append(detection)
                continue

            if 'utf8_string' not in evaluation.anns[detection['eval_id']]:
                incorrect.append(detection)
                continue

            eval_val = decode(evaluation.anns[detection['eval_id']]['utf8_string'])

            detection['gt_string'] = gt_val
            detection['eval_string'] = eval_val
            if editdistance.eval(gt_val, eval_val)<=threshold:
                correct.append(detection)
            else:
                incorrect.append(detection)

        res[setting] = {
            'setting': setting,
            'correct': correct,
            'incorrect': incorrect,
            'ignore': ignore,
            'accuracy': len(correct)*1.0 / len(correct+incorrect)
        }

    return res


def area(bbox):
    return bbox[2] * bbox[3] # width * height


def intersect(bboxA, bboxB):
    """Return a new bounding box that contains the intersection of
    'self' and 'other', or None if there is no intersection
    """
    new_top = max(bboxA[1], bboxB[1])
    new_left = max(bboxA[0], bboxB[0])
    new_right = min(bboxA[0]+bboxA[2], bboxB[0]+bboxB[2])
    new_bottom = min(bboxA[1]+bboxA[3], bboxB[1]+bboxB[3])
    if new_top < new_bottom and new_left < new_right:
        return [new_left, new_top, new_right - new_left, new_bottom - new_top]
    return None


def iou_score(bboxA, bboxB):
    """Returns the Intersection-over-Union score, defined as the area of
    the intersection divided by the intersection over the union of
    the two bounding boxes. This measure is symmetric.
    """
    if intersect(bboxA, bboxB):
        intersection_area = area(intersect(bboxA, bboxB))
    else:
        intersection_area = 0
    union_area = area(bboxA) + area(bboxB) - intersection_area
    if union_area > 0:
        return float(intersection_area) / float(union_area)
    else:
        return 0


def decode(trans):
    trans = trans.encode("ascii" ,'ignore')
    trans = trans.replace('\n', ' ')
    trans2 = re.sub('[^a-zA-Z0-9!?@\_\-\+\*\:\&\/ \.]', '', trans)
    return trans2.lower()

def printDetailedResults(ct, detection_results, transcription_results, name):
    print(name)
    #detected coco-text annids
    found = [x['gt_id'] for x in detection_results['true_positives']]
    n_found = [x['gt_id'] for x in detection_results['false_negatives']]
    fp = [x['eval_id'] for x in detection_results['false_positives']]

    leg_eng_mp = ct.getAnnIds(catIds=[LEGIBLE, ENGLISH, MACHINE_PRINTED])
    leg_eng_hw = ct.getAnnIds(catIds=[LEGIBLE, ENGLISH, HANDWRITTEN])
    leg_mp  = ct.getAnnIds(catIds=[LEGIBLE, MACHINE_PRINTED])
    ileg_mp = ct.getAnnIds(catIds=[ILLEGIBLE, MACHINE_PRINTED])
    leg_hw  = ct.getAnnIds(catIds=[LEGIBLE, HANDWRITTEN])
    ileg_hw = ct.getAnnIds(catIds=[ILLEGIBLE, HANDWRITTEN])
    leg_ot  = ct.getAnnIds(catIds=[LEGIBLE, OTHERS])
    ileg_ot = ct.getAnnIds(catIds=[ILLEGIBLE, OTHERS])

    if detection_results is not None:
        # Detection results
        print("\nDetection")

        print("Recall")

        def recall(data):
            intersection = inter(found + n_found, data)
            if len(intersection) > 0:
                ret = len(inter(found, data))*1. / len(intersection)
            else:
                ret = 0
            return 100*ret

        print('legible & machine printed: %.2f' % recall(leg_mp))
        print('legible & handwritten: %.2f' % recall(leg_hw))
        print('legible & others: %.2f' % recall(leg_ot))
        print('legible overall: %.2f' % recall(leg_mp + leg_hw + leg_ot))
        print('illegible & machine printed: %.2f' % recall(ileg_mp))
        print('illegible & handwritten: %.2f' % recall(ileg_hw))
        print('illegible & others: %.2f' % recall(ileg_ot))
        print('illegible overall: %.2f'% recall(ileg_mp + ileg_hw + ileg_ot))

        t_recall = recall(leg_mp + leg_hw + ileg_mp + ileg_hw)
        print('total recall: %.2f' % t_recall)


        print("Precision")
        t_precision = len(found)*100.0 / (len(found+fp))
        print('total precision: %.2f' % t_precision)

        print("f-score")
        f_score = (2 * t_recall * t_precision / (t_recall + t_precision))\
            if t_recall + t_precision != 0 else 0
        print('f-score localization: %.2f' % f_score)

    if transcription_results is not None:
        # Transcription results
        print("\nTranscription")

        transAcc = transcription_results['exact']['accuracy']
        transAcc1 = transcription_results['distance1']['accuracy']
        print('accuracy for exact matches: %.2f' % (100*transAcc))
        print('accuracy for matches with edit distance<=1: %.2f' % (100*transAcc1))


        print('\nEnd-to-end')
        TP_new = len(inter(found, leg_eng_mp+leg_eng_hw)) * transAcc
        FP_new = len(fp) + len(inter(found, leg_eng_mp+leg_eng_hw)) * (1-transAcc)
        FN_new = len(inter(n_found, leg_eng_mp+leg_eng_hw))\
                 + len(inter(found, leg_eng_mp+leg_eng_hw))*(1-transAcc)
        t_recall = 100 * TP_new / (TP_new + FN_new)
        t_precision = 100 * TP_new / (TP_new + FP_new)\
            if (TP_new + FP_new) > 0 else 0
        f_score = (2 * t_recall * t_precision / (t_recall + t_precision))\
            if (t_recall + t_precision) > 0 else 0

        print('recall: %.2f' % t_recall)
        print('precision: %.2f' % t_precision)
        print('End-to-end f-score: %.2f' % f_score)
