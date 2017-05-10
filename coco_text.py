from __future__ import division, print_function

__author__ = 'andreasveit'
__version__ = '1.1'
# Interface for accessing the COCO-Text dataset.

# COCO-Text is a large dataset designed for text detection and recognition.
# This is a Python API that assists in loading, parsing and visualizing the 
# annotations. The format of the COCO-Text annotations is also described on 
# the project website http://vision.cornell.edu/se3/coco-text/.
# In addition to this API, please download both the COCO images and annotations.
# This dataset is based on Microsoft COCO. Please visit http://mscoco.org/
# for more information on COCO, including for the image data, object annotations
# and caption annotations. 

# An alternative to using the API is to load the annotations directly
# into Python dictionary:
# with open(annotation_filename) as json_file:
#     coco_text = json.load(json_file)
# Using the API provides additional utility functions.

# The following API functions are defined:
#  COCO_Text  - COCO-Text api class that loads COCO annotations and prepare data structures.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.

# COCO-Text Toolbox.        Version 1.1
# Data and  paper available at:  http://vision.cornell.edu/se3/coco-text/
# Code based on Microsoft COCO Toolbox Version 1.0 by Piotr Dollar and Tsung-Yi Lin
# extended and adapted by Andreas Veit, 2016
# revised by Manuel Rota, 2017
# Licensed under the Simplified BSD License [see bsd.txt]

import copy
import datetime
import json
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path

from utils import inter



class COCO_Text:
    def __init__(self, annotation_file=None):
        """
        Constructor of COCO-Text helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = {}
        self.imgToAnns = {}
        self.imgs = {}
        self.cats = {}
        self.val = []
        self.test = []
        self.train = []
        if annotation_file is not None:
            assert(os.path.isfile(annotation_file), "file does not exist")
            print('loading annotations into memory...')
            time_t = datetime.datetime.utcnow()
            self.dataset = json.load(open(annotation_file, 'r'))
            print(datetime.datetime.utcnow() - time_t)
            self.createIndex()

    def createIndex(self):
        """
        Create index to simplify dataset access
        :return:
        """
        print('creating index...')
        self.cats = self.dataset['cats']
        self.imgToAnns = {
            int(cocoid): self.dataset['imgToAnns'][cocoid]
            for cocoid in self.dataset['imgToAnns']
        }
        self.imgs = {
            int(cocoid): self.dataset['imgs'][cocoid]
            for cocoid in self.dataset['imgs']
        }
        self.anns = {
            int(annid): self.dataset['anns'][annid]
            for annid in self.dataset['anns']
        }
        self.val = [
            int(cocoid) for cocoid in self.dataset['imgs']
            if self.dataset['imgs'][cocoid]['set'] == 'val'
        ]
        self.test = [
            int(cocoid) for cocoid in self.dataset['imgs']
            if self.dataset['imgs'][cocoid]['set'] == 'test'
        ]
        self.train = [
            int(cocoid) for cocoid in self.dataset['imgs']
            if self.dataset['imgs'][cocoid]['set'] == 'train'
        ]
        print('index created!')

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('%s: %s' % (key, value))

    def getAnnByCat(self, properties):
        """
        Get ann ids that satisfy given properties
        :param properties (list of tuples of the form [(category type, category)] e.g., [('readability','readable')]
            : get anns for given categories - anns have to satisfy all given property tuples
        :return: ids (int array)       : integer array of ann ids
        """
        return [
            annId for annId in self.anns
            if all([self.anns[annId][category] == value
                    for (category, value) in properties])
        ]

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[]):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds (int array)
               catIds: list of tuples of the form [(category, type)]
               areaRng (float array)
        :return: ids (int array)
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.anns.keys()
        else:
            if len(imgIds) > 0:
                anns = sum(
                    [self.imgToAnns[imgId]
                     for imgId in imgIds if imgId in self.imgToAnns],
                    [])
            else:
                anns = self.anns.keys()

            if len(catIds) > 0:
                anns = inter(anns, self.getAnnByCat(catIds))
            if len(areaRng) == 2:
                anns = [
                    ann for ann in anns
                    if self.anns[ann]['area'] > areaRng[0]
                        and self.anns[ann]['area'] < areaRng[1]
                ]

        return anns

    def getImgIds(self, imgIds=[], catIds=[]):
        """
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            if len(catIds) > 0:
                ids = inter(ids, [self.anns[annid]['image_id']
                                  for annid in self.getAnnByCat(catIds)])

        return ids


    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadImgs(self, ids=[]):
        """
        Load images with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]


    def showAnns(self, anns, show_polygon=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return
        ax = plt.gca()
        boxes = []
        color = []
        for ann in anns:
            c = np.random.random((1, 3)).tolist()[0]
            if show_polygon:
                tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = ann['polygon']
                verts = [
                    (tl_x, tl_y), (tr_x, tr_y),
                    (br_x, br_y), (bl_x, bl_y), (0, 0)
                ]
                codes = [
                    Path.MOVETO, Path.LINETO,
                    Path.LINETO, Path.LINETO, Path.CLOSEPOLY
                ]
                path = Path(verts, codes)
                patch = PathPatch(path, facecolor='none')
                boxes.append(patch)
                left, top = tl_x, tl_y
            else:
                left, top, width, height = ann['bbox']
                boxes.append(Rectangle([left,top],width,height,alpha=0.4))
            color.append(c)
            if 'utf8_string' in ann.keys():
                ax.annotate(ann['utf8_string'],(left,top-4),color=c)
        p = PatchCollection(
            boxes, facecolors=color, edgecolors=(0,0,0,1),
            linewidths=3, alpha=0.4)
        ax.add_collection(p)

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO_Text()
        res.dataset['imgs'] = [img for img in self.dataset['imgs']]

        print('Loading and preparing results...')
        time_t = datetime.datetime.utcnow()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        else:
            anns = resFile
        assert(type(anns) == list, 'results in not an array of objects')
        annsImgIds = [int(ann['image_id']) for ann in anns]

        set_given = set(annsImgIds)
        set_inter = set_given & set(self.getImgIds())
        if set_given != set_inter:
            print('Results do not correspond to current coco set')
            print('skipping %d images' % (len(set_given) - len(set_inter)))
        annsImgIds = list(set_inter)

        res.imgToAnns = {cocoid : [] for cocoid in annsImgIds}
        res.imgs = {cocoid: self.imgs[cocoid] for cocoid in annsImgIds} 

        for id, ann in enumerate(anns):
            if ann['image_id'] not in annsImgIds:
                continue
            bb = ann['bbox']
            ann['area'] = bb[2]*bb[3]
            ann['id'] = str(id)
            res.anns[id] = ann
            res.imgToAnns[ann['image_id']].append(id)
        print('DONE (t=%0.2fs)' % (datetime.datetime.utcnow() - time_t).total_seconds())

        return res
