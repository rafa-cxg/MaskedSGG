"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import logging
from PIL import Image
from detectron2.data import transforms, MetadataCatalog
from matplotlib import pyplot as plt

from detectron2.utils.visualizer import Visualizer
from lavis.datasets.datasets.base_dataset import BaseDataset
import json
import os
import random
from collections import defaultdict, OrderedDict, Counter
import pickle
import h5py
import numpy as np
import torch
from PIL import Image,ImageFile
from tqdm import tqdm
from lavis.common.dist_utils import get_rank
from torch.utils.data.dataloader import default_collate
from pysgg.structures.bounding_box import BoxList


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

BOX_SCALE = 1024  # Scale at which we have the boxes
ImageFile.LOAD_TRUNCATED_IMAGES = True
HEAD = [31, 20, 48, 30]
BODY = [22, 29, 50, 8, 21, 1, 49, 40, 43, 23, 38, 41]
TAIL = [6, 7, 46, 11, 33, 16, 9, 25, 47, 19, 35, 24, 5, 14, 13, 10, 44, 4, 12, 36, 32, 42, 26, 28, 45, 2, 17, 3, 18, 34,
        37, 27, 39, 15]
class VGSGGDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, vis_processor, text_processor, vis_root, ann_paths, split, transforms=None,
                 filter_empty_rels=True, num_im=-1, num_val_im=5000, check_img_file=False,
                 filter_duplicate_rels=True, filter_non_overlap=True, custom_eval=False, custom_path=''):

        """
               Torch dataset for VisualGenome
               Parameters:
                   split: Must be train, test, or val
                   img_dir: folder containing all vg images
                   roidb_file:  HDF5 containing the GT boxes, classes, and relationships
                   dict_file: JSON Contains mapping of classes/relationships to words
                   image_file: HDF5 containing image filenames:image,json
                   filter_empty_rels: True if we filter out images without relationships between
                                    boxes. One might want to set this to false if training a detector.
                   filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
                   num_im: Number of images in the entire dataset. -1 for all images.
                   num_val_im: Number of images in the validation set (must be less than num_im
                      unless num_im is -1.)
               """
        self.vis_processor = vis_processor
        self.roidb_file=ann_paths[0]
        self.dict_file=ann_paths[1]
        self.img_dir=vis_root
        self.image_file =ann_paths[2]
        # for debug
        cfg.DEBUG = True
        if cfg.DEBUG:
            num_im = 1000
            num_val_im = 400
        #
        # num_im = 20000
        # num_val_im = 1000

        assert split in {'train', 'eval', 'test'}
        self.split = split
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'  # 只限定train，所以不担心test受到这两个参数影响
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.repeat_dict = None
        self.check_img_file = check_img_file
        # self.remove_tail_classes = False

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(
            self.dict_file)  # contiguous 151, 51 containing __background__
        with open("ind_to_classes", 'w') as f:
            json.dump(self.ind_to_classes, f)
        with open("ind_to_predicates", 'w') as f:
            json.dump(self.ind_to_predicates, f)

        logger = logging.getLogger("pysgg.dataset")
        self.logger = logger

        self.categories = {i: self.ind_to_classes[i]
                           for i in range(len(self.ind_to_classes))}
        self.custom_eval = custom_eval
        if self.custom_eval:
            self.get_custom_imgs(custom_path)
        else:
            self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
                # 读取h4文件，把gt导入变量中.此时不包括transform和论文中采样method
                self.roidb_file, self.split, num_im, num_val_im=num_val_im,
                filter_empty_rels=False if not cfg.RELATION_ON and split == "train" else True,
                filter_non_overlap=self.filter_non_overlap,
            )

            self.filenames, self.img_info = load_image_filenames(
                self.img_dir, self.image_file, self.check_img_file)  # length equals to split_mask. filenames包括整个数据集的图片
            self.filenames = [self.filenames[i]  # 现在只剩下过滤后的图片了
                              for i in np.where(self.split_mask)[
                                  0]]  # list:57723. like 'datasets/vg/stanford_spilt/VG_100k_images/498334.jpg'
            self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]
            self.idx_list = list(range(len(self.filenames)))

            self.id_to_img_map = {k: v for k, v in enumerate(self.idx_list)}
            self.pre_compute_bbox = None



        # if cfg.MODEL.ROI_RELATION_HEAD.REMOVE_TAIL_CLASSES and self.split == 'train':
        #     self.remove_tail_classes = True

    def __getitem__(self, index):
        # if self.split == 'train':
        #    while(random.random() > self.img_info[index]['anti_prop']):
        #        index = int(random.random() * len(self.filenames))
        if self.custom_eval:
            img = Image.open(self.custom_files[index]).convert("RGB")
            target = torch.LongTensor([-1])
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, index
        if self.repeat_dict is not None:
            index = self.idx_list[index]  # list:165908[0,0,1,1..]类似对数据集的照片index做了重定向（重复index的内容来增加图像数目）

        img = Image.open(self.filenames[index]).convert("RGB")
        # plt.imshow(img)
        # plt.show()
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']),
                  ' ', str(self.img_info[index]['height']), ' ', '=' * 20)

        target = self.get_groundtruth(index, flip_img=False)  # flip是假的
        # todo add pre-compute boxes
        pre_compute_boxlist = None
        if self.pre_compute_bbox is not None:
            # index by image id
            pre_comp_result = self.pre_compute_bbox[int(
                self.img_info[index]['image_id'])]
            boxes_arr = torch.as_tensor(pre_comp_result['bbox']).reshape(-1, 4)
            pre_compute_boxlist = BoxList(boxes_arr, img.size, mode='xyxy')
            pre_compute_boxlist.add_field(
                "pred_scores", torch.as_tensor(pre_comp_result['scores']))
            pre_compute_boxlist.add_field(
                'pred_labels', torch.as_tensor(pre_comp_result['cls']))


        img,target,low_res_img,low_res_target = self.vis_processor(img,target)# low_resulution_img for relation
        #图像和标注可视化
        # plt.imshow(img.permute(1,2,0))
        # plt.show()
        # self.metadata = MetadataCatalog.get(
        #     "visual_genome")
        # self.metadata.thing_classes=self.ind_to_classes
        # vis = Visualizer(img.permute(1,2,0)*255,scale=3,metadata=self.metadata) #metadata
        # htarget=conver_boxlist_to_instance(target)
        # vis_pred = vis.draw_instance_predictions(htarget).get_image()
        # plt.imshow(vis_pred)
        # plt.show()
        # vis = Visualizer(low_res_img.permute(1, 2, 0) * 255, scale=3, metadata=self.metadata)  # metadata
        # low_target = conver_boxlist_to_instance(low_res_target)
        # vis_pred = vis.draw_instance_predictions(low_target).get_image()
        # plt.imshow(vis_pred)
        # plt.show()

        # todo RandomHorizontalFlip(p=0.5) 应该去掉，否则影响SGG

        return {"image": img,"target":target,"low_res_img": low_res_img,"low_res_target":low_res_target,"index":index,"filename":self.filenames[index]}


    def __len__(self):
        if self.custom_eval:
            return len(self.custom_files)
        return len(self.idx_list)

    def collater(self, samples):

        img, labels, low_res_img, low_res_target, index, filename = [], [], [], [], [], []

        for i in samples:
            img.append(i['image'])
            labels.append(i['target'])
            low_res_img.append(i['low_res_img'])
            low_res_target.append(i['low_res_target'])
            index.append(i['index'])
            filename.append(i['filename'])

        samples = {}
        samples['image'] = img
        samples['labels'] = labels
        samples['low_res_img'] = low_res_img
        samples['low_res_target'] = low_res_target
        samples['index'] = index
        samples['filename'] = filename



        return samples
    def get_groundtruth(self, index, evaluation=False, flip_img=False, inner_idx=True):
        if not inner_idx:
            # here, if we pass the index after resampeling, we need to map back to the initial index
            if self.repeat_dict is not None:
                index = self.idx_list[index]

        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))
        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v))
                        for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        relation_non_masked = None

        # add relation to target
        num_box = len(target)
        relation_map_non_masked = None
        if self.repeat_dict is not None:
            relation_map_non_masked = torch.zeros((num_box, num_box), dtype=torch.long)


        relation_map = torch.zeros((num_box, num_box), dtype=torch.long)
        for i in range(relation.shape[0]):
            # Sometimes two objects may have multiple different ground-truth predicates in VisualGenome.
            # In this case, when we construct GT annotations, random selection allows later predicates
            # having the chance to overwrite the precious collided predicate.
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] != 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                    if relation_map_non_masked is not None  :
                        relation_map_non_masked[int(relation_non_masked[i, 0]),
                                                int(relation_non_masked[i, 1])] = int(relation_non_masked[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                if relation_map_non_masked is not None  :
                    relation_map_non_masked[int(relation_non_masked[i, 0]),
                                            int(relation_non_masked[i, 1])] = int(relation_non_masked[i, 2])


        target.add_field("relation", relation_map, is_triplet=True)
        if relation_map_non_masked is not None :
             target.add_field("relation_non_masked", relation_map_non_masked.long(), is_triplet=True)


        target = target.clip_to_image(remove_empty=False)
        target.add_field("relation_tuple", torch.LongTensor(
                relation))  # for evaluation
        return target
def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(
        predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(
        attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes
def load_image_filenames(img_dir, image_file, check_img_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        basename = '{}.jpg'.format(img['image_id'])
        # print(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        # if os.path.exists(filename) or not check_img_file:# commit for faster. we do not know consequence
        fns.append(filename)
        img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info
def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]#区分test和train的划分
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]#108073->62723!这是整个数据集大小
    if num_im > -1:
        image_index = image_index[: num_im]
    if num_val_im > 0:
        if split == 'eval':
            image_index = image_index[: num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, : 2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, : 2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]#list of predicate id包括数据集所有的predicate列表
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0]
            == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    for i in range(len(image_index)):#此时image_index已经是train的数量了
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]#img i的rel起始predicate序号
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start: i_rel_end + 1]

            obj_idx = _relations[i_rel_start: i_rel_end
                                 + 1] - i_obj_start  # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            # (num_rel, 3), representing sub, obj, and pred
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, relationships
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (boxlist_iouboxlist_iouxmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def boxlist_union(boxlist1, boxlist2):
    """
    Compute the union region of two set of boxes

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [N,4].

    Returns:
      (tensor) union, sized [N,4].
    """
    assert len(boxlist1) == len(boxlist2) and boxlist1.size == boxlist2.size
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    union_box = torch.cat((
        torch.min(boxlist1.bbox[:,:2], boxlist2.bbox[:,:2]),
        torch.max(boxlist1.bbox[:,2:], boxlist2.bbox[:,2:])
        ),dim=1)
    return BoxList(union_box, boxlist1.size, "xyxy")

def boxlist_intersection(boxlist1, boxlist2):
    """
    Compute the intersection region of two set of boxes

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [N,4].

    Returns:
      (tensor) intersection, sized [N,4].
    """
    assert len(boxlist1) == len(boxlist2) and boxlist1.size == boxlist2.size
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    inter_box = torch.cat((
        torch.max(boxlist1.bbox[:,:2], boxlist2.bbox[:,:2]),
        torch.min(boxlist1.bbox[:,2:], boxlist2.bbox[:,2:])
        ),dim=1)
    invalid_bbox = torch.max((inter_box[:,0] >= inter_box[:,2]).long(), (inter_box[:,1] >= inter_box[:,3]).long())
    inter_box[invalid_bbox > 0] = 0
    return BoxList(inter_box, boxlist1.size, "xyxy")

# class BoxList(object):
#     """
#     This class represents a set of bounding boxes.
#     The bounding boxes are represented as a Nx4 Tensor.
#     In order to uniquely determine the bounding boxes with respect
#     to an image, we also store the corresponding image dimensions.
#     They can contain extra information that is specific to each bounding box, such as
#     labels.
#     """
#
#     def __init__(self, bbox, image_size, mode="xyxy"):
#         device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
#         bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
#         if bbox.ndimension() != 2:
#             raise ValueError(
#                 "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
#             )
#         if bbox.size(-1) != 4:
#             raise ValueError(
#                 "last dimension of bbox should have a "
#                 "size of 4, got {}".format(bbox.size(-1))
#             )
#         if mode not in ("xyxy", "xywh"):
#             raise ValueError("mode should be 'xyxy' or 'xywh'")
#
#         self.bbox = bbox
#         self.size = image_size  # (image_width, image_height)
#         self.mode = mode
#         self.extra_fields = {}
#         self.triplet_extra_fields = []  # e.g. relation field, which is not the same size as object bboxes and should not respond to __getitem__ slicing v[item]
#         self.custom_extra_fields=[]
#     def add_field(self, field, field_data, is_triplet=False,is_custom=False):
#         # if field in self.extra_fields:
#         #     print('{} is already in extra_fields. Try to replace with new data. '.format(field))
#         self.extra_fields[field] = field_data
#         if is_triplet:
#             self.triplet_extra_fields.append(field)
#         if is_custom:#任意维度tensor或者其他，不受影响
#             self.custom_extra_fields.append(field)
#     def del_field(self, field):
#         # if field in self.extra_fields:
#         #     print('{} is already in extra_fields. Try to replace with new data. '.format(field))
#         del self.extra_fields[field]
#     def get_field(self, field):
#         return self.extra_fields[field]
#
#     def has_field(self, field):
#         return field in self.extra_fields
#
#     def fields(self):
#         return list(self.extra_fields.keys())
#
#     def _copy_extra_fields(self, bbox):
#         for k, v in bbox.extra_fields.items():
#             self.extra_fields[k] = v
#
#     def convert(self, mode):
#         if mode not in ("xyxy", "xywh"):
#             raise ValueError("mode should be 'xyxy' or 'xywh'")
#         if mode == self.mode:
#             return self
#         # we only have two modes, so don't need to check
#         # self.mode
#         xmin, ymin, xmax, ymax = self._split_into_xyxy()
#         if mode == "xyxy":
#             bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
#             bbox = BoxList(bbox, self.size, mode=mode)
#         else:
#             TO_REMOVE = 0
#             bbox = torch.cat(
#                 (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
#             )
#             bbox = BoxList(bbox, self.size, mode=mode)
#         bbox._copy_extra_fields(self)
#         return bbox
#
#     def _split_into_xyxy(self):
#         if self.mode == "xyxy":
#             xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
#             return xmin, ymin, xmax, ymax
#         elif self.mode == "xywh":
#             TO_REMOVE = 0
#             xmin, ymin, w, h = self.bbox.split(1, dim=-1)
#             return (
#                 xmin,
#                 ymin,
#                 xmin + (w - TO_REMOVE).clamp(min=0),
#                 ymin + (h - TO_REMOVE).clamp(min=0),
#             )
#         else:
#             raise RuntimeError("Should not be here")
#
#     def resize(self, size, *args, **kwargs):
#         """
#         Returns a resized copy of this bounding box
#
#         :param size: The requested size in pixels, as a 2-tuple:
#             (width, height).
#         """
#
#         ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
#         if ratios[0] == ratios[1]:
#             ratio = ratios[0]
#             scaled_box = self.bbox * ratio
#             bbox = BoxList(scaled_box, size, mode=self.mode)
#             # bbox._copy_extra_fields(self)
#             for k, v in self.extra_fields.items():
#                 if not isinstance(v, torch.Tensor):
#                     v = v.resize(size, *args, **kwargs)
#                 if k in self.triplet_extra_fields:
#                     bbox.add_field(k, v, is_triplet=True)
#                 else:
#                     bbox.add_field(k, v)
#             return bbox
#
#         ratio_width, ratio_height = ratios
#         xmin, ymin, xmax, ymax = self._split_into_xyxy()
#         scaled_xmin = xmin * ratio_width
#         scaled_xmax = xmax * ratio_width
#         scaled_ymin = ymin * ratio_height
#         scaled_ymax = ymax * ratio_height
#         scaled_box = torch.cat(
#             (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
#         )
#         bbox = BoxList(scaled_box, size, mode="xyxy")
#         # bbox._copy_extra_fields(self)
#         for k, v in self.extra_fields.items():
#             if not isinstance(v, torch.Tensor):
#                 v = v.resize(size, *args, **kwargs)
#             if k in self.triplet_extra_fields:
#                 bbox.add_field(k, v, is_triplet=True)
#             else:
#                 bbox.add_field(k, v)
#
#         return bbox.convert(self.mode)
#
#     def transpose(self, method):
#         """
#         Transpose bounding box (flip or rotate in 90 degree steps)
#         :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
#           :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
#           :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
#           :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
#         """
#         if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
#             raise NotImplementedError(
#                 "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
#             )
#
#         image_width, image_height = self.size
#         xmin, ymin, xmax, ymax = self._split_into_xyxy()
#         if method == FLIP_LEFT_RIGHT:
#             TO_REMOVE = 0
#             transposed_xmin = image_width - xmax - TO_REMOVE
#             transposed_xmax = image_width - xmin - TO_REMOVE
#             transposed_ymin = ymin
#             transposed_ymax = ymax
#         elif method == FLIP_TOP_BOTTOM:
#             transposed_xmin = xmin
#             transposed_xmax = xmax
#             transposed_ymin = image_height - ymax
#             transposed_ymax = image_height - ymin
#
#         transposed_boxes = torch.cat(
#             (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
#         )
#         bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
#         # bbox._copy_extra_fields(self)
#         for k, v in self.extra_fields.items():
#             if not isinstance(v, torch.Tensor):
#                 v = v.transpose(method)
#             if k in self.triplet_extra_fields:
#                 bbox.add_field(k, v, is_triplet=True)
#             else:
#                 bbox.add_field(k, v)
#         return bbox.convert(self.mode)
#
#     def crop(self, box):
#         """
#         Cropss a rectangular region from this bounding box. The box is a
#         4-tuple defining the left, upper, right, and lower pixel
#         coordinate.
#         """
#         xmin, ymin, xmax, ymax = self._split_into_xyxy()
#         w, h = box[2] - box[0], box[3] - box[1]
#         cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
#         cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
#         cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
#         cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)
#
#         # TODO should I filter empty boxes here?
#         if False:
#             is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)
#
#         cropped_box = torch.cat(
#             (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
#         )
#         bbox = BoxList(cropped_box, (w, h), mode="xyxy")
#         # bbox._copy_extra_fields(self)
#         for k, v in self.extra_fields.items():
#             if not isinstance(v, torch.Tensor):
#                 v = v.crop(box)
#             if k in self.triplet_extra_fields:
#                 bbox.add_field(k, v, is_triplet=True)
#             else:
#                 bbox.add_field(k, v)
#         return bbox.convert(self.mode)
#
#     # Tensor-like methods
#
#     def to(self, device):
#         bbox = BoxList(self.bbox.to(device), self.size, self.mode)
#         for k, v in self.extra_fields.items():
#             if hasattr(v, "to"):
#                 v = v.to(device)
#             if k in self.triplet_extra_fields:
#                 bbox.add_field(k, v, is_triplet=True)
#             else:
#                 bbox.add_field(k, v)
#         return bbox
#
#     def __getitem__(self, item):
#         bbox = BoxList(self.bbox[item], self.size, self.mode)
#         for k, v in self.extra_fields.items():
#             if k in self.triplet_extra_fields:#用在n*n矩阵上，严格说不是priplet
#                 bbox.add_field(k, v[item][:,item], is_triplet=True)
#             if k in self.custom_extra_fields:
#                 bbox.add_field(k, v, is_custom=True)
#             else:
#                 bbox.add_field(k, v[item])
#         return bbox
#
#     def __len__(self):
#         return self.bbox.shape[0]
#
#     def clip_to_image(self, remove_empty=True):
#         TO_REMOVE = 0#把proposal的坐标钳制在照片尺寸范围
#         self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
#         self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
#         self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
#         self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
#         if remove_empty:#default: false
#             box = self.bbox
#             keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
#             return self[keep]
#         return self
#
#     def area(self):
#         box = self.bbox
#         if self.mode == "xyxy":
#             TO_REMOVE = 0
#             area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
#         elif self.mode == "xywh":
#             area = box[:, 2] * box[:, 3]
#         else:
#             raise RuntimeError("Should not be here")
#
#         return area
#
#     def copy(self):
#         return BoxList(self.bbox, self.size, self.mode)
#
#     def copy_with_fields(self, fields, skip_missing=False):
#         bbox = BoxList(self.bbox, self.size, self.mode)
#         if not isinstance(fields, (list, tuple)):
#             fields = [fields]
#         for field in fields:
#             if self.has_field(field):
#                 if field in self.triplet_extra_fields:
#                     bbox.add_field(field, self.get_field(field), is_triplet=True)
#                 else:
#                     bbox.add_field(field, self.get_field(field))
#             elif not skip_missing:
#                 raise KeyError("Field '{}' not found in {}".format(field, self))
#         return bbox
#
#     def __repr__(self):
#         s = self.__class__.__name__ + "("
#         s += "num_boxes={}, ".format(len(self))
#         s += "image_width={}, ".format(self.size[0])
#         s += "image_height={}, ".format(self.size[1])
#         s += "mode={})".format(self.mode)
#         return s