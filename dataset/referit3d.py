import json
import multiprocessing as mp
import os
import random
from copy import deepcopy

import jsonlines
import numpy as np
import torch
from dataset.dataset_mixin import DataAugmentationMixin, LoadScannetMixin
from dataset.path_config import SCAN_FAMILY_BASE
from torch.utils.data import Dataset
from utils.eval_helper import (construct_bbox_corners, convert_pc_to_box,
                               eval_ref_one_sample, is_explicitly_view_dependent)
from utils.label_utils import LabelConverter
from utils.lfs_utils import ensure_not_lfs_pointer
from collections import Counter

class Referit3DDataset(Dataset, LoadScannetMixin, DataAugmentationMixin):
    def __init__(self, split='train', anno_type='nr3d', max_obj_len=60, num_points=1024, pc_type='gt', sem_type='607', filter_lang=False, sr3d_plus_aug=False, max_token_length=None):
        # make sure all input params is valid
        # use ground truth for training
        # test can be both ground truth and non-ground truth
        assert pc_type in ['gt', 'pred']
        assert sem_type in ['607']
        assert split in ['train', 'val', 'test']
        valid_anno_types = {'nr3d', 'sr3d', 'nr3d-multi'}
        normalized_anno_type = anno_type.replace('_', '-')
        assert normalized_anno_type in valid_anno_types
        anno_type = normalized_anno_type
        if split == 'train':
            pc_type = 'gt'
            
        # load file
        anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/refer/' + anno_type + '.jsonl')
        split_file = os.path.join(
            SCAN_FAMILY_BASE, f'annotations/splits/scannetv2_{split}.txt'
        )
        split_scan_ids = None
        split_lines = []
        if os.path.exists(split_file):
            ensure_not_lfs_pointer(
                split_file,
                hint="Run `git lfs pull dataset/scanfamily/annotations/splits` to download the ScanNet split lists.",
            )
            with open(split_file, 'r') as sf:
                split_lines = [x.strip() for x in sf if x.strip()]
            if split_lines:
                first_line = split_lines[0]
                if first_line.startswith('version https://git-lfs.github.com/spec/v1'):
                    raise ValueError(
                        f"The split file '{split_file}' looks like a Git-LFS pointer. "
                        "Please run `git lfs pull dataset/scanfamily/annotations/splits` to download the actual list of scan ids."
                    )
                else:
                    split_scan_ids = set(split_lines)
        else:
            print(
                f"[Referit3DDataset] Warning: split file '{split_file}' not found. "
                "Please ensure the ScanNet splits are downloaded. Falling back to using all annotations for now."
            )
        self.scan_ids = set() # scan ids in data
        self.data = [] # scanrefer data
        def within_length_limit(tokens):
            if max_token_length is None:
                return True
            return len(tokens) <= max_token_length

        dropped_long_utterances = 0
        dropped_other_split = 0

        with jsonlines.open(anno_file, 'r') as f:
            for item in f:
                if split_scan_ids is not None and item['scan_id'] not in split_scan_ids:
                    dropped_other_split += 1
                    continue
                if within_length_limit(item['tokens']):
                    self.scan_ids.add(item['scan_id'])
                    self.data.append(item)
                else:
                    dropped_long_utterances += 1
        # special for nr3d
        if sr3d_plus_aug and split == 'train':
            anno_file = os.path.join(SCAN_FAMILY_BASE, 'annotations/refer/' + 'sr3d+' + '.jsonl')
            with jsonlines.open(anno_file, 'r') as f:
                for item in f:
                    if split_scan_ids is not None and item['scan_id'] not in split_scan_ids:
                        dropped_other_split += 1
                        continue
                    if within_length_limit(item['tokens']):
                        self.scan_ids.add(item['scan_id'])
                        self.data.append(item)
                    else:
                        dropped_long_utterances += 1

        if len(self.data) == 0:
            msg = [
                f"Referit3DDataset loaded 0 samples for split '{split}' and anno_type '{anno_type}'.",
                "Possible causes:",
                "  1) The ScanNet split file only contains Git-LFS metadata instead of scene ids.",
                "  2) All utterances were filtered out by max_token_length.",
                "  3) The annotation file path is empty or malformed.",
            ]
            if max_token_length is not None:
                msg.append(
                    f"     - Current max_token_length={max_token_length}, dropped {dropped_long_utterances} long utterances."
                )
            raise ValueError("\n".join(msg))

        if dropped_long_utterances > 0 and max_token_length is not None:
            print(
                f"[Referit3DDataset] Warning: filtered out {dropped_long_utterances} utterances longer than {max_token_length} tokens."
            )
        
        # fill parameters
        self.split = split
        self.max_obj_len = max_obj_len - 1
        self.num_points = num_points
        self.pc_type = pc_type
        self.sem_type = sem_type
        self.filter_lang = filter_lang
        self.anno_type = anno_type
        
        # load category file
        self.int2cat = json.load(open(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2_raw_categories.json"), 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2-labels.combined.tsv"))
        
        # load scans
        self.scans = self.load_scannet(self.scan_ids, self.pc_type, self.split != 'test')
        
        # build unique multiple look up
        for scan_id in self.scan_ids:
            inst_labels = self.scans[scan_id]['inst_labels']
            self.scans[scan_id]['label_count'] = Counter([l for l in inst_labels])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # load scanrefer
        item = self.data[idx]
        item_id = item['item_id']
        scan_id =  item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        is_view_dependent = is_explicitly_view_dependent(item['tokens'])
        anchor_original_ids = []
        for anchor_group in item.get('anchor_ids', []):
            for anchor_idx in anchor_group:
                anchor_idx = int(anchor_idx)
                if anchor_idx not in anchor_original_ids:
                    anchor_original_ids.append(anchor_idx)
        
        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = deepcopy(self.scans[scan_id]['pcds']) # N, 6
            obj_labels = deepcopy(self.scans[scan_id]['inst_labels']) # N
        elif self.pc_type == 'pred':
            obj_pcds = deepcopy(self.scans[scan_id]['pcds_pred'])
            obj_labels = deepcopy(self.scans[scan_id]['inst_labels_pred'])
            # get obj labels by matching
            gt_obj_labels = self.scans[scan_id]['inst_labels'] # N
            obj_center = self.scans[scan_id]['obj_center'] 
            obj_box_size = self.scans[scan_id]['obj_box_size']
            obj_center_pred = self.scans[scan_id]['obj_center_pred'] 
            obj_box_size_pred = self.scans[scan_id]['obj_box_size_pred']
            for i in range(len(obj_center_pred)):
                for j in range(len(obj_center)):
                    if eval_ref_one_sample(construct_bbox_corners(obj_center[j], obj_box_size[j]), construct_bbox_corners(obj_center_pred[i], obj_box_size_pred[i])) >= 0.25:
                        obj_labels[i] = gt_obj_labels[j]
                        break
            
        # filter out background or language
        if self.filter_lang:
            if self.pc_type == 'gt':
                filtered_obj_indices = [i for i, obj_label in enumerate(obj_labels) if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling']) and (self.int2cat[obj_label] in sentence)]
                if tgt_object_id not in filtered_obj_indices:
                    filtered_obj_indices.append(tgt_object_id)
            else:
                filtered_obj_indices = [i for i in range(len(obj_pcds))]
        else:
            if self.pc_type == 'gt':
                filtered_obj_indices = [i for i, obj_label in enumerate(obj_labels) if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
            else:
                filtered_obj_indices = [i for i in range(len(obj_pcds))]
        obj_pcds = [obj_pcds[id] for id in filtered_obj_indices]
        obj_labels = [obj_labels[id] for id in filtered_obj_indices]
        obj_original_indices = list(filtered_obj_indices)

        # build tgt object id and box
        if self.pc_type == 'gt':
           tgt_object_id = filtered_obj_indices.index(tgt_object_id)
           tgt_object_label = obj_labels[tgt_object_id]
           tgt_object_id_iou25_list = [tgt_object_id]
           tgt_object_id_iou50_list = [tgt_object_id]
           assert(self.int2cat[tgt_object_label] == tgt_object_name)
        elif self.pc_type == 'pred':
            gt_pcd = self.scans[scan_id]["pcds"][tgt_object_id]
            gt_center, gt_box_size = convert_pc_to_box(gt_pcd)
            tgt_object_id = -1
            tgt_object_id_iou25_list = []
            tgt_object_id_iou50_list = []
            tgt_object_label = self.cat2int[tgt_object_name]
            # find tgt iou 25
            for i in range(len(obj_pcds)):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size), construct_bbox_corners(gt_center, gt_box_size)) >= 0.25:
                    tgt_object_id = i
                    tgt_object_id_iou25_list.append(i)
            # find tgt iou 50
            for i in range(len(obj_pcds)):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size), construct_bbox_corners(gt_center, gt_box_size)) >= 0.5:
                    tgt_object_id_iou50_list.append(i)
        assert(len(obj_pcds) == len(obj_labels))
        
        # crop objects 
        anchor_filtered_indices = []
        if anchor_original_ids:
            for anchor_idx in anchor_original_ids:
                if anchor_idx in obj_original_indices:
                    anchor_filtered_indices.append(obj_original_indices.index(anchor_idx))

        if self.max_obj_len < len(obj_labels):
            selected_obj_idxs = []

            def add_selected(idx):
                if 0 <= idx < len(obj_labels) and idx not in selected_obj_idxs:
                    selected_obj_idxs.append(idx)

            if tgt_object_id != -1:
                add_selected(tgt_object_id)
            for idx in tgt_object_id_iou25_list:
                add_selected(idx)
            for idx in tgt_object_id_iou50_list:
                add_selected(idx)
            for idx in anchor_filtered_indices:
                add_selected(idx)

            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in selected_obj_idxs:
                    if klabel == tgt_object_label:
                        add_selected(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                for idx in remained_obj_idx:
                    add_selected(idx)
                    if len(selected_obj_idxs) == self.max_obj_len:
                        break
            selected_obj_idxs = selected_obj_idxs[:self.max_obj_len]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            obj_original_indices = [obj_original_indices[i] for i in selected_obj_idxs]
            if tgt_object_id != -1:
                tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_id_iou25_list = [selected_obj_idxs.index(id) for id in tgt_object_id_iou25_list if id in selected_obj_idxs]
            tgt_object_id_iou50_list = [selected_obj_idxs.index(id) for id in tgt_object_id_iou50_list if id in selected_obj_idxs]
            anchor_filtered_indices = [selected_obj_idxs.index(id) for id in anchor_filtered_indices if id in selected_obj_idxs]
            assert len(obj_pcds) == len(obj_labels)

        # rebuild tgt_object_id
        if tgt_object_id == -1:
            tgt_object_id = len(obj_pcds)

        anchor_final_indices = []
        if anchor_original_ids:
            original_to_final = {orig_idx: i for i, orig_idx in enumerate(obj_original_indices)}
            for anchor_idx in anchor_original_ids:
                if anchor_idx in original_to_final:
                    anchor_final_indices.append(original_to_final[anchor_idx])

        # rotate obj
        rot_matrix = self.build_rotate_mat()
        
        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for obj_pcd in obj_pcds:
            # build locs
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            # build box
            obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
            obj_box_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))
            # subsample
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points, replace=(len(obj_pcd) < self.num_points))
            obj_pcd = obj_pcd[pcd_idxs]
            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)
            
        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)
        
        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_fts.shape[0] == obj_locs.shape[0]
        
        # build iou25 and iou50
        tgt_object_id_iou25 = torch.zeros(len(obj_fts) + 1).long()
        tgt_object_id_iou50 = torch.zeros(len(obj_fts) + 1).long()
        for id in tgt_object_id_iou25_list:
            tgt_object_id_iou25[id] = 1
        for id in tgt_object_id_iou50_list:
            tgt_object_id_iou50[id] = 1
        
        # build unique multiple
        is_multiple = self.scans[scan_id]['label_count'][tgt_object_label] > 1
        is_hard = self.scans[scan_id]['label_count'][tgt_object_label] > 2
        
        
        data_dict = {
            "sentence": sentence,
            "tgt_object_id": torch.LongTensor([tgt_object_id]), # 1
            "tgt_object_label": torch.LongTensor([tgt_object_label]), # 1
            "obj_fts": obj_fts, # N, 6
            "obj_locs": obj_locs, # N, 3
            "obj_labels": obj_labels, # N,
            "obj_boxes": obj_boxes, # N, 6
            "data_idx": item_id,
            "tgt_object_id_iou25": tgt_object_id_iou25,
            "tgt_object_id_iou50": tgt_object_id_iou50,
            'is_multiple': is_multiple,
            'is_view_dependent': is_view_dependent,
            'is_hard': is_hard,
            'anchor_ids': torch.LongTensor(anchor_final_indices),
        }
    
        return data_dict

if __name__ == '__main__':
    dataset = Referit3DDataset(split='val', pc_type='pred')
    print(dataset[0])