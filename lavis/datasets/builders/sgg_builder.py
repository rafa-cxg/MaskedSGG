"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import SGGDatasetBuilder
from lavis.datasets.datasets.vg_sgg_datasets import VGSGGDataset



@registry.register_builder("visual_genome")
class VGSGGBuilder(SGGDatasetBuilder):
    train_dataset_cls = VGSGGDataset# choose different strategy for train or test
    eval_dataset_cls = VGSGGDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_sgg.yaml"}# 会读取该默认文件


