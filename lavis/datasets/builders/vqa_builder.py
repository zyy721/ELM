"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.elm_datasets import ELMDataset, ELMDatasetEvalDataset

from lavis.datasets.datasets.elm_datasets_vqgan import ELMDatasetVQGAN, ELMDatasetEvalDatasetVQGAN


@registry.register_builder("elm")
class ELMBuilder(BaseDatasetBuilder):
    train_dataset_cls = ELMDataset
    eval_dataset_cls = ELMDatasetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/elm/defaults.yaml"}


@registry.register_builder("elmvqgan")
class ELMBuilder(BaseDatasetBuilder):
    train_dataset_cls = ELMDatasetVQGAN
    eval_dataset_cls = ELMDatasetEvalDatasetVQGAN

    DATASET_CONFIG_DICT = {"default": "configs/datasets/elmvqgan/defaults.yaml"}