"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from lavis.tasks.base_task import BaseTask
import logging
import os

import torch
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
import json
 
from torch.nn.utils import clip_grad_norm_
@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, ckpt_path, cuda_enabled=True):

        # metric_logger = MetricLogger(delimiter="  ")
        #
        # header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        # for samples in metric_logger.log_every(data_loader, print_freq, header):
        for samples in data_loader:
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            query = samples['conversations'][0]
            image_tensor = samples["image"]

            eval_output, _ = model.chat(query=query,history=[],image_tensor=image_tensor, temperature=1.0, num_beams=15, num_return_sequences=5)

            results.append(eval_output)

        with open('./cache/dataset/challenge/valid/valid_inputonly.json', 'r', encoding='utf-8') as file:
            llm_dataset = json.load(file)

        for index, encounter in enumerate(llm_dataset):
            # Assuming each output is a single string response and you want to wrap it in a list
            encounter["responses"] = [{"content_en": results[index]}]

        suffix = ckpt_path.split('/')[-2:]
        sfx = suffix[1].split('.')[0]
        # updated_file_path = './cache/dataset/challenge/valid/valid_output.json'
        updated_file_path = f'./cache/dataset/challenge/valid/{suffix[0]}{sfx}.json'
        with open(updated_file_path, 'w', encoding='utf-8') as file:
            json.dump(llm_dataset, file, ensure_ascii=False, indent=4)

