"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import argparse
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main(ft_path):
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()
    # breakpoint()
    cfg.model_cfg['finetuned'] = ft_path

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg, ft_path)

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.evaluate(skip_reload=True)

def read_path():
    DIR = './cache/'
    first_dir = 'ckpt/output/'
    base_path = f'{DIR}{first_dir}'
    # directories = [d for d in os.listdir(base_path) if os.path.isdir(d)]
    directories = os.listdir(base_path)
    # print(directories)
    ckpt_paths = []
    for dir in directories[:-1]:
        # print(dir)
        second_dir = f'{base_path}{dir}/'
        files_path = os.listdir(second_dir)
        # print(files_path)
        # ['20240320194', '20240321224', '20240325015', '20240326234']
        for file in files_path:
            if file[-4:] == ".pth":
                ckpt_path = f'{first_dir}{dir}/{file}'
                ckpt_paths.append(ckpt_path)
    print(ckpt_paths)
    return ckpt_paths


if __name__ == "__main__":
    ft_path = ''
    # ckpt_paths = read_path()
    # print(ckpt_paths)
    # ckpt/output/20240321224/checkpoint_9.pth
    ckpt_paths = ['ckpt/output/20240320194/checkpoint_2.pth','ckpt/output/20240321224/checkpoint_8.pth']
    for ft_path in ckpt_paths:
        main(ft_path)
        time.sleep(5)
        torch.cuda.empty_cache()

