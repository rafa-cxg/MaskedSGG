"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

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
from lavis.common.registry import registry
from lavis.common.utils import now


from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *
import wandb
os.environ['WANDB_MODE'] = 'offline'
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#已完成 1 添加wendb查看训练，2 数据集中断处继续训练
def show_params_status(model):
    """
    Prints parameters of a model
    """
    st = {}
    strings = []
    total_params = 0
    trainable_params = 0
    for p_name, p in model.named_parameters():

        if not ("bias" in p_name.split(".")[-1] or "bn" in p_name.split(".")[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
        if p.requires_grad:
            trainable_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in st.items():
        strings.append(
            "{:<80s}: {:<16s}({:8f}) ({})".format(
                p_name, "[{}]".format(",".join(size)), prod, "grad" if p_req_grad else "    "
            )
        )
    strings = "\n".join(strings)
    return (
        f"\n{strings}\n ----- \n \n"
        f"      trainable parameters:  {trainable_params/ 1e6:.3f}/{total_params / 1e6:.3f} M \n "
    )
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

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    if  os.environ['LOCAL_RANK'] == '0':
        wandb.init(project="Project X detect-sgg")

    torch.cuda.empty_cache()
    # torch.cuda.set_per_process_memory_fraction(1.0, 0)
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    model.train()
    # print("trainable models:")
    # print(show_params_status(model))#每部分参数

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":

    main()
