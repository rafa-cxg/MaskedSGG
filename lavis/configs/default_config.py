import os

from yacs.config import CfgNode as CN
_C = CN()
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

_C.model = CN()
_C.model.arch = "SggpFinetune"
_C.model.model_type = "base_cc3m"
#  load_finetuned: False # false means  training on pretrained rather finetuned models
_C.model.load_pretrained =  True
#  pretrained =  "pretrain_always_warmup_frombert_8_8_60000.pth"
_C.model.pretrained =  "/home/cxg/projectX/lavis/output/sggp/finetune/20230529223/checkpoint_best.pth"
_C.model.num_classes =  30522 #todo what is it used for?
_C.model.predict_mode =  "bert"
#  META_ARCHITECTURE =  "GeneralizedRCNN" # detector arch

_C.datasets = CN()
  visual_genome =  # name of the dataset builder
    vis_processor =
        train =
          name =  "blip_image_train" #todo 用sggp pre-processor
          image_size =  224
        eval =
          name =  "blip_image_eval"
          image_size =  224
    text_processor =
        train =
          name =  "blip_caption"
          prompt =  "a picture of "
        eval =
          name =  "blip_caption"

run =
  task =  scene_graph_generation
  # optimization-specific
  lr_sched =  "linear_warmup_cosine_lr"
  init_lr =  2e-5
  min_lr =  1e-6
  weight_decay =  0.05
  max_epoch =  20
  warmup_lr =  1e-6
  warmup_steps =  100
  batch_size_train =  8
  accum_grad_iters =  1
  batch_size_eval =  1
  num_workers =  24


  seed =  42
  output_dir =  "output/sggp/finetune"

  amp =  False
  resume_ckpt_path =  null

  evaluate =  False
  train_splits =  ["train"]
  valid_splits =  ["eval"] #be carefulllllll!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  test_splits =  [] # 如果不是[], 就会test

  # distribution-specific
  device =  "cuda"
  world_size =  1
  dist_url =  "env://"
  distributed =  True
