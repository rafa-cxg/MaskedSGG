# Enhancing Scene Graph Generation via Semantic-Aligned Masked Vision-and-Language Pre-training

This repository provides the official implementation of the paper  
**“Enhancing Scene Graph Generation via Semantic-Aligned Masked Vision-and-Language Pre-training”**,  
submitted to *The Visual Computer (2025)*.

---

## 🧩 Environment Setup

The code is developed and tested with the following environment:

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- torchvision ≥ 0.11  
- CUDA ≥ 11.3  
- Transformers ≥ 4.30  

To install the basic dependencies:



```bash
pip install torch torchvision transformers opencv-python numpy tqdm pillow
```

## 📦 Dataset Preparation

### 1. CC3M (Conceptual Captions 3M)

Used for **caption-based pre-training**.

Download the dataset from the official page:  
🔗 [Conceptual Captions 3M](https://ai.google.com/research/ConceptualCaptions/)

### 2. Visual Genome (VG)
Used for **fine-tuning and evaluation** on the SGG task.  
Download: [Visual Genome Dataset]([https://visualgenome.org/](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)

## 🚀 Pre-training
Run the Vision-and-Language pre-training stage using CC3M:

```bash
python -m torch.distributed.run \
  --nproc_per_node=4 \
  train.py \
  --cfg-path lavis/projects/sggp/train/pretrain.yaml
```
##  🔧 Fine-tuning on Scene Graph Generation
After pre-training, fine-tune the model on Visual Genome:

```bash
python train_finetune.py \
  --dataset vg \
  --batch_size 32 \
  --epochs 10 \
  --lr 5e-5 \
  --pretrained checkpoints/pretrain/model_best.pth \
  --output_dir checkpoints/finetune/
```
Evaluate the fine-tuned model (e.g., for PredCls):

```bash
python eval_sgg.py \
  --dataset vg \
  --split test \
  --model checkpoints/finetune/model_best.pth
```
