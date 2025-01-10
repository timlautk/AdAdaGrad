# AdAdaGrad: Adaptive Batch Size Schemes for Adaptive Gradient Methods
This repository provides the code for the paper "AdAdaGrad: Adaptive Batch Size Schemes for Adaptive Gradient Methods" ([arXiv:2402.11215
](https://arxiv.org/abs/2402.11215)) by **Tim Tsz-Kit Lau**, Han Liu and Mladen Kolar. The current implementation is based on PyTorch and [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) by Lightning AI.


## Quick Start
The following command allows you to clone the repository and create a conda environment.
```bash
git clone https://github.com/timlautk/AdAdaGrad.git
cd AdAdaGrad

conda create -n adadagrad python=3.11 -y
conda activate adadagrad
```

To install the required dependencies, run the following command:
```bash
pip install -U numpy typing-extensions tensorboard jsonargparse \
pytorch-lightning lightning lightning-fabric torchmetrics litdata transformers datasets

# Install PyTorch nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

For experiments on ImageNet, we use the [ImageNet-1k](https://image-net.org/download) dataset and `ffcv` for fast dataloading. To install `ffcv`, follow the instructions in the [official repository](https://github.com/libffcv/ffcv) and follow the [`ffcv-imagenet` repository](https://github.com/libffcv/ffcv-imagenet/) for data downloading and preprocessing procedures before training. 


## Design and Implementation
We implemented the norm test [1] and the (augmented) inner product test [2] for training deep neural networks. The current implementation allows for multiple-GPU training using PyTorch's DistributedDataParallel (DDP), together with the technique of gradient accumulation. The detailed usage of the codebase is outline in the next section. To avoid interrupting the main training loop, the per-sample gradients are computed via a (deep) copy of the model and `torch.func`. Note that this implementation might not be the most computationally and memory efficient, as we aim to provide a simple and easy-to-implement codebase for the proposed AdAdaGrad. 

[1] Richard H. Byrd, Gillian M. Chin, Jorge Nocedal, and Yuchen Wu. Sample size selection in optimization methods for machine learning. *Mathematical Programming*, 134(1):127–155,
2012.


[2] Raghu Bollapragada, Richard Byrd, and Jorge Nocedal. Adaptive sampling strategies for stochastic optimization. *SIAM Journal on Optimization*, 28(4):3312–3343, 2018.


## Usage
Examples of usage can be found in `expt.sh`. 

For example, to run the experiment of training a 3-layer CNN on the MNIST dataset for image classification with the norm test with $\eta=0.1$, simply run the following command:
```bash
fabric run main.py --devices=4 --num_workers=0 --optimizer=adagrad \
--batch_size=16 --max_test_micro_batch_size=15000 --max_batch_size=60000 \
--learning_rate=5e-1 --max_lr=5e-2 --decay_lr=False --max_samples=6000000 \
--model=cnn --dataset=mnist --test_type=norm --eta=1e-1
```
Training a ResNet-18 on the CIFAR-10 dataset with the norm test with $\eta=0.5$:
```bash
fabric run main.py --devices=4 --num_workers=0 --optimizer=adagrad \
--batch_size=8 --max_test_micro_batch_size=100 --max_micro_batch_size=128 --max_batch_size=50000 \
--max_samples=10000000 --warmup_samples=1000000 --lr_decay_samples=9000000 \
--learning_rate=5e-2 --min_lr=5e-3 --model=resnet18 --dataset=cifar10 --test_type=norm --eta=5e-1
```
Training a ResNet-50 on the CIFAR-100 dataset with the augmented inner product test with $\theta=0.5$ and $\nu=1$:
```bash
fabric run main.py --devices=4 --num_workers=0 --optimizer=sgdm \
--batch_size=256 --max_test_micro_batch_size=64 --max_micro_batch_size=100 --max_batch_size=50000 \
--max_samples=10000000 --warmup_samples=1000000 --lr_decay_samples=9000000 \
--learning_rate=5e-1 --min_lr=5e-2 --model=resnet50 --dataset=cifar100 --test_type=ip+orth --theta=5e-1 --nu=1e0
```
Training a ResNet-101 on the ImageNet dataset with the norm test with $\eta=0.1$:
```bash
fabric run main.py --devices=4 --num_workers=12 --precision=bf16-mixed --optimizer=sgdm \
--data_dir=/data/imagenet-gen/ \
--batch_size=256 --max_test_micro_batch_size=64 --max_micro_batch_size=512 --max_batch_size=111360 \
--max_samples=256233400 --warmup_samples=6405835 --lr_decay_samples=249827565 \
--learning_rate=2.5e0 --min_lr=2.5e-1 --weight_decay=1e-4 --gradient_accumulation_steps=1 \
--model=resnet101 --dataset=imagenet --test_type=norm --eta=1e-1
```


Explanation of the arguments:
- `devices`: The number of GPUs to use.
- `num_nodes`: The number of nodes for distributed training.
- `precision`: The precision for training. 
- `num_workers`: The number of workers for data loading (use `0` if you are using Slurm, except for ImageNet).
- `optimizer`: The optimizer to use. Choose from `adagrad`, `adagradnorm`, `adam`, `adamw`, `sgd`, `sgdm`. Any other `torch.optim.Optimizer` can also be used (modified on your own).
- `batch_size`: The initial batch size.
- `max_micro_batch_size`: The maximum micro batch size for each GPU.
- `max_test_micro_batch_size`: The maximum batch size for the norm test or the (augmented) inner product test.
- `max_batch_size`: The maximum batch size.
- `learning_rate`: The (peak) learning rate.
- `weight_decay`: The weight decay.
- `beta1`: The beta1 parameter for the Adam optimizer.
- `beta2`: The beta2 parameter for the Adam optimizer.
- `grad_clip`: The gradient clipping value.
- `max_lr`: The maximum learning rate for the learning rate schedule.
- `decay_lr`: Whether to decay the learning rate.
- `max_samples`: The maximum number of samples to train.
- `warmup_samples`: The number of samples for warmup.
- `lr_decay_samples`: The number of samples for learning rate decay.
- `scheduler`: The learning rate scheduler to use. Choose from `cosine`, `linear`, `multistep`.
- `gradient_accumulation_steps`: The number of gradient accumulation steps.
- `model`: The model to use. Choose from `logistic`, `cnn`, or any other model from `torchvision.models` (see below; modified on your own).
- `dataset`: The dataset to use. Choose from `mnist`, `cifar10`, `cifar100`, `imagenet`, or any other dataset (modified on your own).
- `test_type`: The test type to use. Choose from `norm` (norm test) or `ip` (inner product test) or `ip+orth` (inner product test with orthogonality test; i.e., augmented inner product test).
- `eta`: The parameter $\eta$ for the norm test.
- `theta`: The parameter $\theta$ for the inner product test.
- `nu`: The parameter $\nu$ for the orthogonality test.



## Supported Models and Datasets
In principle, the proposed AdAdaGrad can be applied to any model (as `nn.Module`) and dataset (constructed with `torch.utils.data.DataLoader`), which can be modified on your own in `main.py`. We provide the following examples which can be used directly in the codebase with the following arguments:

### Models
- `logistic`: A simple logistic regression model for classification (for MNIST only).
- `cnn`: A simple 3-layer CNN model for image classification (for MNIST and CIFAR-10, with different architectures).
- All models from [`torchvision.models` for classification](https://pytorch.org/vision/main/models.html#classification): ResNet, VGG, DenseNet, etc.

### Datasets
- `mnist`: The MNIST dataset.
- `cifar10`: The CIFAR-10 dataset.
- `cifar100`: The CIFAR-100 dataset.
- `imagenet`: The ImageNet dataset.


## Citation
If you find this repository useful for your research, please consider citing the following paper:
```
@article{lau2024adadagrad,
	title={\textsc{AdAdaGrad}: Adaptive Batch Size Schemes for Adaptive Gradient Methods},
	author={Lau, Tim Tsz-Kit and Han Liu and Mladen Kolar},
	year={2024},
	journal={arXiv preprint arXiv:2402.11215}
}
```