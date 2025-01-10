from jsonargparse import ArgumentParser

import os
import time
import math
import copy
from pathlib import Path
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models

import lightning as L
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.strategies import DDPStrategy

from torchmetrics.classification import Accuracy
from torchmetrics.aggregation import RunningMean

from data import prepare_data
from modules import MNISTModule, LogisticRegression, CIFAR10Module, LitResnet, ImageNetModule
from utils import (get_warmup_cosine_lr_iter, get_warmup_cosine_lr_sample, get_warmup_linear_lr_iter, 
                   get_warmup_linear_lr_sample, get_multistep_lr_iter, get_multistep_lr_sample
                )
from tests import (compute_per_sample_gradients, norm_test, inner_product_test, 
                    norm_test_distributed, inner_product_test_distributed, orthogonality_test_distributed, 
                    inner_product_orthogonality_test, inner_product_orthogonality_test_distributed,
                    var_sample_grad_accumulate, norm_test_accumulate, norm_test_distributed_accumulate
                )
from adagradnorm import Adagradnorm


def train(fabric, model, optimizer, hparams, train_set, val_set, train_data, val_data):
    log_interval = hparams.log_interval
    eval_interval = hparams.eval_interval
    micro_batch_size = next(iter(train_data))[0].shape[0]
    gradient_accumulation_steps = hparams.gradient_accumulation_steps
    global_batch_size = micro_batch_size * fabric.world_size * gradient_accumulation_steps
    max_global_batch_size = hparams.max_batch_size
    max_micro_batch_size = hparams.max_micro_batch_size
    max_iters = hparams.max_iters
    max_samples = hparams.max_samples
    lr = optimizer.param_groups[0]['lr']
    learning_rate = hparams.learning_rate
    
    if hparams.stagewise_ramp_up:
        global_batch_sizes = hparams.global_batch_sizes
        global_batch_size = hparams.global_batch_sizes[0]
        stage = 0
        if len(hparams.samples_stages) == 0:
            samples_stage = max_samples // len(global_batch_sizes)
            samples_stages = [samples_stage] * len(global_batch_sizes)
        else:
            samples_stages = hparams.samples_stages

    if hparams.linear_ramp_up:
        ramp_up_increment = max_micro_batch_size * fabric.world_size
        global_batch_sizes = [global_batch_size + ramp_up_increment * i \
                                for i in range((max_global_batch_size - global_batch_size) // ramp_up_increment + 1)]
        stage = 0
    
    # training loop
    master_process = fabric.global_rank == 0
    samples_processed = 0
    samples_processed_stagewise = 0
    iter_num = 0
    if hparams.dataset == "imagenet":
        accuracy_train = Accuracy(num_classes=1000, task="multiclass").to(fabric.device)
        accuracy_train_top_5 = Accuracy(num_classes=1000, task="multiclass", top_k=5).to(fabric.device)
    elif hparams.dataset == "cifar100":
        accuracy_train = Accuracy(num_classes=100, task="multiclass").to(fabric.device)
        accuracy_train_top_5 = Accuracy(num_classes=100, task="multiclass", top_k=5).to(fabric.device)
    else:
        accuracy_train = Accuracy(num_classes=10, task="multiclass").to(fabric.device)
    running_loss = RunningMean(window=gradient_accumulation_steps, dist_sync_on_step=False, sync_on_compute=False).to(fabric.device)
    fabric.barrier()    
    t0 = time.perf_counter()
    train_iter = iter(train_data)

    while True:        
        last_global_batch_size = global_batch_size
        last_micro_batch_size = micro_batch_size
        last_gradient_accumulation_steps = gradient_accumulation_steps
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # create a stateless model used for all gradient accumulation steps 
        # for per-sample gradients computation if test is enabled
        if hparams.test and iter_num % hparams.test_interval == 0 and last_global_batch_size < max_global_batch_size:
            with torch.no_grad():
                stateless_model = copy.deepcopy(model._forward_module)

        for micro_step in range(gradient_accumulation_steps):
            try:
                data, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_data)
                data, targets = next(train_iter)

            # compute per-sample gradients if test is enabled
            if hparams.test and iter_num % hparams.test_interval == 0 and last_global_batch_size < max_global_batch_size:
                with torch.no_grad():
                    ft_per_sample_grads = compute_per_sample_gradients(stateless_model, data, targets)

                    if gradient_accumulation_steps > 1:
                        if micro_step == 0:
                            mean_per_sample_gradients, mean_per_sample_gradients_squared, mean_per_sample_gradients_variance = {}, {}, {}
                        mean_per_sample_gradients, mean_per_sample_gradients_squared, mean_per_sample_gradients_variance = \
                            var_sample_grad_accumulate(micro_step, gradient_accumulation_steps, ft_per_sample_grads, 
                                            mean_per_sample_gradients, mean_per_sample_gradients_squared, mean_per_sample_gradients_variance)
                        del ft_per_sample_grads
            
            with fabric.no_backward_sync(model, enabled=(micro_step < gradient_accumulation_steps - 1)):
                logits = model(data)
                loss = F.cross_entropy(logits, targets)
                fabric.backward(loss / gradient_accumulation_steps)
                accuracy_train(logits.argmax(-1), targets)
                if hparams.dataset == "imagenet" or hparams.dataset == "cifar100":
                        accuracy_train_top_5(logits, targets)
            
            running_loss(loss)
        
        if hparams.grad_clip != 0.0:
            fabric.clip_gradients(model, optimizer, max_norm=hparams.grad_clip)
        optimizer.step()
        train_acc = accuracy_train.compute()
        loss = running_loss.compute()
        if hparams.dataset == "imagenet" or hparams.dataset == "cifar100":
            train_acc_top_5 = accuracy_train_top_5.compute()
        samples_processed += last_global_batch_size

        # norm test or (augmented) inner product test
        if hparams.test and iter_num % hparams.test_interval == 0 and last_global_batch_size < max_global_batch_size:
            if gradient_accumulation_steps == 1:
                if fabric.world_size == 1:
                    if hparams.test_type == "norm":
                        global_batch_size, variance_one_norm, batch_grad_norm_squared \
                            = norm_test(ft_per_sample_grads, hparams.eta)
                    elif hparams.test_type == "ip":
                        global_batch_size, variances_ip, batch_grad_norm_squared \
                            = inner_product_test(ft_per_sample_grads, hparams.theta)
                    elif hparams.test_type == "ip+orth":
                        global_batch_size, variances_ip, variances_orth, batch_grad_norm_squared \
                            = inner_product_orthogonality_test(ft_per_sample_grads, hparams.theta, hparams.nu)
                else: 
                    if hparams.test_type == "norm":
                        global_batch_size, variance_one_norm, batch_grad_norm_squared \
                            = norm_test_distributed(fabric, ft_per_sample_grads, hparams.eta)
                    elif hparams.test_type == "ip":
                        global_batch_size, variances_ip, batch_grad_norm_squared \
                            = inner_product_test_distributed(fabric, ft_per_sample_grads, hparams.theta)
                    elif hparams.test_type == "ip+orth":
                        global_batch_size, variances_ip, variances_orth, batch_grad_norm_squared \
                            = inner_product_orthogonality_test_distributed(fabric, ft_per_sample_grads, hparams.theta, hparams.nu)
                del stateless_model, ft_per_sample_grads
            else:
                if hparams.test_type == "norm":
                    if fabric.world_size == 1:
                        global_batch_size, variance_one_norm, batch_grad_norm_squared = norm_test_accumulate(gradient_accumulation_steps, micro_batch_size, 
                            mean_per_sample_gradients, mean_per_sample_gradients_squared, mean_per_sample_gradients_variance, hparams.eta)
                    else:
                        global_batch_size, variance_one_norm, batch_grad_norm_squared = norm_test_distributed_accumulate(fabric, gradient_accumulation_steps, micro_batch_size, 
                                            mean_per_sample_gradients, mean_per_sample_gradients_squared, mean_per_sample_gradients_variance, hparams.eta)
                else:
                    raise NotImplementedError("Test type not implemented for gradient accumulation steps > 1")
                del stateless_model, mean_per_sample_gradients, mean_per_sample_gradients_squared, mean_per_sample_gradients_variance

            global_batch_size = min(global_batch_size, max_global_batch_size)
        
        # update global batch size of stage if using stagewise ramp-up
        if hparams.stagewise_ramp_up:
            samples_processed_stagewise += last_global_batch_size
            if samples_processed_stagewise >= sum(samples_stages[:stage+1]) and stage < len(global_batch_sizes) - 1:
                stage += 1
                global_batch_size = global_batch_sizes[stage]
        
        # update global batch size of stage if using linear ramp-up
        if hparams.linear_ramp_up:
            if global_batch_size < max_global_batch_size:
                global_batch_size += hparams.linear_ramp_up_increment
                global_batch_size = min(global_batch_size, max_global_batch_size)

        # timing and logging
        t1 = time.perf_counter()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item()
            fabric.log_dict({
                    "sample/step": iter_num,
                    "sample/train/loss": lossf, 
                    "sample/train/accuracy": train_acc, 
                    "sample/lr": lr, 
                    "sample/batch_size": last_global_batch_size,
                    "sample/micro_batch_size": micro_batch_size,
                    "sample/gradient_accumulation_steps": last_gradient_accumulation_steps,
                    "sample/eta": hparams.eta,
                    "sample/theta": hparams.theta,
                    "sample/nu": hparams.nu,
                }, 
                step=samples_processed)
            fabric.log_dict({
                    "step/train/loss": lossf, 
                    "step/train/accuracy": train_acc, 
                    "step/time": dt,
                    "step/lr": lr, 
                    "step/batch_size": last_global_batch_size,
                    "step/micro_batch_size": micro_batch_size,
                    "step/gradient_accumulation_steps": last_gradient_accumulation_steps,
                }, 
                step=iter_num)
            if hparams.dataset == "imagenet" or hparams.dataset == "cifar100":
                fabric.log_dict({
                        "sample/train/accuracy_top_5": train_acc_top_5,
                    }, 
                    step=samples_processed)
                fabric.log_dict({
                        "step/train/accuracy_top_5": train_acc_top_5,
                    }, 
                    step=iter_num)
            if hparams.test and iter_num % hparams.test_interval == 0 and last_global_batch_size < max_global_batch_size:
                if hparams.test_type == "norm":
                    fabric.log_dict({
                        "sample/var_one_norm": variance_one_norm, 
                        "sample/grad_norm_sq": batch_grad_norm_squared
                    }, step=samples_processed)
                    fabric.log_dict({
                        "step/var_one_norm": variance_one_norm, 
                        "step/grad_norm_sq": batch_grad_norm_squared
                    }, step=iter_num)
                elif hparams.test_type == "ip":
                    fabric.log_dict({
                        "sample/var_inner_prod": variances_ip, 
                        "sample/grad_norm_sq": batch_grad_norm_squared
                    }, step=samples_processed)
                    fabric.log_dict({
                        "step/var_inner_prod": variances_ip, 
                        "step/grad_norm_sq": batch_grad_norm_squared
                    }, step=iter_num)
                elif hparams.test_type == "ip+orth":
                    fabric.log_dict({
                        "sample/var_inner_prod": variances_ip, 
                        "sample/var_orth": variances_orth, 
                        "sample/grad_norm_sq": batch_grad_norm_squared
                    }, step=samples_processed)
                    fabric.log_dict({
                        "step/var_inner_prod": variances_ip, 
                        "step/var_orth": variances_orth, 
                        "step/grad_norm_sq": batch_grad_norm_squared
                    }, step=iter_num)
            if hparams.training_type == 'iter':
                if hparams.dataset == "imagenet" or hparams.dataset == "cifar100":
                    fabric.print(
                        "iter {}/{}: loss {:.6f}, time {:.2f}s, train acc {:.2f}%, top 5 train acc {:.2f}%, lr {:.4f}, global batch size {}".format(
                            iter_num, max_iters, 
                            lossf,
                            dt,
                            100.0 * train_acc,
                            100.0 * train_acc_top_5,
                            lr,
                            last_global_batch_size,
                        )
                    )
                else:
                    fabric.print(
                        "iter {}/{}: loss {:.6f}, time {:.2f}s, train acc {:.2f}%, lr {:.4f}, global batch size {}".format(
                            iter_num, max_iters, 
                            lossf,
                            dt,
                            100.0 * train_acc,
                            lr,
                            last_global_batch_size,
                        )
                    )
            elif hparams.training_type == 'sample':
                if hparams.dataset == "imagenet" or hparams.dataset == "cifar100":
                    fabric.print(
                        "samples {}/{}: loss {:.6f}, time {:.2f}s, train acc {:.2f}%, top 5 train acc {:.2f}%, lr {:.4f}, global batch size {}".format(
                            samples_processed, max_samples, 
                            lossf,
                            dt,
                            100.0 * train_acc,
                            100.0 * train_acc_top_5,
                            lr,
                            last_global_batch_size,
                        )
                    )
                else:
                    fabric.print(
                        "samples {}/{}: loss {:.6f}, time {:.2f}s, train acc {:.2f}%, lr {:.4f}, global batch size {}".format(
                            samples_processed, max_samples, 
                            lossf,
                            dt,
                            100.0 * train_acc,
                            lr,
                            last_global_batch_size,
                        )
                    )

        # validation
        if iter_num % eval_interval == 0:
            validate(fabric, model, iter_num, hparams, val_data, samples_processed, last_global_batch_size)
            fabric.barrier()
        
        # update batch size if necessary
        if hparams.test and iter_num % hparams.test_interval == 0:
            micro_batch_size = math.ceil(global_batch_size / (fabric.world_size * gradient_accumulation_steps)) 
            gradient_accumulation_steps = math.ceil(global_batch_size / (fabric.world_size * micro_batch_size))
            global_batch_size = micro_batch_size * fabric.world_size * gradient_accumulation_steps
            if global_batch_size > max_global_batch_size:
                global_batch_size = max_global_batch_size
                micro_batch_size = max_micro_batch_size
                gradient_accumulation_steps = math.ceil(global_batch_size / (fabric.world_size * micro_batch_size))

        if global_batch_size > last_global_batch_size:
            # update gradient accumulation steps, effective micro and global batch sizes
            micro_batch_size = math.ceil(global_batch_size / (fabric.world_size * gradient_accumulation_steps)) 
            global_batch_size = micro_batch_size * fabric.world_size * gradient_accumulation_steps

            if micro_batch_size > max_micro_batch_size:
                micro_batch_size = max_micro_batch_size
                gradient_accumulation_steps = math.ceil(global_batch_size / (micro_batch_size * fabric.world_size))
                global_batch_size = micro_batch_size * fabric.world_size * gradient_accumulation_steps

            if global_batch_size > max_global_batch_size:
                global_batch_size = max_global_batch_size
                micro_batch_size = math.ceil(global_batch_size / (fabric.world_size * gradient_accumulation_steps)) 
                gradient_accumulation_steps = math.ceil(global_batch_size / (micro_batch_size * fabric.world_size))
                global_batch_size = micro_batch_size * fabric.world_size * gradient_accumulation_steps

        # update dataloaders after ramping up micro batch size
        if micro_batch_size != last_micro_batch_size:
            del train_data, val_data
            gc.collect()
            train_data, val_data = None, None
            
            if hparams.dataset == "imagenet":
                from data import get_imagenet_loader
                train_data = get_imagenet_loader(data_pth=Path(hparams.data_dir) / "train_500_0.50_90.ffcv", batch_size=micro_batch_size,
                    num_workers=hparams.num_workers, drop_last=True, device=fabric.device, train=True, seed=(hparams.seed + iter_num),
                    distributed=True, res=224, in_memory=True
                )        
                val_data = get_imagenet_loader(data_pth=Path(hparams.data_dir) / "val_500_0.50_90.ffcv", batch_size=micro_batch_size,
                    num_workers=hparams.num_workers, drop_last=False, device=fabric.device, train=False, seed=(hparams.seed + iter_num),
                    distributed=True, res=224, in_memory=True
                )
            else:
                train_data = DataLoader(train_set, pin_memory=torch.cuda.is_available(), shuffle=hparams.shuffle, num_workers=hparams.num_workers, 
                                    batch_size=micro_batch_size, generator=torch.Generator().manual_seed(iter_num))
                val_data = DataLoader(val_set, pin_memory=torch.cuda.is_available(), shuffle=False, num_workers=hparams.num_workers, 
                                    batch_size=micro_batch_size, generator=torch.Generator().manual_seed(iter_num))
                train_data, val_data = fabric.setup_dataloaders(train_data, val_data)
            fabric.barrier()
            train_iter = iter(train_data)

        # determine and set the learning rate for the next iteration
        if hparams.decay_lr:
            if hparams.lr_decay_type == 'iter':
                if hparams.scheduler == 'cosine':
                    lr = get_warmup_cosine_lr_iter(iter_num, hparams.warmup_iters, hparams.lr_decay_iters, hparams.min_lr, learning_rate)
                elif hparams.scheduler == 'linear':
                    lr = get_warmup_linear_lr_iter(iter_num, hparams.warmup_iters, hparams.lr_decay_iters, hparams.min_lr, learning_rate)
                elif hparams.scheduler == 'multistep':
                    lr = get_multistep_lr_iter(samples_processed, learning_rate, hparams.milestones, hparams.gamma)
            elif hparams.lr_decay_type == 'sample':
                if hparams.scheduler == 'cosine':
                    lr = get_warmup_cosine_lr_sample(samples_processed, hparams.warmup_samples, hparams.lr_decay_samples, hparams.min_lr, learning_rate)
                elif hparams.scheduler == 'linear':
                    lr = get_warmup_linear_lr_sample(samples_processed, hparams.warmup_samples, hparams.lr_decay_samples, hparams.min_lr, learning_rate)
                elif hparams.scheduler == 'multistep':
                    lr = get_multistep_lr_sample(samples_processed, learning_rate, hparams.milestones, hparams.gamma)
        else: 
            lr = learning_rate
    
        for param_group in optimizer.param_groups:
            if not hparams.decay_lr:
                lr = min(lr, hparams.max_lr)
            param_group['lr'] = lr
        
        running_loss.reset()
        accuracy_train.reset()
        if hparams.dataset == "imagenet" or hparams.dataset == "cifar100":
            accuracy_train_top_5.reset()
        iter_num += 1
        # termination conditions
        if hparams.training_type == 'iter':
            if iter_num >= hparams.max_iters:
                break
        elif hparams.training_type == 'sample':
            if samples_processed >= hparams.max_samples:
                break
    
    # fabric.print(torch.cuda.memory_summary())

def validate(fabric, model, iter_num, hparams, dataloader, samples_processed, batch_size):
    # Validation loop
    model.eval()
    loss_val = 0
    if hparams.dataset == "imagenet":
        accuracy_val = Accuracy(num_classes=1000, task="multiclass").to(fabric.device)
        accuracy_val_top_5 = Accuracy(num_classes=1000, task="multiclass", top_k=5).to(fabric.device)
    elif hparams.dataset == "cifar100":
        accuracy_val = Accuracy(num_classes=100, task="multiclass").to(fabric.device)
        accuracy_val_top_5 = Accuracy(num_classes=100, task="multiclass", top_k=5).to(fabric.device)
    else:
        accuracy_val = Accuracy(num_classes=10, task="multiclass").to(fabric.device)
    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            logits = model(data)
            loss = F.cross_entropy(logits, targets)
            loss_val += loss
            accuracy_val(logits.argmax(-1), targets)
            if hparams.dataset == "imagenet" or hparams.dataset == "cifar100":
                accuracy_val_top_5(logits, targets)
    fabric.barrier()
    loss_val = fabric.all_reduce(loss_val, reduce_op="mean") / len(dataloader)
    val_acc = accuracy_val.compute()
    if hparams.dataset == "imagenet" or hparams.dataset == "cifar100":
        val_acc_top_5 = accuracy_val_top_5.compute()
    fabric.log_dict({
            "sample/val/loss": loss_val,
            "sample/val/accuracy": val_acc,
        },
        step=samples_processed)
    fabric.log_dict({
            "step/val/loss": loss_val,
            "step/val/accuracy": val_acc,
        },
        step=iter_num)
    if hparams.dataset == "imagenet" or hparams.dataset == "cifar100":
        fabric.log_dict({
                "sample/val/accuracy_top_5": val_acc_top_5,
            },
            step=samples_processed)
        fabric.log_dict({
                "step/val/accuracy_top_5": val_acc_top_5,
            },
            step=iter_num)
    fabric.barrier()
    if fabric.global_rank == 0:
        if hparams.dataset == "imagenet" or hparams.dataset == "cifar100":
            fabric.print(
                "iter {}: val loss {:.6f}, val acc {:.2f}%, top 5 val acc {:.2f}%, global batch size {}".format(
                    iter_num,
                    loss_val,
                    100.0 * val_acc,
                    100.0 * val_acc_top_5,
                    batch_size
                    )
                )
        else:
            fabric.print(
                "iter {}: val loss {:.6f}, val acc {:.2f}%, global batch size {}".format(
                    iter_num,
                    loss_val,
                    100.0 * val_acc,
                    batch_size
                    )
                )
    accuracy_val.reset()
    if hparams.dataset == "imagenet" or hparams.dataset == "cifar100":
        accuracy_val_top_5.reset()


def main(hparams):
    torch.set_float32_matmul_precision("medium")
    # torch.set_float32_matmul_precision("high") # use float32 for matmuls (default is float16)
    # torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    if hparams.test:
        if hparams.test_type == "norm":
            name = f"lightning_logs_adaptive_local_sgd_{hparams.model}_{hparams.dataset}_eta{hparams.eta}_optimizer_{hparams.optimizer}_"
        elif hparams.test_type == "ip":
            name = f"lightning_logs_adaptive_local_sgd_{hparams.model}_{hparams.dataset}_theta{hparams.theta}_local_steps{hparams.local_sgd_steps}"
        elif hparams.test_type == "ip+orth":
            name = f"lightning_logs_adaptive_local_sgd_{hparams.model}_{hparams.dataset}_theta{hparams.theta}_local_steps{hparams.local_sgd_steps}"
    elif hparams.stagewise_ramp_up:
        name = f"lightning_logs_adaptive_local_sgd_{hparams.model}_{hparams.dataset}_stagewise_{'_'.join(str(c) for c in hparams.global_batch_sizes)}_local_steps{hparams.local_sgd_steps}"
    elif hparams.linear_ramp_up:
        name = f"lightning_logs_adaptive_local_sgd_{hparams.model}_{hparams.dataset}_linear_{hparams.linear_ramp_up_increment}_local_steps{hparams.local_sgd_steps}"
    else:
        name = f"lightning_logs_adaptive_local_sgd_{hparams.model}_{hparams.dataset}_const{hparams.batch_size}_local_steps{hparams.local_sgd_steps}"
    
    logger = TensorBoardLogger(root_dir=(hparams.out_dir / "logs"), name=name)
    strategy = DDPStrategy()

    if hparams.slurm:
        fabric = L.Fabric(devices=hparams.devices, num_nodes=hparams.num_nodes, precision=hparams.precision, strategy=strategy, loggers=logger)
        fabric.launch()
    else:
        fabric = L.Fabric(strategy=strategy, loggers=logger)

    fabric._loggers[0].log_hyperparams(vars(hparams))
    master_process = fabric.global_rank == 0

    if master_process:
        os.makedirs(hparams.out_dir, exist_ok=True)

    fabric.seed_everything(hparams.seed + fabric.global_rank)  # instead of torch.manual_seed(...)

    # choose a model
    fabric.print(f"Creating model '{hparams.model}'")
    if hparams.model == "cnn":
        if hparams.dataset == "mnist":
            model = MNISTModule()
        elif hparams.dataset == "cifar10":
            model = CIFAR10Module()
    elif hparams.model == "logistic":
        model = LogisticRegression()
    elif hparams.model == "resnet18":
        if hparams.dataset == "cifar10":
            model = LitResnet(hparams.model, n_class=10)
        elif hparams.dataset == "cifar100":
            model = LitResnet(hparams.model, n_class=100)
    elif hparams.model == "resnet50":
        if hparams.dataset == "cifar10":
            model = LitResnet(hparams.model, n_class=10)
        elif hparams.dataset == "cifar100":
            model = LitResnet(hparams.model, n_class=100)

    if hparams.dataset == "imagenet":        
        model = ImageNetModule(hparams.model)

    # compile the model
    if hparams.compile:
        print("compiling the model... (takes a ~minute)")
        # unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # set up model (for distributed training)
    model = fabric.setup(model)

    # choose an optimizer
    if hparams.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=hparams.learning_rate)
    elif hparams.optimizer == "sgdm":
        optimizer = torch.optim.SGD(model.parameters(), lr=hparams.learning_rate, momentum=0.9, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, betas=(hparams.beta1, hparams.beta2))
    elif hparams.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.learning_rate)
    elif hparams.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=hparams.learning_rate)
    elif hparams.optimizer == "adagradnorm":
        optimizer = Adagradnorm(model.parameters(), lr=hparams.learning_rate)
    else:
        raise NotImplementedError("Optimizer not implemented")
    
    # set up optimizer (for distributed training)
    optimizer = fabric.setup_optimizers(optimizer)

    # set initial learning rate to zero if warmup is enabled
    if hparams.decay_lr:
        if hparams.scheduler == 'cosine' or hparams.scheduler == 'linear':
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0

    # set up datasets and dataloaders
    train_set, val_set, train_dataloader, val_dataloader = prepare_data(fabric, hparams, master_process)
    
    # Training loop
    train(fabric, model, optimizer, hparams, train_set, val_set, train_dataloader, val_dataloader)


if __name__ == "__main__":
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    model_names.extend(["cnn", "logistic"])

    parser = ArgumentParser(description="Fabric AdAdaGrad Example")
    # I/O
    parser.add_argument("--out_dir", type=Path, default='out', help="Output directory (default: 'out')")
    parser.add_argument("--eval_interval", type=int, default=1, help="evaluate every n iterations (default: 1)")
    parser.add_argument("--log_interval", type=int, default=1, help="log every n iterations (default: 1)")
    parser.add_argument("--data_dir", type=Path, default='data', help="Dataset directory (default: 'data')")
    parser.add_argument("--slurm", type=bool, default=False, help="Whether to use Slurm (default: False)")
    # hardware
    parser.add_argument("--devices", type=int, default=4, help="devices (default: 4)")
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes (default: 1)")
    parser.add_argument("--precision", type=str, default="32-true", help="precision (default: 32-true)")
    # model
    parser.add_argument("--model", type=str, default='resnet18', choices=model_names, 
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    # data
    parser.add_argument("--dataset", type=str, default='mnist', help="dataset name (default: mnist; cifar10)")
    parser.add_argument("--shuffle", type=bool, default=True, help="whether to shuffle the dataset (default: True)")
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers (CPU core) for dataloader (default: 16)")
    parser.add_argument("--batch_ahead", type=int, default=3, help="number of batches to prefetch (default: 3)")
    # used to simulate larger batch sizes
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps (default: 1)")
    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer")
    parser.add_argument("--max_lr", type=float, default=1e-3, metavar="LR", help="max learning rate if not decaying lr (default: 1e-3)")
    parser.add_argument("--batch_size", type=int, default=64, help="global batch size")
    parser.add_argument("--max_micro_batch_size", type=int, default=5000, help="maximum micro batch size (default: 5000)")
    parser.add_argument("--max_test_micro_batch_size", type=int, default=2000, help="maximum micro batch size for testing (default: 2000)")
    parser.add_argument("--max_batch_size", type=int, default=50000, help="maximum global batch size (default: 50000)")
    parser.add_argument("--learning_rate", type=float, default=6e-4, metavar="LR", help="learning rate (default: 6e-4)")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay (default: 5e-4)")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 (default: 0.9)")    
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 (default: 0.95)")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="clip gradients at this value, or disable if == 0.0 (default: 0.0)")
    # iteration-based vs. sample-based training
    parser.add_argument("--training_type", type=str, default='sample', help="iteration-based vs. sample-based training (default: sample; iter)")
    parser.add_argument("--max_iters", type=int, default=750, help="total number of training iterations (default: 600000)")
    parser.add_argument("--max_samples", type=int, default=150000000, metavar="N", help="total number of training samples (default: 150000000)")
    # learning rate decay settings
    parser.add_argument("--decay_lr", type=bool, default=True, help="whether to decay the learning rate")
    parser.add_argument("--lr_decay_type", type=str, default='sample', help="iteration-based vs. sample-based lr decay (default: sample; iter)")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="minimum learning rate, should be ~= learning_rate/10 per Chinchilla (default: 6e-5)")
    # iteration-based learning rate decay settings
    parser.add_argument("--warmup_iters", type=int, default=2000, help="how many steps to warm up for (default: 2000)")
    parser.add_argument("--lr_decay_iters", type=int, default=600000, help="should be ~= max_iters per Chinchilla (default: 600000)")    
    # sample-based learning rate decay settings
    parser.add_argument("--warmup_samples", type=int, default=4000000, help="how many samples to warm up for (default: 4000000)")
    parser.add_argument("--lr_decay_samples", type=int, default=150000000, help="should be ~= max_samples per Chinchilla (default: 150000000)")
    # learning rate scheduler
    parser.add_argument("--scheduler", type=str, default='cosine', help="learning rate scheduler with warmup (default: cosine; linear, multistep)")
    parser.add_argument("--milestones", type=list, default=[80,120], help="list of milestones (default: [80,120])")
    parser.add_argument("--gamma", type=float, default=0.1, help="gamma for multistep lr scheduler (default: 0.1)")
    # tests
    parser.add_argument("--test_interval", type=int, default=1, help="test every n iterations (default: 1)")
    parser.add_argument("--test", type=bool, default=True, help="whether to test for batch size ramp-up")
    parser.add_argument("--test_type", type=str, default="norm", help="name of test for batch size ramp-up (default: norm; ip, ip+orth)")
    parser.add_argument("--eta", type=float, default=0.1, help="eta for norm test")
    parser.add_argument("--theta", type=float, default=0.1, help="theta for inner product test")
    parser.add_argument("--nu", type=float, default=0.1, help="nu for orthogonality test")
    parser.add_argument("--ramp_up_type", type=str, default="test", help="ramp up type (default: test; factor)")
    parser.add_argument("--ramp_up_factor", type=float, default=2., help="ramp up factor (default: 2.0)")
    # batch size rampup settings
    parser.add_argument("--stagewise_ramp_up", type=bool, default=False, help="whether to stagewise ramp up the batch size")
    parser.add_argument("--samples_stages", type=list, default=[], help="number of samples of all stages (default: None; computed as equal for each stage)")
    parser.add_argument("--global_batch_sizes", type=list, default=[512,1024,2048,4096,8192,16384,32768,60000], help="list of global batch sizes (default: [512,1024,2048,4096,8192,16384,32768])")
    parser.add_argument("--linear_ramp_up", type=bool, default=False, help="whether to linear ramp up the batch size (default: False)")
    parser.add_argument("--linear_ramp_up_increment", type=int, default=256, help="batch size increment of linear ramp up (default: 256)")
    # system
    parser.add_argument("--compile", type=bool, default=False, help="if True, use PyTorch 2.0 to compile the model to be faster")
    # seed
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")

    hparams = parser.parse_args()
    main(hparams)