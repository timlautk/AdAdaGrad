import math
from bisect import bisect_right


# iteration-based learning rate decay scheduler (cosine with warmup)
def get_warmup_cosine_lr_iter(iter, warmup_iters, lr_decay_iters, min_lr, learning_rate):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # 2) if iter > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges from 0 to 1
    return min_lr + coeff * (learning_rate - min_lr)


# sample-based learning rate decay scheduler (cosine with warmup)
def get_warmup_cosine_lr_sample(sample, warmup_samples, lr_decay_samples, min_lr, learning_rate):
    # 1) linear warmup for warmup_samples samples
    if sample < warmup_samples:
        return learning_rate * sample / warmup_samples
    # 2) if sample > lr_decay_samples, return min learning rate
    if sample > lr_decay_samples:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (sample - warmup_samples) / (lr_decay_samples - warmup_samples)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges from 0 to 1
    return min_lr + coeff * (learning_rate - min_lr)


# iteration-based learning rate decay scheduler (linear with warmup)
def get_warmup_linear_lr_iter(iter, warmup_iters, lr_decay_iters, min_lr, learning_rate):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # 2) if iter > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use linear decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    return min_lr + decay_ratio * (learning_rate - min_lr)


# sample-based learning rate decay scheduler (linear with warmup)
def get_warmup_linear_lr_sample(sample, warmup_samples, lr_decay_samples, min_lr, learning_rate):
    # 1) linear warmup for warmup_samples samples
    if sample < warmup_samples:
        return learning_rate * sample / warmup_samples
    # 2) if sample > lr_decay_samples, return min learning rate
    if sample > lr_decay_samples:
        return min_lr
    # 3) in between, use linear decay down to min learning rate
    decay_ratio = (sample - warmup_samples) / (lr_decay_samples - warmup_samples)
    assert 0 <= decay_ratio <= 1
    return min_lr + decay_ratio * (learning_rate - min_lr)


# iteration-based learning rate decay scheduler (multistep)
def get_multistep_lr_iter(iter, learning_rate, milestones, gamma):
    return learning_rate * gamma ** bisect_right(milestones, iter)


# sample-based learning rate decay scheduler (multistep with warmup)
def get_multistep_lr_sample(sample, learning_rate, milestones, gamma):
    return learning_rate * gamma ** bisect_right(milestones, sample)


# stage-wise batch size ramp up scheduler
def get_global_batch_size(fabric, samples_processed, batch_size, global_batch_sizes, max_samples):
    n_stages = len(global_batch_sizes)
    samples_stage = max_samples // n_stages
    global_batch_size = global_batch_sizes[samples_processed // samples_stage]
    gradient_accumulation_steps = global_batch_size // (batch_size * fabric.world_size)
    return global_batch_size, gradient_accumulation_steps
