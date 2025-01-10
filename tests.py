import math

import torch
import torch.nn.functional as F

from litgpt.utils import chunked_cross_entropy

# compute per-sample loss
def compute_loss(params, buffers, model, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    predictions = torch.func.functional_call(model, (params, buffers), (batch,))
    loss = F.cross_entropy(predictions, targets)
    return loss

def compute_loss_gpt2(params, buffers, model, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    logits = torch.func.functional_call(model, (params, buffers), (batch,))
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss

def compute_loss_litgpt(params, buffers, model, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    logits = torch.func.functional_call(model, (params, buffers), (batch,))
    loss = chunked_cross_entropy(logits, targets, chunk_size=0)
    return loss

def compute_loss_transformer(params, buffers, model, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    output = torch.func.functional_call(model, (params, buffers), (batch, targets,))
    loss = F.nll_loss(output.view(-1, output.shape[-1]), targets.view(-1))
    return loss

# compute per-sample gradients
def compute_per_sample_gradients(model, data, targets):
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    ft_compute_grad = torch.func.grad(compute_loss)
    ft_compute_sample_grad = torch.func.vmap(ft_compute_grad, in_dims=(None, None, None, 0, 0))
    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, model, data, targets)
    del params, buffers, ft_compute_grad, ft_compute_sample_grad
    return ft_per_sample_grads

def compute_per_sample_gradients_gpt2(model, data, targets):
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    ft_compute_grad = torch.func.grad(compute_loss_gpt2)
    ft_compute_sample_grad = torch.func.vmap(ft_compute_grad, in_dims=(None, None, None, 0, 0))
    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, model, data, targets)
    del params, buffers, ft_compute_grad, ft_compute_sample_grad
    return ft_per_sample_grads

def compute_per_sample_gradients_litgpt(model, data, targets):
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    ft_compute_grad = torch.func.grad(compute_loss_litgpt)
    ft_compute_sample_grad = torch.func.vmap(ft_compute_grad, in_dims=(None, None, None, 0, 0))
    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, model, data, targets)
    del params, buffers, ft_compute_grad, ft_compute_sample_grad
    return ft_per_sample_grads

def compute_per_sample_gradients_transformer(model, data, targets):
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    ft_compute_grad = torch.func.grad(compute_loss_transformer)
    ft_compute_sample_grad = torch.func.vmap(ft_compute_grad, in_dims=(None, None, None, 0, 0), randomness='same')
    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, model, data, targets)
    del params, buffers, ft_compute_grad, ft_compute_sample_grad
    return ft_per_sample_grads

def compute_per_sample_gradients_losses(model, data, targets):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    ft_compute_grad = torch.func.grad_and_value(compute_loss)
    ft_compute_sample_grad = torch.func.vmap(ft_compute_grad, in_dims=(None, None, None, 0, 0))
    ft_per_sample_grads, per_sample_losses = ft_compute_sample_grad(params, buffers, model, data, targets)
    del params, buffers, ft_compute_grad, ft_compute_sample_grad
    return ft_per_sample_grads, per_sample_losses

def var_sample_grad(per_sample_gradient):
    batch_grad = torch.mean(per_sample_gradient, dim=0)
    return batch_grad, torch.var(per_sample_gradient, dim=0, correction=1)

def var_sample_batch_grad(per_sample_gradient):
    batch_size = per_sample_gradient.shape[0]
    batch_grad = torch.mean(per_sample_gradient, dim=0)
    inner_prod = torch.inner(per_sample_gradient.view(batch_size, -1), batch_grad.ravel())
    return batch_grad, torch.var(inner_prod, dim=0, correction=1)

def var_orthogonality(per_sample_gradient):
    batch_size = per_sample_gradient.shape[0]
    batch_grad = torch.mean(per_sample_gradient, dim=0)
    inner_prod = torch.inner(per_sample_gradient.view(batch_size, -1), batch_grad.ravel())
    ortho = per_sample_gradient.view(batch_size, -1) - torch.outer(inner_prod, batch_grad.ravel()) \
        / torch.linalg.vector_norm(batch_grad.ravel())**2
    del inner_prod
    return batch_grad, torch.var(ortho, dim=0, correction=1).sum()

def var_sample_batch_grad_orthogonality(per_sample_gradient):
    batch_size = per_sample_gradient.shape[0]
    batch_grad = torch.mean(per_sample_gradient, dim=0)
    inner_prod = torch.inner(per_sample_gradient.view(batch_size, -1), batch_grad.ravel())
    ortho = per_sample_gradient.view(batch_size, -1) - torch.outer(inner_prod, batch_grad.ravel()) \
        / torch.linalg.vector_norm(batch_grad.ravel())**2
    return batch_grad, torch.var(inner_prod, dim=0, correction=1), torch.var(ortho, dim=0, correction=1).sum()


def norm_test(per_sample_grads, eta):
    variance_one_norm = 0
    batch_grad_norm_squared = 0
    per_sample_grads = [per_sample_grad.detach() for per_sample_grad in per_sample_grads.values()]
    for per_sample_grad in per_sample_grads:
        batch_grad, variance = var_sample_grad(per_sample_grad)
        batch_size = per_sample_grad.shape[0]
        variance_one_norm += torch.linalg.vector_norm(variance.ravel(), ord=1).item()
        batch_grad_norm_squared += (torch.linalg.vector_norm(batch_grad.ravel())**2).item()
    del batch_grad, variance
    new_batch_size = variance_one_norm / (eta**2 * batch_grad_norm_squared)
    if new_batch_size > batch_size:
        batch_size = new_batch_size
    return math.ceil(batch_size), variance_one_norm, batch_grad_norm_squared

def inner_product_test(per_sample_grads, theta):
    batch_grad_norm_squared = 0
    variances = 0
    per_sample_grads = [per_sample_grad.detach() for per_sample_grad in per_sample_grads.values()]
    for per_sample_grad in per_sample_grads:
        batch_size = per_sample_grad.shape[0]
        batch_grad, variance = var_sample_batch_grad(per_sample_grad)
        batch_grad_norm_squared += (torch.linalg.vector_norm(batch_grad.ravel())**2).item()
        variances += variance.item()
    del batch_grad, variance
    new_batch_size = variances / (theta**2 * batch_grad_norm_squared**2)
    if new_batch_size > batch_size:
        batch_size = new_batch_size
    return math.ceil(batch_size), variances, batch_grad_norm_squared

def orthogonality_test(per_sample_grads, nu):
    batch_grad_norm_squared = 0
    variances = 0
    per_sample_grads = [per_sample_grad.detach() for per_sample_grad in per_sample_grads.values()]
    for per_sample_grad in per_sample_grads:
        batch_size = per_sample_grad.shape[0]
        batch_grad, variance = var_orthogonality(per_sample_grad)
        batch_grad_norm_squared += (torch.linalg.vector_norm(batch_grad.ravel())**2).item()
        variances += variance.item()
    del batch_grad, variance
    new_batch_size = variances / (nu**2 * batch_grad_norm_squared)
    if new_batch_size > batch_size:
        batch_size = new_batch_size
    return math.ceil(batch_size), variances, batch_grad_norm_squared

def inner_product_orthogonality_test(per_sample_grads, theta, nu):
    batch_grad_norm_squared = 0
    variances_ip = 0
    variances_orth = 0
    per_sample_grads = [per_sample_grad.detach() for per_sample_grad in per_sample_grads.values()]
    for per_sample_grad in per_sample_grads:
        batch_size = per_sample_grad.shape[0]
        batch_grad, variance_ip, variance_ortho = var_sample_batch_grad_orthogonality(per_sample_grad)
        batch_grad_norm_squared += (torch.linalg.vector_norm(batch_grad.ravel())**2).item()
        variances_ip += variance_ip.item()
        variances_orth += variance_ortho.item()
    del batch_grad, variance_ip, variance_ortho
    new_batch_size = max(variances_ip / (theta**2 * batch_grad_norm_squared**2), variances_orth / (nu**2 * batch_grad_norm_squared))
    if new_batch_size > batch_size:
        batch_size = new_batch_size
    return math.ceil(batch_size), variances_ip, variances_orth, batch_grad_norm_squared


## Distributed version (DDP)
def var_sample_grad_distributed(fabric, per_sample_gradient):
    batch_grad = torch.mean(per_sample_gradient, dim=0)
    batch_grad_variance = torch.var(per_sample_gradient, dim=0, correction=1)
    fabric.barrier()
    # Use the law of total expectation to find the global batch gradient
    all_batch_grad = fabric.all_reduce(batch_grad, reduce_op="mean")
    # Use the law of total variance to find the global batch variance
    all_batch_grad_variance = fabric.all_reduce(batch_grad_variance, reduce_op="mean") + \
        fabric.all_reduce(batch_grad**2, reduce_op="mean") - all_batch_grad ** 2
    del batch_grad, batch_grad_variance
    return all_batch_grad, all_batch_grad_variance

def var_sample_batch_grad_distributed(fabric, per_sample_gradient):
    batch_size = per_sample_gradient.shape[0]
    batch_grad = torch.mean(per_sample_gradient, dim=0)
    fabric.barrier()
    # Use law of total expectation to find the global batch gradient
    all_batch_grad = fabric.all_reduce(batch_grad, reduce_op="mean")
    del batch_grad
    inner_prod = torch.inner(per_sample_gradient.view(batch_size, -1), all_batch_grad.ravel())
    inner_prod_mean = torch.mean(inner_prod, dim=0)
    inner_prod_variance = torch.var(inner_prod, dim=0, correction=1)
    fabric.barrier()
    # Use law of total variance to find the global batch variance of 
    # the inner product of the per-sample gradients and the batch gradient
    all_inner_prod_variance = fabric.all_reduce(inner_prod_variance, reduce_op="mean") + \
        fabric.all_reduce(inner_prod_mean**2, reduce_op="mean") - fabric.all_reduce(inner_prod_mean, reduce_op="mean") ** 2
    del inner_prod, inner_prod_mean, inner_prod_variance
    return all_batch_grad, all_inner_prod_variance

def var_orthogonality_distributed(fabric, per_sample_gradient):
    batch_size = per_sample_gradient.shape[0]
    batch_grad = torch.mean(per_sample_gradient, dim=0)
    fabric.barrier()
    # Use law of total expectation to find the global batch gradient
    all_batch_grad = fabric.all_reduce(batch_grad, reduce_op="mean")
    del batch_grad
    inner_prod = torch.inner(per_sample_gradient.view(batch_size, -1), all_batch_grad.ravel())
    ortho = per_sample_gradient.view(batch_size, -1) - torch.outer(inner_prod, all_batch_grad.ravel()) \
        / torch.linalg.vector_norm(all_batch_grad.ravel())**2
    ortho_mean = torch.mean(ortho, dim=0)
    ortho_variance = torch.var(ortho, dim=0, correction=1).sum()
    fabric.barrier()
    # Use law of total variance to find the global batch variance of
    # the per-sample gradients orthogonal to the batch gradient
    all_ortho_variance = fabric.all_reduce(ortho_variance, reduce_op="mean") + \
        fabric.all_reduce(ortho_mean**2, reduce_op="mean").sum() - fabric.all_reduce(ortho_mean, reduce_op="mean").sum() ** 2
    del inner_prod, ortho, ortho_mean, ortho_variance
    return all_batch_grad, all_ortho_variance

def var_sample_batch_grad_orthogonality_distributed(fabric, per_sample_gradient):
    batch_size = per_sample_gradient.shape[0]
    batch_grad = torch.mean(per_sample_gradient, dim=0)
    fabric.barrier()
    # Use law of total expectation to find the global batch gradient
    all_batch_grad = fabric.all_reduce(batch_grad, reduce_op="mean")
    del batch_grad
    inner_prod = torch.inner(per_sample_gradient.view(batch_size, -1), all_batch_grad.ravel())
    inner_prod_mean = torch.mean(inner_prod, dim=0)
    inner_prod_variance = torch.var(inner_prod, dim=0, correction=1)
    ortho = per_sample_gradient.view(batch_size, -1) - torch.outer(inner_prod, all_batch_grad.ravel()) \
        / torch.linalg.vector_norm(all_batch_grad.ravel())**2
    ortho_mean = torch.mean(ortho, dim=0)
    ortho_variance = torch.var(ortho, dim=0, correction=1).sum()
    fabric.barrier()
    # Use law of total variance to find the global batch variance of 
    # the inner product of the per-sample gradients and the batch gradient
    all_inner_prod_variance = fabric.all_reduce(inner_prod_variance, reduce_op="mean") + \
        fabric.all_reduce(inner_prod_mean**2, reduce_op="mean") - fabric.all_reduce(inner_prod_mean, reduce_op="mean") ** 2
    all_ortho_variance = fabric.all_reduce(ortho_variance, reduce_op="mean") + \
        fabric.all_reduce(ortho_mean**2, reduce_op="mean").sum() - fabric.all_reduce(ortho_mean, reduce_op="mean").sum() ** 2
    del inner_prod, inner_prod_mean, inner_prod_variance, ortho, ortho_mean, ortho_variance
    return all_batch_grad, all_inner_prod_variance, all_ortho_variance


def norm_test_distributed(fabric, per_sample_grads, eta):
    all_batch_grad_variance_one_norm = 0
    all_batch_grad_norm_squared = 0
    # for per_sample_grad in per_sample_grads.values():
    per_sample_grads = [per_sample_grad.detach() for per_sample_grad in per_sample_grads.values()]
    for per_sample_grad in per_sample_grads:
        all_batch_grad, all_batch_grad_variance = var_sample_grad_distributed(fabric, per_sample_grad)
        batch_size = per_sample_grad.shape[0]
        all_batch_grad_variance_one_norm += torch.linalg.vector_norm(all_batch_grad_variance.ravel(), ord=1).item()
        all_batch_grad_norm_squared += (torch.linalg.vector_norm(all_batch_grad.ravel())**2).item()
    global_batch_size = batch_size * fabric.world_size
    new_global_batch_size = all_batch_grad_variance_one_norm / (eta**2 * all_batch_grad_norm_squared)
    if new_global_batch_size > global_batch_size:
        global_batch_size = new_global_batch_size      
    return math.ceil(global_batch_size), all_batch_grad_variance_one_norm, all_batch_grad_norm_squared

def inner_product_test_distributed(fabric, per_sample_grads, theta):
    all_batch_grad_norm_squared = 0
    all_inner_prod_variances = 0
    per_sample_grads = [per_sample_grad.detach() for per_sample_grad in per_sample_grads.values()]
    for per_sample_grad in per_sample_grads:
        all_batch_grad, all_inner_prod_variance = var_sample_batch_grad_distributed(fabric, per_sample_grad)
        batch_size = per_sample_grad.shape[0]
        all_batch_grad_norm_squared += (torch.linalg.vector_norm(all_batch_grad.ravel())**2).item()
        all_inner_prod_variances += all_inner_prod_variance.item()
    global_batch_size = batch_size * fabric.world_size
    new_global_batch_size = all_inner_prod_variances / (theta**2 * all_batch_grad_norm_squared**2)
    if new_global_batch_size > global_batch_size:
        global_batch_size = new_global_batch_size    
    return math.ceil(global_batch_size), all_inner_prod_variances, all_batch_grad_norm_squared

def orthogonality_test_distributed(fabric, per_sample_grads, nu):
    all_batch_grad_norm_squared = 0
    all_ortho_variance_one_norm = 0
    for per_sample_grad in per_sample_grads.values():
        all_batch_grad, all_ortho_variance = var_orthogonality_distributed(fabric, per_sample_grad)
        batch_size = per_sample_grad.shape[0]
        all_batch_grad_norm_squared += (torch.linalg.vector_norm(all_batch_grad.ravel())**2).item()
        all_ortho_variance_one_norm += torch.linalg.vector_norm(all_ortho_variance.ravel(), ord=1).item()
    global_batch_size = batch_size * fabric.world_size
    new_global_batch_size = all_ortho_variance_one_norm / (nu**2 * all_batch_grad_norm_squared)
    if new_global_batch_size > global_batch_size:
        global_batch_size = new_global_batch_size    
    return math.ceil(global_batch_size), all_ortho_variance_one_norm, all_batch_grad_norm_squared

def inner_product_orthogonality_test_distributed(fabric, per_sample_grads, theta, nu):
    all_batch_grad_norm_squared = 0
    all_inner_prod_variances = 0
    all_ortho_variance_one_norm = 0
    for per_sample_grad in per_sample_grads.values():
        all_batch_grad, all_inner_prod_variance, all_ortho_variance = var_sample_batch_grad_orthogonality_distributed(fabric, per_sample_grad)
        batch_size = per_sample_grad.shape[0]
        all_batch_grad_norm_squared += (torch.linalg.vector_norm(all_batch_grad.ravel())**2).item()
        all_inner_prod_variances += all_inner_prod_variance.item()
        all_ortho_variance_one_norm += torch.linalg.vector_norm(all_ortho_variance.ravel(), ord=1).item()
    global_batch_size = batch_size * fabric.world_size
    new_global_batch_size = max(all_inner_prod_variances / (theta**2 * all_batch_grad_norm_squared**2), all_ortho_variance_one_norm / (nu**2 * all_batch_grad_norm_squared))
    if new_global_batch_size > global_batch_size:
        global_batch_size = new_global_batch_size    
    return math.ceil(global_batch_size), all_inner_prod_variances, all_ortho_variance_one_norm, all_batch_grad_norm_squared 


## Gradient accumulation
def var_sample_grad_accumulate(micro_step, gradient_accumulation_steps, per_sample_gradients, 
                                        mean_per_sample_gradients, mean_per_sample_gradients_squared, mean_per_sample_gradients_variance):
    for key, per_sample_gradient in per_sample_gradients.items():
        batch_grad = torch.mean(per_sample_gradient, dim=0) / gradient_accumulation_steps
        batch_grad_squared = torch.mean(per_sample_gradient, dim=0)**2 / gradient_accumulation_steps
        batch_grad_variance = torch.var(per_sample_gradient, dim=0, correction=1) / gradient_accumulation_steps
        if micro_step == 0:
            mean_per_sample_gradients[key] = batch_grad
            mean_per_sample_gradients_squared[key] = batch_grad_squared
            mean_per_sample_gradients_variance[key] = batch_grad_variance
        else:
            mean_per_sample_gradients[key].add_(batch_grad)
            mean_per_sample_gradients_squared[key].add_(batch_grad_squared)
            mean_per_sample_gradients_variance[key].add_(batch_grad_variance)
    del batch_grad, batch_grad_squared, batch_grad_variance
    return mean_per_sample_gradients, mean_per_sample_gradients_squared, mean_per_sample_gradients_variance


## gradient accumulation for single device
def norm_test_accumulate(gradient_accumulation_steps, micro_batch_size,
                        mean_per_sample_gradients, mean_per_sample_gradients_squared, mean_per_sample_gradients_variance, eta):
    all_batch_grad_variance_one_norm = 0
    all_batch_grad_norm_squared = 0
    for key in mean_per_sample_gradients.keys():
        # Use the law of total expectation to find the global batch gradient
        all_batch_grad = mean_per_sample_gradients[key]
        # Use the law of total variance to find the global batch variance
        all_batch_grad_variance = mean_per_sample_gradients_variance[key] + mean_per_sample_gradients_squared[key] - all_batch_grad**2
        all_batch_grad_variance_one_norm += torch.linalg.vector_norm(all_batch_grad_variance.ravel(), ord=1).item()
        del all_batch_grad_variance
        all_batch_grad_norm_squared += (torch.linalg.vector_norm(all_batch_grad.ravel())**2).item()
        del all_batch_grad
    global_batch_size = micro_batch_size * gradient_accumulation_steps
    new_global_batch_size = all_batch_grad_variance_one_norm / (eta**2 * all_batch_grad_norm_squared)
    if new_global_batch_size > global_batch_size:
        global_batch_size = new_global_batch_size
    del mean_per_sample_gradients, mean_per_sample_gradients_squared, mean_per_sample_gradients_variance
    return math.ceil(global_batch_size), all_batch_grad_variance_one_norm, all_batch_grad_norm_squared


## Distributed version with gradient accumulation (only for the norm test)
def norm_test_distributed_accumulate(fabric, gradient_accumulation_steps, micro_batch_size,
                                    mean_per_sample_gradients, mean_per_sample_gradients_squared, mean_per_sample_gradients_variance, eta):
    all_batch_grad_variance_one_norm = 0
    all_batch_grad_norm_squared = 0
    fabric.barrier()
    all_batch_grads = fabric.all_reduce(mean_per_sample_gradients, reduce_op="mean")
    del mean_per_sample_gradients
    all_batch_grads_squared = fabric.all_reduce(mean_per_sample_gradients_squared, reduce_op="mean")
    del mean_per_sample_gradients_squared
    all_batch_grads_variance = fabric.all_reduce(mean_per_sample_gradients_variance, reduce_op="mean")
    del mean_per_sample_gradients_variance
    for key in all_batch_grads.keys():
        # Use the law of total expectation to find the global batch gradient
        all_batch_grad = all_batch_grads[key]
        # Use the law of total variance to find the global batch variance
        all_batch_grad_variance = all_batch_grads_variance[key] + all_batch_grads_squared[key] - all_batch_grad**2
        all_batch_grad_variance_one_norm += torch.linalg.vector_norm(all_batch_grad_variance.ravel(), ord=1).item()
        del all_batch_grad_variance
        all_batch_grad_norm_squared += (torch.linalg.vector_norm(all_batch_grad.ravel())**2).item()
        del all_batch_grad
    global_batch_size = micro_batch_size * fabric.world_size * gradient_accumulation_steps
    new_global_batch_size = all_batch_grad_variance_one_norm / (eta**2 * all_batch_grad_norm_squared)
    if new_global_batch_size > global_batch_size:
        global_batch_size = new_global_batch_size
    del all_batch_grads, all_batch_grads_squared, all_batch_grads_variance
    return math.ceil(global_batch_size), all_batch_grad_variance_one_norm, all_batch_grad_norm_squared
