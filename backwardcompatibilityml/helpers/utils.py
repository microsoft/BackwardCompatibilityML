# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import gc


def clean_from_gpu(tensors):
    """
    Utility function to clean tensors from the GPU.
    This is only intended to be used when investigating
    why memory usage is high.
    An in production solution should instead rely on
    correctly structuring your code so that Python
    garbage collection automatically removes the
    unreferenced tensors as they move out of function scope.
    Args:
        tensors: A list of tensor objects to clean from the GPU.
    Returns:
        None
    """
    for t in tensors:
        del t
    torch.cuda.empty_cache()


def show_allocated_tensors():
    """
    Attempts to print out the tensors in memory.
    Args:
        None
    Returns:
        None
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except Exception:
            pass


def get_gpu_mem():
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()


def generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        mem_all, mem_cached = get_gpu_mem()
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached,
        })

    return hook


def add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)

    h = mod.register_forward_hook(generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)

    h = mod.register_backward_hook(generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)


def log_mem(model, mem_log=None, exp=None):
    """
    Utility funtion for adding memory usage logging to a Pytorch model.

    Example usage:
        |  model = MyModel()
        |  hook_handles, mem_log = log_mem(model, exp="memory-profiling-experiment")
        |  ... then do a training run ...
        |  mem_log should now contain the results of the memory profiling experiment.
      
    Args:
        model: A pytorch model
        mem_log: Optional list object, which may contain data from previous
            profiling experiments.
        exp: String identifier for the profiling experiment name.

    Returns:
        A pair consisting of **mem_log** - either the same mem_log list 
        object that was passed in, or a newly constructed one, 
        that will contain the results of the logging, and 
        **hook_handles** - a list of handles for our logging hooks
        that will need to be cleared when we are done logging.
    """
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hook_handles = []
    for idx, module in enumerate(model.modules()):
        add_memory_hooks(idx, module, mem_log, exp, hook_handles)

    return hook_handles, mem_log


def remove_memory_hooks(hook_handles):
    """
    Clear the memory profiling hooks put in place by log_mem
    Args:
        hook_handles: A list of hook hndles to clear
    Returns:
        None
    """
    for hook_handle in hook_handles:
        hook_handle.remove()


def get_class_probabilities(batch_label_tensor):
    bin_count = torch.bincount(batch_label_tensor)
    return torch.tensor(bin_count, dtype=torch.float32) / torch.sum(bin_count)


def labels_to_probabilities(batch_class_labels, num_classes=None, batch_size=None):
    probabilities = torch.zeros(num_classes * batch_size, dtype=torch.float32)
    probabilities = probabilities.view(batch_size, num_classes)
    probabilities[torch.arange(probabilities.size(0)), batch_class_labels] = 1.0
    return probabilities


def sigmoid_to_labels(batch_sigmoids, discriminant_pivot=0.5):
    return torch.tensor((batch_sigmoids >= discriminant_pivot), dtype=torch.int)
