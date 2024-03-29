import torch

def current_memory_usage():
    """
    Returns the current memory usage in megabytes.

    Returns:
        float: The current memory usage in megabytes.
    """
    current_memory_usage = torch.cuda.memory_allocated()

    # Convert to megabytes
    current_memory_usage_mb = current_memory_usage / 1024 / 1024
    return current_memory_usage_mb