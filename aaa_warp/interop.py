"""Torch/Warp stream interoperability, including externally captured streams."""
import torch
import warp as wp


def current_stream(device):
    """Reuse the active Warp wrapper when it is already Torch's current stream.

    Creating and destroying another wrapper for an externally captured stream
    unregisters that stream from Warp and loses its capture bookkeeping.
    """
    torch_stream = torch.cuda.current_stream(device)
    active = wp.get_stream(wp.device_from_torch(device))
    if active.cuda_stream == torch_stream.cuda_stream:
        return active
    return wp.stream_from_torch(torch_stream)
