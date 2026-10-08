"""Torch/Warp stream interoperability, including externally captured streams."""

from .dispatch import FrameBindings
import torch
import warp as wp


def contiguous_features(tensor: torch.Tensor) -> torch.Tensor:
    """Canonical inner strides for Warp's fixed-length vector descriptors.

    Torch regards empty/singleton dimensions as contiguous even when their
    stored stride is greater than one. Warp vector arrays require stride one.
    """
    value = tensor.contiguous()
    if value.stride(-1) != 1:
        value = value.clone(memory_format=torch.contiguous_format)
    return value


def current_stream(
    device: torch.device | str | int, *, bindings: FrameBindings | None = None
) -> wp.Stream:
    """Reuse the active Warp wrapper when it is already Torch's current stream.

    Creating and destroying another wrapper for an externally captured stream
    unregisters that stream from Warp and loses its capture bookkeeping.
    """
    torch_stream = torch.cuda.current_stream(device)
    key = (torch_stream.device, torch_stream.cuda_stream)
    if bindings is not None:
        cached = bindings._streams.get(key)
        if cached is not None:
            return cached
    active = wp.get_stream(wp.device_from_torch(device))
    if active.cuda_stream == torch_stream.cuda_stream:
        stream = active
    else:
        stream = wp.stream_from_torch(torch_stream)
    if bindings is not None:
        bindings._streams[key] = stream
    return stream
