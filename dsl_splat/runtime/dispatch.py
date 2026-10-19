"""Phase-local Torch descriptors for forward or backward kernel arguments.

Descriptors alias Torch storage and keep tensor owners for one phase. Cached
launches retain only host packets, rebound before use; scan/radix utilities
still use wp.array.
"""

from collections import OrderedDict
from typing import Any, Sequence
import torch
import ctypes
import threading

import warp as wp

_MAX_LAUNCH_COMMANDS = 64
_LaunchKey = tuple[wp.Kernel, int, int]
_LaunchPacket = tuple[Any, tuple[int, ...]]
_LaunchState = tuple[OrderedDict[_LaunchKey, _LaunchPacket], set[_LaunchKey]]
_LayoutKey = tuple[torch.device, torch.dtype, type, tuple[int, ...], tuple[int, ...]]


class _DescriptorLayouts(threading.local):
    """Bounded, per-thread layouts validated by Warp on their first use.

    Templates contain only shape/stride bytes. Each binding gets a fresh packet
    with the current pointer and Tensor owner, including after model resizing.
    """

    def __init__(self) -> None:
        self.templates: OrderedDict[_LayoutKey, ctypes.Structure] = OrderedDict()

    def bind(self, tensor: torch.Tensor, dtype: type) -> ctypes.Structure:
        key = (tensor.device, tensor.dtype, dtype, tuple(tensor.shape), tensor.stride())
        template = self.templates.get(key)
        if template is None:
            value = wp.from_torch(
                tensor, dtype=dtype, requires_grad=False, return_ctype=True
            )
            template = type(value).from_buffer_copy(value)
            template.data = 0
            template.grad = 0
            self.templates[key] = template
            if len(self.templates) > 64:
                self.templates.popitem(last=False)
            return value
        self.templates.move_to_end(key)
        value = type(template).from_buffer_copy(template)
        value.data = tensor.data_ptr()
        value._ref = tensor
        return value


_descriptor_layouts = _DescriptorLayouts()


class FrameBindings:
    """Reuse Tensor descriptors and stream wrappers within one execution phase.

    Each forward and backward creates its own bindings. Every new Tensor is
    bound to its current data pointer through a Warp-validated layout.
    Keeping owners on the live descriptors prevents Tensor-id reuse; cached
    layouts retain no Tensor owners or device pointers.
    """

    def __init__(self, launch_cache: "LaunchCache | None" = None) -> None:
        self.launch_cache = launch_cache
        self._descriptors: dict[tuple[int, type], ctypes.Structure] = {}
        self._streams: dict[tuple[torch.device, int], wp.Stream] = {}

    def array(
        self, tensor: torch.Tensor, dtype: type | None = None
    ) -> ctypes.Structure:
        if dtype is None:
            dtype = wp.dtype_from_torch(tensor.dtype)
        key = (id(tensor), dtype)
        value = self._descriptors.get(key)
        if value is None:
            value = _descriptor_layouts.bind(tensor, dtype)
            # The descriptor retains tensor as _ref, preventing id reuse.
            self._descriptors[key] = value
        return value


def kernel_arg(
    tensor: torch.Tensor | None,
    dtype: type | None = None,
    bindings: FrameBindings | None = None,
) -> ctypes.Structure | None:
    """Bind a Tensor or a null array; the descriptor cache is phase-local."""
    if tensor is None:
        return None
    if bindings is not None:
        return bindings.array(tensor, dtype)
    return wp.from_torch(tensor, dtype=dtype, requires_grad=False, return_ctype=True)


class LaunchCache:
    """Bounded thread-local Warp commands, with fresh arguments on each launch.

    Warp's parameter update API rebinds current shape/stride/pointer packets.
    Struct copies omit Tensor-owner attributes; FrameBindings retains the owners
    for the phase. The 64-entry cache holds no inputs or intermediate GPU buffers.
    Outputs are freshly allocated by ordinary stages. No CUDA Graph is used.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def __getstate__(self) -> dict[str, Any]:
        # Native handles and thread-local packets are rebuilt after copy/pickle.
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._local = threading.local()

    def launch(
        self,
        kernel: wp.Kernel,
        dim: int | Sequence[int],
        inputs: list[Any],
        outputs: list[Any],
        stream: wp.Stream,
        block_dim: int,
    ) -> None:
        state: _LaunchState | None = getattr(self._local, "state", None)
        if state is None:
            state = (OrderedDict(), set())
            self._local.state = state
        commands, busy = state
        key = (kernel, id(stream.device), block_dim)
        # A nested call cannot mutate the packet being prepared by its caller.
        if key in busy:
            return wp.launch(
                kernel,
                dim=dim,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
                block_dim=block_dim,
            )
        busy.add(key)
        try:
            # Cached commands must not extend the lifetime of phase-local Tensors.
            packed = [
                (
                    type(value).from_buffer_copy(value)
                    if isinstance(value, ctypes.Structure)
                    else value
                )
                for value in inputs + outputs
            ]
            dimensions = (dim,) if isinstance(dim, int) else tuple(dim)
            entry = commands.get(key)
            if entry is None:
                command = wp.launch(
                    kernel,
                    dim=dim,
                    inputs=packed,
                    stream=stream,
                    block_dim=block_dim,
                    record_cmd=True,
                )
                commands[key] = command, dimensions
                if len(commands) > _MAX_LAUNCH_COMMANDS:
                    commands.popitem(last=False)
            else:
                command, previous_dimensions = entry
                if dimensions != previous_dimensions:
                    command.set_dim(dim)
                    commands[key] = command, dimensions
                for index, value in enumerate(packed):
                    if value is None:
                        # Warp packs None as a null array through its typed API.
                        command.set_param_at_index(index, None)
                    else:
                        command.set_param_at_index_from_ctype(index, value)
                commands.move_to_end(key)
            return command.launch(stream=stream)
        finally:
            busy.remove(key)


def launch(
    kernel: wp.Kernel,
    *,
    dim: int | Sequence[int],
    inputs: list[Any],
    outputs: list[Any],
    stream: wp.Stream,
    block_dim: int = 256,
    bindings: FrameBindings | None = None,
) -> None:
    """Normal submission or reusable host command; stream is always explicit."""
    if bindings is not None and bindings.launch_cache is not None:
        return bindings.launch_cache.launch(
            kernel, dim, inputs, outputs, stream, block_dim
        )
    return wp.launch(
        kernel,
        dim=dim,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
        block_dim=block_dim,
    )


def launch_tiled(
    kernel: wp.Kernel,
    *,
    dim: int | Sequence[int],
    inputs: list[Any],
    outputs: list[Any],
    stream: wp.Stream,
    block_dim: int,
    bindings: FrameBindings | None = None,
) -> None:
    if bindings is not None and bindings.launch_cache is not None:
        dimensions = (dim,) if isinstance(dim, int) else tuple(dim)
        return launch(
            kernel,
            dim=(*dimensions, block_dim),
            inputs=inputs,
            outputs=outputs,
            stream=stream,
            block_dim=block_dim,
            bindings=bindings,
        )
    return wp.launch_tiled(
        kernel,
        dim=dim,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
        block_dim=block_dim,
    )
