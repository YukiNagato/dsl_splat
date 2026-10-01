"""Frame-local Torch descriptors for forward kernel arguments.

Descriptors alias Torch storage and keep tensor owners for one forward. Cached
launches retain only host packets, rebound before use; scan/radix utilities
still use wp.array.
"""
import ctypes
import threading

import warp as wp


class FrameBindings:
    """Reuse descriptors for identical Tensor objects within one forward."""
    def __init__(self, launch_cache=None):
        self.launch_cache = launch_cache
        self._descriptors = {}

    def array(self, tensor, dtype=None):
        if dtype is None:
            dtype = wp.dtype_from_torch(tensor.dtype)
        key = (id(tensor), dtype)
        value = self._descriptors.get(key)
        if value is None:
            value = wp.from_torch(tensor, dtype=dtype, requires_grad=False, return_ctype=True)
            # from_torch's descriptor retains tensor as _ref, preventing id reuse.
            self._descriptors[key] = value
        return value


def kernel_arg(tensor, dtype=None, bindings=None):
    """Build a direct kernel descriptor; optional cache is scoped to one frame."""
    if bindings is not None:
        return bindings.array(tensor, dtype)
    return wp.from_torch(tensor, dtype=dtype, requires_grad=False, return_ctype=True)


class LaunchCache:
    """Thread-local host command packets, with current arguments on each launch.

    Array packets are copied as small host structs without their Tensor-owner
    attributes. FrameBindings keeps those owners alive for the entire forward;
    the cache therefore retains neither old inputs nor intermediate GPU buffers.
    Outputs are freshly allocated by the ordinary stages. No CUDA Graph is used.
    """
    def __init__(self):
        self._local = threading.local()

    def __getstate__(self):
        # Native handles and thread-local packets are rebuilt after copy/pickle.
        return {}

    def __setstate__(self, state):
        self.__init__()

    def launch(self, kernel, dim, inputs, outputs, stream, block_dim):
        state = getattr(self._local,'state',None)
        if state is None:
            state = ({},set())
            self._local.state = state
        commands,busy = state
        key = (kernel,id(stream.device),block_dim)
        # A nested call cannot mutate the packet being prepared by its caller.
        if key in busy:
            return wp.launch(kernel,dim=dim,inputs=inputs,outputs=outputs,
                             stream=stream,block_dim=block_dim)
        busy.add(key)
        try:
            values = inputs+outputs
            entry = commands.get(key)
            if entry is None:
                packed = [type(v).from_buffer_copy(v) if isinstance(v,ctypes.Structure) else v
                          for v in values]
                command = wp.launch(kernel,dim=dim,inputs=packed,stream=stream,
                                    block_dim=block_dim,record_cmd=True)
                dimensions = (dim,) if isinstance(dim,int) else tuple(dim)
                # Only scalars are compared between calls; every array is rebound.
                previous = [None if isinstance(v,ctypes.Structure) else v for v in values]
                # Warp accepts raw array descriptors verbatim. Keep those
                # owner-free packets at stable host addresses and copy new
                # descriptor bytes into them; the command already points here.
                arrays = [(index,ctypes.addressof(packet),ctypes.sizeof(packet))
                          for index,packet in enumerate(packed) if isinstance(packet,ctypes.Structure)]
                scalars = [index for index,v in enumerate(values) if not isinstance(v,ctypes.Structure)]
                commands[key] = command,dimensions,previous,arrays,scalars
            else:
                command,dimensions,previous,arrays,scalars = entry
                new_dimensions = (dim,) if isinstance(dim,int) else tuple(dim)
                if new_dimensions != dimensions:
                    command.set_dim(dim)
                    commands[key] = command,new_dimensions,previous,arrays,scalars
                for index,address,size in arrays:
                    ctypes.memmove(address,ctypes.addressof(values[index]),size)
                for index in scalars:
                    value = values[index]
                    if value != previous[index]:
                        command.set_param_at_index_from_ctype(index,value)
                        previous[index] = value
            return command.launch(stream=stream)
        finally:
            busy.remove(key)


def launch(kernel, *, dim, inputs, outputs, stream, block_dim=256, bindings=None):
    """Normal submission or reusable host command; stream is always explicit."""
    if bindings is not None and bindings.launch_cache is not None:
        return bindings.launch_cache.launch(kernel,dim,inputs,outputs,stream,block_dim)
    return wp.launch(kernel,dim=dim,inputs=inputs,outputs=outputs,stream=stream,block_dim=block_dim)


def launch_tiled(kernel, *, dim, inputs, outputs, stream, block_dim, bindings=None):
    if bindings is not None and bindings.launch_cache is not None:
        dimensions = (dim,) if isinstance(dim,int) else tuple(dim)
        return launch(kernel,dim=(*dimensions,block_dim),inputs=inputs,outputs=outputs,
                      stream=stream,block_dim=block_dim,bindings=bindings)
    return wp.launch_tiled(kernel,dim=dim,inputs=inputs,outputs=outputs,
                           stream=stream,block_dim=block_dim)
