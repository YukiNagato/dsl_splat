"""Reusable CUDA Graph for the complete AAA 3D hierarchical forward.

Tensor addresses and shapes are bound at construction. In-place changes to
Gaussian and camera tensors are read on every replay. Capacity overflow is
bounded on the device and checked before returning a result; growth recaptures
and reruns the full pipeline rather than returning truncated rendering.
"""
import copy
from types import MappingProxyType

import torch
import warp as wp

from .binning import _FixedBinning
from .preprocess import preprocess
from .render_hierarchical_native import render_hierarchical_3d


class PreparedRasterizer3D:
    """Prepare a forward-only renderer for fixed tensor shapes and settings.

    Update bound tensors in place, then call ``render()`` on the CUDA stream
    that constructed this object. Returned output buffers are reused: consume
    or clone them before another call. Scalar settings and SH degree are fixed;
    changes to those or to tensor layouts require a new prepared renderer.

    By default, small workspaces reserve N times the number of tiles, a
    guaranteed upper bound that avoids per-frame host synchronization. Larger
    workspaces start at 125% of the observed pair count and check that count
    after each replay. ``pair_capacity`` explicitly selects either behavior:
    capacities below the upper bound are checked and grow automatically.
    Construction includes compilation, warmup and graph capture, and is not
    included in steady-state latency.
    """
    @torch.no_grad()
    def __init__(self, *, raster_settings, pair_capacity=None, **inputs):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('construct the prepared renderer outside CUDA Graph capture')
        self.inputs = MappingProxyType(dict(inputs))
        self.raster_settings = raster_settings._replace(
            settings=copy.deepcopy(raster_settings.settings))
        initial = preprocess(**self.inputs, raster_settings=self.raster_settings)
        settings = self.raster_settings.settings
        if not settings.eval_3D or int(settings.sort_settings.sort_mode) != 3:
            raise ValueError('requires eval_3D=True and sort_mode=HIER')
        if self.raster_settings.render_depth:
            raise ValueError('render_depth is not supported')
        sizes = settings.sort_settings.queue_sizes
        if sizes.tile_4x4 != 64 or sizes.tile_2x2 != 8 or sizes.per_pixel != 4:
            raise ValueError('only queue sizes 64/8/4 are supported')
        self._device = initial['radii'].device
        self._n = initial['radii'].shape[0]
        tile_count = ((self.raster_settings.image_width+15)//16) * (
            (self.raster_settings.image_height+15)//16)
        self._upper_bound = self._n * tile_count
        if self._upper_bound > 2**31-1:
            raise ValueError('N times the tile count exceeds the int32 prefix-sum limit')
        if pair_capacity is not None:
            if isinstance(pair_capacity, bool) or not isinstance(pair_capacity, int) or pair_capacity < 0:
                raise ValueError('pair_capacity must be a nonnegative integer')
            capacity = min(pair_capacity, self._upper_bound)
        elif self._upper_bound <= 262_144:
            capacity = self._upper_bound
        else:
            count = int(initial['tiles_touched'].sum().item())
            capacity = self._padded_capacity(count)
        self._owner = torch.cuda.current_stream(self._device)
        self._capture_stream = torch.cuda.Stream(device=self._device)
        self._warp_stream = wp.stream_from_torch(self._capture_stream)
        tensors = list(self.inputs.values()) + [getattr(self.raster_settings, name) for name in
            ('bg','viewmatrix','projmatrix','inv_viewprojmatrix','campos')]
        self._bindings = [(t, t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
                          for t in tensors if isinstance(t, torch.Tensor)]
        self.output = None
        self._build(capacity)
        # Initialize the captured state and ensure an explicitly undersized
        # initial capacity cannot expose a truncated constructor result.
        self.render()

    def _padded_capacity(self, count):
        return min(self._upper_bound, ((count + count//4 + 255)//256)*256)

    def _build(self, capacity):
        self._binning = _FixedBinning(self._n,self.raster_settings,self._device,capacity)
        self._capture_stream.wait_stream(self._owner)
        warp_stream = self._warp_stream
        with torch.cuda.stream(self._capture_stream), wp.ScopedStream(warp_stream,sync_enter=False):
            # Warm every kernel and the CUB scratch allocations on this stream.
            for _ in range(2):
                state = preprocess(**self.inputs,raster_settings=self.raster_settings)
                bins = self._binning.launch(state,self.raster_settings)
                self.output = render_hierarchical_3d(state,bins,self.raster_settings,
                                                      output=self.output)
                if self._n == 0:
                    self.output['color'].zero_()
        self._capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with wp.ScopedStream(warp_stream,sync_enter=False), torch.cuda.graph(
                graph,stream=self._capture_stream):
            # Register Torch's external capture so Warp retains CUB temporary
            # storage rather than freeing it while CUDA is recording the graph.
            wp.capture_begin(stream=warp_stream,external=True,capture_mode=wp.CaptureMode.GLOBAL)
            try:
                state = preprocess(**self.inputs,raster_settings=self.raster_settings)
                bins = self._binning.launch(state,self.raster_settings)
                render_hierarchical_3d(state,bins,self.raster_settings,output=self.output)
                if self._n == 0:
                    self.output['color'].zero_()
            finally:
                self._warp_graph = wp.capture_end(stream=warp_stream)
        self._graph = graph
        self._state = state  # Retain every captured preprocess output.
        self._owner.wait_stream(self._capture_stream)

    @property
    def pair_capacity(self):
        return self._binning.capacity

    @property
    def pair_count(self):
        """Device int32 tensor containing the latest actual pair count."""
        return self._binning.pair_count

    @property
    def checks_capacity(self):
        return self.pair_capacity < self._upper_bound

    @property
    def preprocessed(self):
        return self._state

    @property
    def bins(self):
        """Padded point_list and exact tile ranges; only ranges delimit pairs."""
        return self._binning.result

    @torch.no_grad()
    def render(self):
        """Replay on the owner stream; overflow grows and reruns before return."""
        if torch.cuda.current_stream(self._device) != self._owner:
            raise ValueError('use the CUDA stream that constructed this prepared renderer')
        for tensor, ptr, shape, stride, dtype, device in self._bindings:
            if (tensor.data_ptr() != ptr or tuple(tensor.shape) != shape or
                    tuple(tensor.stride()) != stride or tensor.dtype != dtype or tensor.device != device):
                raise ValueError('a bound tensor changed storage or layout; create a new prepared renderer')
        if self.checks_capacity and torch.cuda.is_current_stream_capturing():
            raise ValueError('checked capacity replay requires a host read; run it outside capture')
        while True:
            self._graph.replay()
            if not self.checks_capacity:
                return self.output
            needed = int(self.pair_count.item())
            if needed <= self.pair_capacity:
                return self.output
            self._build(max(self._padded_capacity(needed),
                            min(self._upper_bound,2*self.pair_capacity)))
