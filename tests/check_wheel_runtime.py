"""Render from an extracted wheel outside the workspace with a fresh JIT cache.

Run: python -m tests.check_wheel_runtime /tmp/wheels/dsl_splat-0.1.0-py3-none-any.whl
The child process rejects every reference-extension import and only uses public
package settings. Installed Torch and Warp are the runtime dependencies.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

PROBE = r"""
import importlib.abc
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()/'package'))
class RejectReference(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.startswith(('diff_gaussian_rasterization', 'aaa_gaussian_rasterization')) or name == 'gsplat' or name.startswith('gsplat.'):
            raise ImportError('reference extension must not be imported')
sys.meta_path.insert(0, RejectReference())
import torch
import warp as wp
wp.config.kernel_cache_dir = str(Path.cwd()/'fresh-jit-cache')
import dsl_splat
from dsl_splat import (GaussianRasterizer, GaussianRasterizationSettings,
                     ExtendedSettings, SortSettings, SortMode, CullingSettings)
__AAA_RENDERER_IMPORT__
assert Path(dsl_splat.__file__).is_relative_to(Path.cwd()/'package')
device = 'cuda'
projection = torch.tensor([[1/.8,0,0,0], [0,1/.6,0,0],
                           [0,0,10/9.9,-1/9.9], [0,0,1,0]], device=device).T.contiguous()
config = GaussianRasterizationSettings(
    image_height=48, image_width=64, tanfovx=.8, tanfovy=.6,
    bg=torch.zeros(3,device=device), scale_modifier=1.,
    viewmatrix=torch.eye(4,device=device), projmatrix=projection,
    inv_viewprojmatrix=torch.linalg.inv(projection).contiguous(),
    sh_degree=0, campos=torch.zeros(3,device=device), prefiltered=False,
    settings=ExtendedSettings.from_dict(__AAA_DEFAULT_SETTINGS__),
    render_depth=False, debug=False)
renderer = GaussianRasterizer(config__AAA_RENDERER_ARGUMENT__)
for n,width,height,sh,channels in ((3,64,48,False,3), (17,97,71,True,3),
                                 (0,7,5,False,3), (3,64,48,False,7),
                                 (0,7,5,False,7), (1,64,48,False,1)):
    inputs = dict(means3D=torch.zeros(n,3,device=device),
                  scales=torch.full((n,3),.05,device=device),
                  rotations=torch.zeros(n,4,device=device),
                  opacities=torch.full((n,1),.7,device=device))
    inputs['means3D'][:,2] = 3.
    inputs['rotations'][:,0] = 1.
    if sh:
        inputs['shs'] = torch.zeros(n,1,3,device=device)
    else:
        inputs['features'] = torch.ones(n,channels,device=device)
    frame=config._replace(image_width=width,image_height=height,
                          bg=torch.zeros(channels,device=device))
    if n == 0:
        frame=frame._replace(bg=torch.linspace(.2,.4,channels,device=device))
    color,radii = renderer(**inputs,raster_settings=frame)
    assert color.shape == (channels,height,width) and radii.shape == (n,)
    assert bool(torch.isfinite(color).all())
    if n:
        assert bool((radii>0).all()) and float(color.max()) > .1
    else:
        assert not bool(color.any())
    print('PASS wheel runtime',n,f'{width}x{height}','SH' if sh else f'C{channels}')
    for name,value in inputs.items():
        value.requires_grad_()
    color,_ = renderer(**inputs,raster_settings=frame)
    color.mean().backward()
    assert all(value.grad is not None and bool(torch.isfinite(value.grad).all())
               for value in inputs.values())
    print('PASS wheel backward',n,f'{width}x{height}')
from dsl_splat.aaa.gsplat import rasterization
for value in inputs.values():
    value.grad = None
features = torch.ones(n,7,device=device,requires_grad=True)
images,alphas,meta = rasterization(
    inputs['means3D'], inputs['rotations'], inputs['scales'],
    inputs['opacities'].view(n), features,
    torch.eye(4,device=device)[None],
    torch.tensor([[[40.,0.,32.],[0.,40.,24.],[0.,0.,1.]]],device=device),
    64,48,render_mode='RGB',packed=False,absgrad=True)
assert images.shape == (1,48,64,7) and alphas.shape == (1,48,64,1)
(images.mean()+alphas.mean()).backward()
assert features.grad is not None and bool(torch.isfinite(features.grad).all())
assert meta['supports_means2d_grad'] and meta['means2d'].grad is not None
assert bool(torch.isfinite(meta['means2d'].absgrad).all())
print('PASS standalone wheel gsplat-style feature/alpha training')
from dsl_splat import gs3d
from dsl_splat.statistics import MaskedContributionCount
for value in inputs.values():
    value.grad = None
features.grad = None
pose = torch.eye(4,device=device)[None].requires_grad_()
result = gs3d.rasterize(
    inputs['means3D'], inputs['rotations'], inputs['scales'],
    inputs['opacities'].view(n),
    pose,
    torch.tensor([[[40.,0.,32.],[0.,40.,24.],[0.,0.,1.]]],device=device),
    64,48,features=features,absgrad=True,
    statistics=(MaskedContributionCount('hits',torch.ones(48,64,dtype=torch.bool,device=device)),))
assert result.features.shape == (1,48,64,7)
assert result.statistics['hits'].shape == (1,n)
assert bool((result.statistics['hits']>0).all())
(result.features.mean()+result.alpha.mean()).backward()
assert all(value.grad is not None and bool(torch.isfinite(value.grad).all())
           for name,value in inputs.items() if name != 'features')
assert features.grad is not None and bool(torch.isfinite(features.grad).all())
assert pose.grad is not None and bool(torch.isfinite(pose.grad).all()) and bool(pose.grad.any())
assert bool(torch.isfinite(result.means2d.absgrad).all())
print('PASS standalone wheel original 3DGS forward/backward and statistics')
torch.cuda.synchronize()
assert not any(name.startswith(('diff_gaussian_rasterization', 'aaa_gaussian_rasterization')) for name in sys.modules)
print('PASS standalone GPU rendering, reference imports blocked, fresh Warp JIT')
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--renderer", choices=["native", "python"], default="native")
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="aaa-wheel-runtime-") as directory:
        with ZipFile(options.wheel) as wheel:
            forbidden = [
                name
                for name in wheel.namelist()
                if name.endswith((".so", ".cu", ".cuh", ".cpp", ".h"))
            ]
            assert not forbidden, forbidden
            wheel.extractall(Path(directory) / "package")
        settings = json.loads(
            (Path(__file__).resolve().parents[1] / "config.json").read_text()
        )
        probe = PROBE.replace("__AAA_DEFAULT_SETTINGS__", repr(settings))
        probe = probe.replace(
            "__AAA_RENDERER_IMPORT__",
            (
                "from dsl_splat.aaa.render_hierarchical_warp import render_hierarchical_3d"
                if options.renderer == "python"
                else ""
            ),
        )
        probe = probe.replace(
            "__AAA_RENDERER_ARGUMENT__",
            ", renderer=render_hierarchical_3d" if options.renderer == "python" else "",
        )
        subprocess.run([sys.executable, "-I", "-c", probe], cwd=directory, check=True)


if __name__ == "__main__":
    main()
