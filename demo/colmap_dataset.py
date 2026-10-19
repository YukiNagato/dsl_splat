from dataclasses import dataclass
import torch
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

if __package__:
    from .colmap_utils import CameraModel, Image, qvec2rotmat, Reconstruction
else:
    from colmap_utils import CameraModel, Image, qvec2rotmat, Reconstruction


@dataclass
class ImageFrame:
    """RGB HWC image and OpenCV w2c: p_camera = p_world @ R.T + t."""

    K: torch.Tensor = None
    im_path: Path = None
    image: torch.Tensor = None
    w2c: torch.Tensor = None
    height: int = None
    width: int = None
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None
    gray_image: torch.Tensor | None = None

    def set_intrinsic(self, K: torch.Tensor) -> None:
        self.K = K
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

    @classmethod
    def from_colmap_image(cls, root: Path, im: Image):
        qvec = im.qvec
        tvec = im.tvec
        w2c = np.identity(4)
        w2c[:3, :3] = qvec2rotmat(qvec)
        w2c[:3, 3] = tvec
        im_path = root / im.name
        image = cv2.imread(str(im_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {im_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = torch.from_numpy(image).float() / 255
        w2c = torch.from_numpy(w2c).float()

        return cls(im_path=im_path, image=image, w2c=w2c, height=height, width=width)


class Dataset:
    def __init__(self, data_root: Path, im_stride=8):
        image_folder_name = "images"
        if im_stride > 1:
            image_folder_name += f"_{im_stride}"
        self.image_folder = data_root / image_folder_name
        colmap_root = data_root / "sparse" / "0"
        reconstruction = Reconstruction.from_binary(colmap_root)
        cameras = reconstruction.cameras
        intrinsic_dict = {}
        for cam_key, camera in cameras.items():
            if camera.model == CameraModel.PINHOLE:
                fx, fy, cx, cy = camera.params
            elif camera.model == CameraModel.SIMPLE_PINHOLE:
                fx, cx, cy = camera.params
                fy = fx
            else:
                raise ValueError(
                    f"Undistort COLMAP images first; unsupported camera: {camera.model.value}"
                )
            K = np.array(
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1],
                ]
            )
            K = torch.from_numpy(K).float()
            intrinsic_dict[cam_key] = K

        points_xyz = []
        points_rgb = []
        for point in reconstruction.points3D.values():
            points_xyz.append(point.xyz)
            points_rgb.append(point.rgb)
        self.points_xyz = (
            torch.tensor(np.array(points_xyz), dtype=torch.float32)
            if points_xyz
            else None
        )
        self.points_rgb = (
            torch.tensor(np.array(points_rgb), dtype=torch.float32)
            if points_rgb
            else None
        )
        colmap_images = list(reconstruction.images.values())
        self.frames = []
        for im in tqdm(colmap_images, desc="cache images"):
            frame = ImageFrame.from_colmap_image(self.image_folder, im)
            cam_id = im.camera_id
            K = intrinsic_dict[cam_id].clone()
            # Use actual image dimensions: downsampling can round width/height.
            K[0] *= frame.width / cameras[cam_id].width
            K[1] *= frame.height / cameras[cam_id].height
            frame.set_intrinsic(K)
            self.frames.append(frame)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx) -> ImageFrame:
        return self.frames[idx]
