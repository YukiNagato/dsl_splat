from dataclasses import dataclass
import torch
from pathlib import Path
import numpy as np
import cv2
from colmap_utils import Image, qvec2rotmat, Reconstruction
from tqdm import tqdm


@dataclass
class ImageFrame:
    K: torch.Tensor = None
    im_path: Path = None
    image: torch.Tensor = None
    w2c: torch.Tensor = None

    @classmethod
    def from_colmap_image(cls, root: Path, im: Image):
        qvec = im.qvec
        tvec = im.tvec
        w2c = np.identity(4)
        w2c[:3, :3] = qvec2rotmat(qvec)
        w2c[:3, 3] = tvec
        im_path = root / im.name
        image = cv2.imread(im_path)
        image = torch.from_numpy(image).float() / 255
        w2c = torch.from_numpy(w2c).float()
        return cls(im_path=im_path, image=image, w2c=w2c)


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
            if im_stride > 1:
                camera.width = int(camera.width / im_stride)
                camera.height = int(camera.height / im_stride)
                camera.params = camera.params / im_stride
            K = np.array(
                [
                    [camera.params[0], 0, camera.params[2]],
                    [0, camera.params[1], camera.params[3]],
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
        self.points_xyz = torch.from_numpy(np.array(points_xyz)).float()
        self.points_rgb = torch.from_numpy(np.array(points_rgb)).float()
        colmap_images = list(reconstruction.images.values())
        self.frames = []
        for im in tqdm(colmap_images, desc="cache images"):
            frame = ImageFrame.from_colmap_image(self.image_folder, im)
            cam_id = im.camera_id
            frame.K = intrinsic_dict[cam_id]
            self.frames.append(frame)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx) -> ImageFrame:
        return self.frames[idx]

