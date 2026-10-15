from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import Enum
from io import BytesIO, StringIO
from pathlib import Path
import struct
from typing import BinaryIO, ClassVar, Mapping, TextIO, get_args, get_type_hints

import numpy as np


class CameraModel(Enum):
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"
    PINHOLE = "PINHOLE"
    SIMPLE_RADIAL = "SIMPLE_RADIAL"
    RADIAL = "RADIAL"
    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"
    FULL_OPENCV = "FULL_OPENCV"
    FOV = "FOV"
    SIMPLE_RADIAL_FISHEYE = "SIMPLE_RADIAL_FISHEYE"
    RADIAL_FISHEYE = "RADIAL_FISHEYE"
    THIN_PRISM_FISHEYE = "THIN_PRISM_FISHEYE"
    RAD_TAN_THIN_PRISM_FISHEYE = "RAD_TAN_THIN_PRISM_FISHEYE"
    SIMPLE_DIVISION = "SIMPLE_DIVISION"
    DIVISION = "DIVISION"
    SIMPLE_FISHEYE = "SIMPLE_FISHEYE"
    FISHEYE = "FISHEYE"
    EUCM = "EUCM"
    EQUIRECTANGULAR = "EQUIRECTANGULAR"

    @property
    def model_id(self) -> int:
        return _CAMERA_MODELS[self][0]

    @property
    def num_params(self) -> int:
        return _CAMERA_MODELS[self][1]


_CAMERA_MODELS = dict(zip(CameraModel, (
    (0, 3), (1, 4), (2, 4), (3, 5), (4, 8), (5, 8), (6, 12),
    (7, 5), (8, 4), (9, 5), (10, 12), (11, 16), (12, 4),
    (13, 5), (14, 3), (15, 4), (16, 6), (17, 2),
)))
_MODEL_IDS = {model.model_id: model for model in CameraModel}
_OBSERVATION_DTYPE = np.dtype([("xy", "<f8", (2,)), ("id", "<i8")])


def _read(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated COLMAP binary data")
    return data


def _unpack(stream: BinaryIO, fmt: str) -> tuple:
    return struct.unpack("<" + fmt, _read(stream, struct.calcsize("<" + fmt)))


def _line(stream: TextIO) -> str:
    for line in stream:
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    raise EOFError("No more COLMAP records")


def _numbers(values) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def _shape(array: np.ndarray, shape: tuple) -> None:
    if array.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {array.shape}")


class ModelRecord(ABC):
    """Single COLMAP record; binary output excludes the file's record count."""

    id_field: ClassVar[str]

    @property
    def id(self) -> int:
        return getattr(self, self.id_field)

    def to_txt(self, path: Path | None = None) -> str:
        data = self._to_txt()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data, encoding="utf-8")
        return data

    def to_binary(self, path: Path | None = None) -> bytes:
        data = self._to_binary()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return data

    @abstractmethod
    def _to_txt(self) -> str:
        ...

    @abstractmethod
    def _to_binary(self) -> bytes:
        ...

    @classmethod
    def from_txt(cls, data: str | Path):
        if isinstance(data, Path):
            data = data.read_text(encoding="utf-8")
        stream = StringIO(data)
        record = cls._read_txt(stream)
        if any(line.strip() and not line.lstrip().startswith("#") for line in stream):
            raise ValueError("Expected a single COLMAP record")
        return record

    @classmethod
    def from_binary(cls, data: bytes | Path):
        if isinstance(data, Path):
            data = data.read_bytes()
        stream = BytesIO(data)
        record = cls._read_binary(stream)
        if stream.read(1):
            raise ValueError("Trailing bytes after COLMAP record")
        return record

    @classmethod
    @abstractmethod
    def _read_txt(cls, stream: TextIO):
        ...

    @classmethod
    @abstractmethod
    def _read_binary(cls, stream: BinaryIO):
        ...


@dataclass
class Camera(ModelRecord):
    id_field: ClassVar[str] = "camera_id"

    camera_id: int
    model: CameraModel
    width: int
    height: int
    params: np.ndarray  # (P,), parameter order follows the camera model

    def _to_txt(self) -> str:
        _shape(self.params, (self.model.num_params,))
        return (f"{self.camera_id} {self.model.value} {self.width} {self.height} "
                f"{_numbers(self.params)}\n")

    @classmethod
    def _read_txt(cls, stream: TextIO) -> Camera:
        items = _line(stream).split()
        camera = cls(int(items[0]), CameraModel(items[1]), int(items[2]),
                     int(items[3]), np.array(items[4:], dtype=np.float64))
        _shape(camera.params, (camera.model.num_params,))
        return camera

    def _to_binary(self) -> bytes:
        _shape(self.params, (self.model.num_params,))
        return struct.pack("<IiQQ", self.camera_id, self.model.model_id,
                           self.width, self.height) + self.params.astype("<f8").tobytes()

    @classmethod
    def _read_binary(cls, stream: BinaryIO) -> Camera:
        camera_id, model_id, width, height = _unpack(stream, "IiQQ")
        if model_id not in _MODEL_IDS:
            raise ValueError(f"Unknown COLMAP camera model ID: {model_id}")
        model = _MODEL_IDS[model_id]
        params = np.frombuffer(_read(stream, 8 * model.num_params), dtype="<f8").copy()
        return cls(camera_id, model, width, height, params)


@dataclass
class Image(ModelRecord):
    id_field: ClassVar[str] = "image_id"

    image_id: int
    camera_id: int
    name: str  # Relative to the image root
    qvec: np.ndarray  # (4,), unit Hamilton quaternion (w, x, y, z), world-to-camera
    tvec: np.ndarray  # (3,), world-to-camera: X_cam = R @ X_world + t
    xys: np.ndarray  # (N, 2), pixel coordinates (x, y)
    point3D_ids: np.ndarray  # (N,), int64; -1 for untriangulated observations

    def _validate(self) -> None:
        _shape(self.qvec, (4,))
        _shape(self.tvec, (3,))
        _shape(self.xys, (len(self.point3D_ids), 2))
        _shape(self.point3D_ids, (len(self.xys),))
        if not np.issubdtype(self.point3D_ids.dtype, np.signedinteger):
            raise ValueError("point3D_ids must be signed integers (-1 means unmatched)")
        if np.any(self.point3D_ids < -1):
            raise ValueError("point3D_ids must be nonnegative or -1")
        if not self.name or self.name != self.name.strip() or any(c in self.name for c in "\0\r\n"):
            raise ValueError("Invalid COLMAP image name")

    def _to_txt(self) -> str:
        self._validate()
        header = (f"{self.image_id} {_numbers(self.qvec)} {_numbers(self.tvec)} "
                  f"{self.camera_id} {self.name}\n")
        points = " ".join(f"{_numbers(xy)} {int(point_id)}"
                          for xy, point_id in zip(self.xys, self.point3D_ids))
        return header + points + "\n"

    @classmethod
    def _read_txt(cls, stream: TextIO) -> Image:
        header = _line(stream).split(maxsplit=9)
        if len(header) != 10:
            raise ValueError("Invalid COLMAP image header")
        line = stream.readline()
        if not line:
            raise ValueError("Missing image observation line (may be empty)")
        points = line.split()
        if len(points) % 3:
            raise ValueError("Image observations must be (x, y, point3D_id) triples")
        xys = np.array([(float(points[i]), float(points[i + 1]))
                        for i in range(0, len(points), 3)], dtype=np.float64).reshape(-1, 2)
        ids = np.array([int(value) for value in points[2::3]], dtype=np.int64)
        image = cls(int(header[0]), int(header[8]), header[9],
                    np.array(header[1:5], dtype=np.float64),
                    np.array(header[5:8], dtype=np.float64), xys, ids)
        image._validate()
        return image

    def _to_binary(self) -> bytes:
        self._validate()
        points = np.empty(len(self.xys), dtype=_OBSERVATION_DTYPE)
        points["xy"], points["id"] = self.xys, self.point3D_ids
        return (struct.pack("<I7dI", self.image_id, *self.qvec, *self.tvec, self.camera_id)
                + self.name.encode("utf-8") + b"\0"
                + struct.pack("<Q", len(points)) + points.tobytes())

    @classmethod
    def _read_binary(cls, stream: BinaryIO) -> Image:
        header = _unpack(stream, "I7dI")
        name = bytearray()
        while (char := _read(stream, 1)) != b"\0":
            name.extend(char)
        count, = _unpack(stream, "Q")
        points = np.frombuffer(_read(stream, count * 24), dtype=_OBSERVATION_DTYPE)
        image = cls(header[0], header[8], name.decode("utf-8"),
                    np.array(header[1:5]), np.array(header[5:8]),
                    points["xy"].copy(), points["id"].copy())
        image._validate()
        return image


@dataclass
class Point3D(ModelRecord):
    id_field: ClassVar[str] = "point3D_id"

    point3D_id: int
    xyz: np.ndarray  # (3,), world coordinates
    rgb: np.ndarray  # (3,), uint8
    error: float  # Mean reprojection error in pixels
    track: np.ndarray  # (L, 2), int64; (image_id, zero-based index into Image.xys)

    def _validate(self) -> None:
        _shape(self.xyz, (3,))
        _shape(self.rgb, (3,))
        _shape(self.track, (len(self.track), 2))
        if not np.issubdtype(self.rgb.dtype, np.integer) or np.any((self.rgb < 0) | (self.rgb > 255)):
            raise ValueError("RGB must contain integers in [0, 255]")
        if not np.issubdtype(self.track.dtype, np.integer) or np.any((self.track < 0) | (self.track > 2**32 - 1)):
            raise ValueError("Track entries must be uint32-compatible integers")

    def _to_txt(self) -> str:
        self._validate()
        header = (f"{self.point3D_id} {_numbers(self.xyz)} "
                  f"{' '.join(str(int(v)) for v in self.rgb)} {self.error:.17g}")
        track = " ".join(str(int(v)) for v in self.track.flat)
        return header + (" " + track if track else "") + "\n"

    @classmethod
    def _read_txt(cls, stream: TextIO) -> Point3D:
        items = _line(stream).split()
        if len(items) < 8 or (len(items) - 8) % 2:
            raise ValueError("Invalid COLMAP point3D record")
        point = cls(int(items[0]), np.array(items[1:4], dtype=np.float64),
                    np.array(items[4:7], dtype=np.int64), float(items[7]),
                    np.array(items[8:], dtype=np.int64).reshape(-1, 2))
        point._validate()
        point.rgb = point.rgb.astype(np.uint8)
        return point

    def _to_binary(self) -> bytes:
        self._validate()
        return (struct.pack("<Q3d3BdQ", self.point3D_id, *self.xyz,
                            *self.rgb, self.error, len(self.track))
                + self.track.astype("<u4").tobytes())

    @classmethod
    def _read_binary(cls, stream: BinaryIO) -> Point3D:
        header = _unpack(stream, "Q3d3BdQ")
        track = np.frombuffer(_read(stream, header[8] * 8), dtype="<u4")
        return cls(header[0], np.array(header[1:4]),
                   np.array(header[4:7], dtype=np.uint8), header[7],
                   track.astype(np.int64).reshape(-1, 2))


@dataclass
class Reconstruction:
    cameras: dict[int, Camera] = field(default_factory=dict)
    images: dict[int, Image] = field(default_factory=dict)
    points3D: dict[int, Point3D] = field(default_factory=dict)

    def to_txt(self, path: Path | None = None) -> dict[str, str]:
        """Return COLMAP file contents and optionally write them to a directory."""
        result = {}
        for f in fields(self):
            name = f.name
            records = getattr(self, name)
            self._check_ids(records)
            result[f"{name}.txt"] = (f"# Number of {name}: {len(records)}\n"
                                     + "".join(records[key].to_txt() for key in sorted(records)))
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            for name, data in result.items():
                (path / name).write_text(data, encoding="utf-8")
        return result

    @classmethod
    def from_txt(cls, files: Mapping[str, str] | Path) -> Reconstruction:
        result = cls()
        hints = get_type_hints(cls)
        for f in fields(cls):
            name = f.name
            record_type = get_args(hints[name])[1]
            data = ((files / f"{name}.txt").read_text(encoding="utf-8")
                    if isinstance(files, Path) else files[f"{name}.txt"])
            stream = StringIO(data)
            records = getattr(result, name)
            while True:
                try:
                    record = record_type._read_txt(stream)
                except EOFError:
                    break
                cls._insert(records, record)
        return result

    def to_binary(self, path: Path | None = None) -> dict[str, bytes]:
        """Return COLMAP file contents and optionally write them to a directory."""
        result = {}
        for f in fields(self):
            name = f.name
            records = getattr(self, name)
            self._check_ids(records)
            result[f"{name}.bin"] = (struct.pack("<Q", len(records))
                                     + b"".join(records[key].to_binary() for key in sorted(records)))
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            for name, data in result.items():
                (path / name).write_bytes(data)
        return result

    @classmethod
    def from_binary(cls, files: Mapping[str, bytes] | Path) -> Reconstruction:
        result = cls()
        hints = get_type_hints(cls)
        for f in fields(cls):
            name = f.name
            record_type = get_args(hints[name])[1]
            data = ((files / f"{name}.bin").read_bytes()
                    if isinstance(files, Path) else files[f"{name}.bin"])
            stream = BytesIO(data)
            count, = _unpack(stream, "Q")
            for _ in range(count):
                cls._insert(getattr(result, name), record_type._read_binary(stream))
            if stream.read(1):
                raise ValueError(f"Trailing bytes in {name}.bin")
        return result


    @staticmethod
    def _check_ids(records: dict[int, ModelRecord]) -> None:
        for key, record in records.items():
            if key != record.id:
                raise ValueError(f"Dictionary keys must match {record.id_field}")

    @staticmethod
    def _insert(records: dict[int, ModelRecord], record: ModelRecord) -> None:
        if record.id in records:
            raise ValueError(f"Duplicate {record.id_field}: {record.id}")
        records[record.id] = record

        
def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [  # type: ignore
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
