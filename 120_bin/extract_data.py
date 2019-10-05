"""Extract data from ZIP archive."""

import pathlib
from zipfile import ZipFile, ZipInfo
from pyprojroot import here


data_p: pathlib.Path = here("020_data")
assert data_p.is_dir()

archive = ZipFile(data_p / "facial-keypoints-detection.zip")
archive.extractall(data_p)

for fn in ("test.zip", "training.zip"):
    archive = ZipFile(data_p / fn)
    archive.extractall(data_p)
    (data_p / fn).unlink()
