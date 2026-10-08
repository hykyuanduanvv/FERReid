"""Build the cropped CUHK-SYSU ReID split (cuhksysu4reid, DG-ReID Protocol-2) from the person-search release.

Input: the unzipped release (dataset/Image/SSM/*.jpg, dataset/annotation/test/train_test/{Train,TestG100}.mat).
Output (adapters/cuhksysu.py layout, <pid>_<scene>_<n>.jpg, pid = the number of the release's "p<id>" name):
    train/    every box of the 5,532 training persons                     (expected 15,088 crops)
    query/    the query box of each of the 2,900 test persons             (expected  2,900 crops)
    gallery/  the boxes of those persons in their TestG100 gallery scenes (expected  5,447 crops)
Boxes are (xmin, ymin, width, height) as in the release README. Refuses to write into a non-empty directory.

  python scripts/convert_cuhksysu.py --src /data1/yangbin/dz/datasets/cuhksysu/dataset \
      --dst $FERREID_DATA_ROOT/cuhksysu/cuhksysu4reid
"""
import argparse
import os

import numpy as np
import scipy.io as sio
from PIL import Image


def _list(x):
    return list(np.atleast_1d(x))


def _crop(src_img_dir, imname, box, dst):
    x, y, w, h = [int(v) for v in box]
    with Image.open(os.path.join(src_img_dir, imname)) as im:
        im.convert("RGB").crop((x, y, x + w, y + h)).save(dst, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the release's dataset/ directory")
    ap.add_argument("--dst", required=True, help="output cuhksysu4reid directory")
    a = ap.parse_args()
    img_dir = os.path.join(a.src, "Image", "SSM")
    ann = os.path.join(a.src, "annotation", "test", "train_test")
    if os.path.isdir(a.dst) and os.listdir(a.dst):
        raise SystemExit("{} is not empty".format(a.dst))
    for s in ("train", "query", "gallery"):
        os.makedirs(os.path.join(a.dst, s), exist_ok=True)
    pid_of = lambda name: int(str(name).lstrip("p"))

    train = sio.loadmat(os.path.join(ann, "Train.mat"), squeeze_me=True, struct_as_record=False)["Train"]
    n = 0
    for person in _list(train):
        pid = pid_of(person.idname)
        for k, sc in enumerate(_list(person.scene)):
            _crop(img_dir, sc.imname, sc.idlocate,
                  os.path.join(a.dst, "train", "{}_{}_{}.jpg".format(pid, os.path.splitext(sc.imname)[0], k)))
            n += 1
    print("train: {} persons, {} crops".format(len(_list(train)), n))

    test = sio.loadmat(os.path.join(ann, "TestG100.mat"), squeeze_me=True, struct_as_record=False)["TestG100"]
    nq = ng = 0
    for item in _list(test):
        q = item.Query
        pid = pid_of(q.idname)
        _crop(img_dir, q.imname, q.idlocate,
              os.path.join(a.dst, "query", "{}_{}.jpg".format(pid, os.path.splitext(q.imname)[0])))
        nq += 1
        for k, g in enumerate(_list(item.Gallery)):
            if np.asarray(g.idlocate).size != 4:  # the person does not appear in this gallery scene
                continue
            _crop(img_dir, g.imname, g.idlocate,
                  os.path.join(a.dst, "gallery", "{}_{}_{}.jpg".format(pid, os.path.splitext(g.imname)[0], k)))
            ng += 1
    print("query: {} crops | gallery: {} crops".format(nq, ng))


if __name__ == "__main__":
    main()
