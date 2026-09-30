"""CUHK-SYSU for person ReID (the cropped re-identification version used by DG-ReID Protocol-2).

Expected layout under $FERREID_DATA_ROOT (verify against the copy you obtained; see docs/DEPLOYMENT.md):

    cuhksysu/cuhksysu4reid/
        train/    <pid>_<anything>.jpg      or   train/<pid>/<anything>.jpg
        query/    ...                            query/<pid>/...
        gallery/  ...                            gallery/<pid>/...

CUHK-SYSU crops come from street snaps and movie frames: there are no camera labels. Following the
usual convention, every train image gets camid 0, query images camid 0 and gallery images camid 1, so
that torchreid's "same pid and same camera" filter never removes a true match. The dataset is listed in
adapters/config_reid.py::NO_CAMERA_DOMAINS, so simulated annotation pairs an anchor with any other
image of the same person.

Reference sizes reported by DG-ReID papers for this split (check with scripts/check_datasets.py;
the numbers below are to be confirmed against the data, not assumed):
    train 5,532 ids / 15,088 images; query 2,900 ids / 2,900 images; gallery 2,900 ids / 5,447 images
"""
import glob
import os.path as osp
import re

from torchreid.data import ImageDataset

_PID = re.compile(r"^(-?\d+)")


def _pid_of(path, split_dir):
    rel = osp.relpath(path, split_dir)
    parts = rel.replace("\\", "/").split("/")
    token = parts[0] if len(parts) > 1 else osp.basename(path)  # <pid>/<file> or <pid>_<rest>.jpg
    m = _PID.match(token)
    if m is None:
        raise ValueError("cannot parse a person id from {} (expected <pid>_*.jpg or <pid>/*.jpg)".format(path))
    return int(m.group(1))


class CUHKSYSU(ImageDataset):
    dataset_dir = "cuhksysu/cuhksysu4reid"

    def __init__(self, root="", **kwargs):
        self.dataset_dir = osp.join(osp.abspath(osp.expanduser(root)), self.dataset_dir)
        dirs = [osp.join(self.dataset_dir, s) for s in ("train", "query", "gallery")]
        self.check_before_run(dirs)
        train = self._process(dirs[0], camid=0, relabel=True)
        query = self._process(dirs[1], camid=0)
        gallery = self._process(dirs[2], camid=1)
        super().__init__(train, query, gallery, **kwargs)

    @staticmethod
    def _process(split_dir, camid, relabel=False):
        paths = sorted(p for ext in ("jpg", "jpeg", "png")
                       for p in glob.glob(osp.join(split_dir, "**", "*." + ext), recursive=True))
        if not paths:
            raise RuntimeError("no images found under {}".format(split_dir))
        items = [(p, _pid_of(p, split_dir)) for p in paths]
        items = [(p, pid) for p, pid in items if pid >= 0]  # negative ids = junk / distractors
        if relabel:
            pid2label = {pid: i for i, pid in enumerate(sorted({pid for _, pid in items}))}
            items = [(p, pid2label[pid]) for p, pid in items]
        return [(p, pid, camid) for p, pid in items]
