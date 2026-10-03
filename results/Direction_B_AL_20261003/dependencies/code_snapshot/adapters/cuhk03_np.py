"""CUHK03-NP (labeled, 767/700 protocol) from the pre-split folder layout
CUHK03_pytorch/{train_all,query,gallery}/<pid>/<pair>_<pid>_<cam>_<idx>.jpg.
torchreid's built-in CUHK03 needs the raw .mat release, so we load this layout directly."""
import glob
import os.path as osp

from torchreid.data import ImageDataset


class CUHK03NP(ImageDataset):
    dataset_dir = "cuhk03/CUHK03_pytorch/CUHK03_pytorch"

    def __init__(self, root="", **kwargs):
        self.dataset_dir = osp.join(osp.abspath(osp.expanduser(root)), self.dataset_dir)
        self.check_before_run([self.dataset_dir])
        train = self._process(osp.join(self.dataset_dir, "train_all"), relabel=True)
        query = self._process(osp.join(self.dataset_dir, "query"))
        gallery = self._process(osp.join(self.dataset_dir, "gallery"))
        super().__init__(train, query, gallery, **kwargs)

    @staticmethod
    def _process(split_dir, relabel=False):
        paths = sorted(glob.glob(osp.join(split_dir, "*", "*.jpg")))
        pids = sorted({int(osp.basename(osp.dirname(p))) for p in paths})
        pid2label = {pid: i for i, pid in enumerate(pids)}
        data = []
        for p in paths:
            pid = int(osp.basename(osp.dirname(p)))
            pair, _, cam, _ = osp.splitext(osp.basename(p))[0].split("_")
            # 5 campus camera pairs x 2 views -> 10 distinct camera ids
            camid = (int(pair) - 1) * 2 + int(cam) - 1
            data.append((p, pid2label[pid] if relabel else pid, camid))
        return data
