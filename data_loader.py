import h5py
import numpy as np
from os import PathLike
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict, List, Union, Optional

class DataLoaderMusicAE:
    def __init__(
        self,
        data_dir: Union[str, PathLike],
        prefix: str,
        split: str,
    ):
        """
        Create DataLoader
        """
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        elif not isinstance(data_dir, PathLike):
            raise TypeError("data_dir is neither str or os.PathLike")

        self.dataset: h5py.File = h5py.File(str(data_dir / f'{prefix}-{split}.h5'), 'r')
        self.sample_count: int = int(self.dataset['sample_count'][()])

    def data_count(self) -> int:
        return self.sample_count

    def load_data(self, index: int) -> Dict[str, Any]:
        # TODO pad melody??
        index_str = str(index)
        return {k: self.dataset[index_str][k][...] for k in self.dataset[index_str].keys()}

# Helper class for PyTorch Dataset API, which supports automatic batching
class DatasetMusicAE(Dataset):
    def __init__(
        self,
        data_dir: Union[str, PathLike],
        prefix: str,
        split: str,
    ):
        super(DatasetMusicAE, self).__init__()
        self.data_loader = DataLoaderMusicAE(data_dir, prefix, split)

    def __len__(self):
        return self.data_loader.data_count()

    def __getitem__(self, index):
        return self.data_loader.load_data(index)
