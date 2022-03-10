import numpy as np
from loguru import logger
from torch.utils.data import Dataset
class MixedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = [datasets['agora'], datasets['curatedExpose'], datasets['h36m']]
        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets[0:2]])
        self.length = max([len(ds) for ds in self.datasets])
        """
        Data distribution inside each batch:
        Real = 70% pseudo 30%
        """
        self.partition = [0.7 * len(self.datasets[0])/length_itw,
                        0.7 * len(self.datasets[1])/length_itw,
                        0.3
                        ]
        self.partition = np.array(self.partition).cumsum()
    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(3):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]
    def __len__(self):
        return self.length