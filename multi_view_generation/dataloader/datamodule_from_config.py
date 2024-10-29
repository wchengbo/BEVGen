from typing import Optional #Optional[x]用于表示某个值可为特定类型x，也可为None

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, RandomSampler, Subset, Dataset

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, 
                batch_size: int, 
                num_workers: int,#数据加载的工作线程数量
                pin_memory: bool,#是否将数据固定到内存中（在使用 GPU 时常见）
                train: Optional[Dataset] = None,
                validation: Optional[Dataset] = None,
                test: Optional[Dataset] = None,
                smoke_test: bool = False,#通常用于快速测试代码，不跑完整数据集，只运行小样本的简单测试。为 True 时，模型会使用少量数据检查代码是否正常运行
                small_val: bool = False,#是否使用小型验证集，可以在代码调试或资源有限时运行更少的数据
                val_batch_size: Optional[int] = None,
                num_gpus: Optional[int] = None,#指定训练时要使用的 GPU 数量
                single_val: bool = False,#是否仅使用单次验证。用于控制在训练期间是否使用一次验证
                shuffle_test: bool = True,#测试数据是否随机打乱，适用于测试集的样本随机化
                large_mini_dataset: bool = False#是否使用较大的 mini 数据集，用于设置小型数据集的大小
                ):
        super().__init__()

        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.smoke_test = smoke_test
        self.small_val = small_val
        self.num_workers = 0 if self.smoke_test else num_workers
        self.val_batch_size = val_batch_size
        self.num_gpus = num_gpus
        self.single_val = single_val
        self.shuffle_test = shuffle_test    
        self.large_mini_dataset = large_mini_dataset

        self.data_train: Optional[Dataset] = train
        self.data_val: Optional[Dataset] = validation
        self.data_test: Optional[Dataset] = test

    def setup(self, stage: Optional[str] = None):
        if self.data_train is not None:
            print(f'Train dataset has {len(self.data_train)} samples')
        
        if self.data_val is not None:
            print(f'Val dataset has {len(self.data_val)} samples')

        if self.data_test is not None:
            print(f'Test dataset has {len(self.data_test)} samples')

    def get_random_subset(self, dataset, samples, replacement=False):#从给定数据集中随机选出一个子集
        return Subset(dataset, list(RandomSampler(self.data_val, num_samples=samples, replacement=replacement)))#Subset(dataset,indices:list)
        #torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)

    def train_dataloader(self):
        return DataLoader(self.get_random_subset(self.data_train, 10000, True) if self.large_mini_dataset else self.data_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory, drop_last=True)
'''根据 large_mini_dataset 确定 DataLoader 的数据来源。如果 large_mini_dataset 为 True，则 DataLoader 会从 self.data_train 中随机采样 10,000 个样本组成一个子集；否则，DataLoader 会使用整个 data_train 数据集'''
    def val_dataloader(self):
        batch_size = self.val_batch_size if self.val_batch_size else self.batch_size
        if self.num_gpus:
            data_val = self.get_random_subset(self.data_val, self.num_gpus * batch_size)
        elif self.smoke_test or self.small_val:
            data_val = self.get_random_subset(self.data_val, 2 * batch_size)
        elif self.single_val:
            self.data_val.reset_selected()
            data_val = self.data_val
        else:
            data_val = self.data_val
        return DataLoader(dataset=data_val, batch_size=batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=self.shuffle_test)
