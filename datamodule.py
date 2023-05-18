from pytorch_lightning import LightningDataModule
from newsgroups import NewsGroupsDataset
from nytimes import NYTimesDataset
from torch.utils.data import DataLoader

DATA_MAP = {
    '20news' : NewsGroupsDataset,
    'nytimes' : NYTimesDataset
}

class GeneralDataModule(LightningDataModule):
    def __init__(self, tokenizer, out_dim, dataset : str, batch_size : int):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        if dataset in DATA_MAP:
            self.dataset = DATA_MAP[dataset]
        else:
            raise NotImplementedError('dataset not implemented')

        self.train_dataset = self.dataset(split='train', out_dim = out_dim, tokenizer=tokenizer)
        self.val_dataset = self.dataset(split='val', out_dim = out_dim, tokenizer=tokenizer)
        self.test_dataset = self.dataset(split='test', out_dim = out_dim, tokenizer=tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False)

    def teardown(self, stage: str) -> None:
        pass

    def predict_dataloader(self):
        pass