# modify finetune.py from submodule
import argparse
from transformer2vae.model_t5 import T5VAE
from transformers import T5TokenizerFast
import math
from datamodule import GeneralDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from torch import nn

class LMHead(nn.Module):
    def __init__(self, vocab_size, input_dim, dropout_p = 0.2):
        super().__init__()
        # dropout
        self.drop = nn.Dropout(p = dropout_p)
        self.head = nn.Linear(input_dim, vocab_size)
        self.soft = nn.Softmax()
        self.flattener = nn.Flatten()

    def forward(self, x):
        x = self.soft(self.head(self.drop(self.flattener(x))))

        return x

BATCH_SIZE = 64
DATASET='20news'
NUM_WORKERS = 8
NUM_TRAINERS = 1
LOG_DIR = '../logs'
DEVICE = 'cpu'
NUM_EPOCHS = 100
OUT_DIM = 32

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    tb_logger = TensorBoardLogger(save_dir=LOG_DIR, name='t5-small')

    tokenizer = T5TokenizerFast.from_pretrained('t5-small')

    dm = GeneralDataModule(tokenizer, OUT_DIM, DATASET, BATCH_SIZE)

    iterations_per_training_epoch = math.ceil(
        len(dm.train_dataloader()) / BATCH_SIZE / NUM_TRAINERS
    )

    bow_head = LMHead(len(tokenizer.get_vocab()), input_dim = OUT_DIM * 512) # OUT_DIM x 512

    model = T5VAE(tokenizer=tokenizer,
                  iterations_per_training_epoch=iterations_per_training_epoch,
                  latent_dim=32,
                  pooling_strategy='max',
                  min_z=0.5,
                  fixed_reg_weight=None,
                  denoise_percentage=0.4,
                  base_model='t5-small',
                  bow_head = bow_head)

    runner = Trainer(logger=tb_logger,
                     log_every_n_steps=5,
                     check_val_every_n_epoch=3,
                     accelerator=DEVICE,
                     devices=NUM_TRAINERS,
                     max_epochs=NUM_EPOCHS,
                     profiler='pytorch', )

    model.val_dataloader = dm.val_dataloader

    runner.fit(model, datamodule=dm)