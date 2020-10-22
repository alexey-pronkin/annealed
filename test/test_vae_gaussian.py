import unittest
from avo.models import VaeGaussian
from avo import MNISTDataModule
import torch
import pytorch_lightning as pl


class TestVAEGaussian(unittest.TestCase):
    def setUp(self) -> None:
        self._device = "cuda:0"
        self._model = VaeGaussian().to(self._device)
        self._data_module = MNISTDataModule()

    # noinspection PyTypeChecker
    def test_training(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.fit(self._model, self._data_module)
        self._model.to(self._device)

    def test_testing(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.test(self._model, self._model.data_loader(100))
