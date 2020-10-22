import unittest

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from avo import MNISTDataModule
from avo.models import VaeGaussian
from avo.utils.vae_result_evaluator import show_vae_reconstruction, show_vae_generation


class TestVAEGaussian(unittest.TestCase):
    def setUp(self) -> None:
        self._device = "cuda:0"
        self._model = VaeGaussian()
        self._data_module = MNISTDataModule()

    # noinspection PyTypeChecker
    def test_training(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.fit(self._model, self._data_module)

    def test_testing(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        self._data_module.setup("test")
        trainer.test(self._model, self._data_module.test_dataloader())

    def test_show_vae_reconstruction(self):
        show_vae_reconstruction(self._model, self._data_module, dpi=150, figsize=(2.5, 6))
        plt.savefig("tmp/reconstruction.png")

    def test_show_vae_generation(self):
        show_vae_generation(self._model, dpi=150, figsize=(2.5, 6))
        plt.savefig("tmp/generation.png")
