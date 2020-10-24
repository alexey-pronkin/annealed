import unittest
from avo.models import HVIAVO
from avo.models import IATransform
import torch
from avo.toy_dist import PickleRick
import pytorch_lightning as pl
from test_config import device

class TestHVIAVOIA(unittest.TestCase):
    def setUp(self) -> None:
        self._device = device
        l_target = torch.cholesky(torch.tensor([[1., 0.95], [0.95, 1.]]))
        target = PickleRick(torch.tensor([0., 0.]).unsqueeze(0).to(self._device), l_target.to(self._device))
        transitions = [IATransform(2, [4, 4]) for _ in range(10)]
        self._model = HVIAVO(2, 10, target, transitions=transitions, is_flow=True).to(self._device)

    def test_training(self):
        trainer = pl.Trainer(max_epochs=1, gpus=0)
        trainer.fit(self._model, self._model.data_loader(100))
        self._model.to(self._device)
        self._model.sample(100)

    def test_testing(self):
        trainer = pl.Trainer(max_epochs=1, gpus=0)
        trainer.test(self._model, self._model.data_loader(100))
