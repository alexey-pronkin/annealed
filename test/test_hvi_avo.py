import unittest
from avo.models import HVIAVO
import torch
from avo.toy_dist import PickleRick
import pytorch_lightning as pl


class TestHVIAVO(unittest.TestCase):
    def setUp(self) -> None:
        device = "cuda:0"
        l_target = torch.cholesky(torch.tensor([[1., 0.95], [0.95, 1.]]))
        target = PickleRick(torch.tensor([0., 0.]).unsqueeze(0).to(device), l_target.to(device))

        self._model = HVIAVO(2, 10, target, hidden_dimension=10).cuda()

    def test_training(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.fit(self._model, self._model.data_loader(100))
        self._model.to("cuda:0")
        self._model.sample(100)

    def test_testing(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.test(self._model, self._model.data_loader(100))
