import unittest

import pytorch_lightning as pl
import torch
from avo.models import IAFELBO, IAFAVO, HouseholderELBO, HouseholderAVO
from avo.toy_dist import DFunction, PickleRick


class TestIAFELBO(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda:0"
        l_target = torch.cholesky(torch.tensor([[1., 0.95], [0.95, 1.]]))
        # target = PickleRick(torch.tensor([0., 0.]).unsqueeze(0).to(device), l_target.to(device))
        target = DFunction(device)
        self._model = IAFELBO(2, 10, target, hidden_dimension=10).cuda()

    def test_training(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.fit(self._model, self._model.data_loader(100))
        self._model.to("cuda:0")
        self._model.sample(100)

    def test_testing(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.test(self._model, self._model.data_loader(100))



class TestIAFAVO(unittest.TestCase):
    def setUp(self) -> None:
        self._device = "cpu"
        l_target = torch.cholesky(torch.tensor([[1., 0.95], [0.95, 1.]]))
        target = PickleRick(torch.tensor([0., 0.]).unsqueeze(0).to(self._device), l_target.to(self._device))

        self._model = IAFAVO(2, 10, target, hidden_dimension=10).to(self._device)

    def test_training(self):
        trainer = pl.Trainer(max_epochs=1, gpus=0)
        trainer.fit(self._model, self._model.data_loader(100))
        self._model.to(self._device)
        self._model.sample(100)

    def test_testing(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.test(self._model, self._model.data_loader(100))
class TestHVIELBO(unittest.TestCase):
    def setUp(self) -> None:
        device = "cuda:0"
        l_target = torch.cholesky(torch.tensor([[1., 0.95], [0.95, 1.]]))
        # target = PickleRick(torch.tensor([0., 0.]).unsqueeze(0).to(device), l_target.to(device))
        target = DFunction(device)
        self._model = HVIELBO(2, 10, target, hidden_dimension=10).cuda()

    def test_training(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.fit(self._model, self._model.data_loader(100))
        self._model.to("cuda:0")
        self._model.sample(100)

    def test_testing(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.test(self._model, self._model.data_loader(100))



class TestHVIAVO(unittest.TestCase):
    def setUp(self) -> None:
        self._device = "cpu"
        l_target = torch.cholesky(torch.tensor([[1., 0.95], [0.95, 1.]]))
        target = PickleRick(torch.tensor([0., 0.]).unsqueeze(0).to(self._device), l_target.to(self._device))

        self._model = HVIAVO(2, 10, target, hidden_dimension=10).to(self._device)

    def test_training(self):
        trainer = pl.Trainer(max_epochs=1, gpus=0)
        trainer.fit(self._model, self._model.data_loader(100))
        self._model.to(self._device)
        self._model.sample(100)

    def test_testing(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.test(self._model, self._model.data_loader(100))
