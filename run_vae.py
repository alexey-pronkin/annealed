import avo.models
import argparse
from avo import MNISTDataModule
import pytorch_lightning as pl
import torch
from avo.utils.vae_result_evaluator import show_vae_reconstruction, show_vae_generation
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="VAE training and testing")
parser.add_argument('command', type=str, choices=["test", "train"], help="test or train")
parser.add_argument('model', type=str, choices=['vae', 'vae_hvi', 'vae_hvi_avo'],
                    help='Model')
parser.add_argument('-model-checkpoint', type=str, help='Path to checkpoint', default="checkpoint.ckpt")
parser.add_argument('-epoch', type=int, help='Epochs count', default=1)


def make_model(model_type):
    if model_type == "vae":
        return avo.models.VaeGaussian(encoder_config={"batch_norm": True}, beta=0.5, gamma=1 / 80.)
    elif model_type == "vae_hvi":
        return avo.models.VAEHVI(encoder_config={"batch_norm": True}, beta=0.5, gamma=1 / 80.)
    elif model_type == "vae_hvi_avo":
        return avo.models.VAEHVIAVO(encoder_config={"batch_norm": True}, beta=0.5, gamma=1 / 80.)


arguments = parser.parse_args()
if arguments.command == "train":
    data_module = MNISTDataModule(batch_size=256)
    model = make_model(arguments.model)
    trainer = pl.Trainer(max_epochs=arguments.epoch, gpus=1, progress_bar_refresh_rate=40)
    trainer.fit(model, data_module)
elif arguments.command == "test":
    model = make_model(arguments.model)
    model.load_state_dict(torch.load(arguments.model_checkpoint)["state_dict"])
    data_module = MNISTDataModule(batch_size=256)
    show_vae_reconstruction(model, data_module, dpi=150, figsize=(2.5, 6))
    plt.savefig("reconstruction.png")
    show_vae_generation(model, dpi=200, figsize=(2.5, 4))
    plt.savefig("generation.png")
    trainer = pl.Trainer(max_epochs=1, gpus=1)
    data_module.setup("test")
    trainer.test(model, data_module.test_dataloader())
