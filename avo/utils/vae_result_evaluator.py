import matplotlib.pyplot as plt
import numpy as np
import torch


def show_vae_generation(model, image_h=5, image_w=5, device="cpu", **kwargs):
    plt.figure(**kwargs)
    image_count = image_h * image_w
    with torch.no_grad():
        images = model.generate_x(image_count, device)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()[:, :, :, 0]
    for i in range(image_count):
        plt.subplot(image_h, image_w, i + 1)
        plt.imshow(np.clip(images[i], 0, 1), cmap="gray")
        plt.axis('off')


def show_vae_reconstruction(model, data_module, image_count=10, device="cpu", **kwargs):
    plt.figure(**kwargs)
    data_module.setup("test")
    batches = data_module.test_dataloader()
    input_data = None
    for batch in batches:
        input_data = batch[0]
        break
    model.eval()
    output_data = model.forward(input_data.to(device))[0]
    images = output_data.detach().cpu().permute(0, 2, 3, 1).numpy()[:, :, :, 0]
    input_images = input_data.detach().cpu().permute(0, 2, 3, 1).numpy()[:, :, :, 0]
    for i in range(image_count):
        plt.subplot(image_count, 2, 2 * i + 1)
        plt.imshow(np.clip(input_images[i], 0, 1), cmap="gray")
        plt.axis('off')
        plt.subplot(image_count, 2, 2 * i + 2)
        plt.imshow(np.clip(images[i], 0, 1), cmap="gray")
        plt.axis('off')
