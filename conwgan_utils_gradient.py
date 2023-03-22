import numpy as np
import torch
import torchvision
from model.conwgan_gradient import *

def weights_init(m):
    if isinstance(m, MyConvo2d): 
        if m.conv.weight is not None:
            if m.he_init:
                torch.nn.init.kaiming_uniform_(m.conv.weight)
            else:
                torch.nn.init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            torch.nn.init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)

def calc_gradient_penalty(device, discriminator, real_data, fake_data, batch_size = 64, LAMBDA = 10):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, discriminator.img_size, discriminator.img_size)
    alpha = alpha.to(device)

    fake_data = fake_data.view(batch_size, 3, discriminator.img_size, discriminator.img_size)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)   

    disc_interpolates, _ = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def get_noise(device, num_classes, labels=None, batch_size = 64, latent_size =100):
    if labels is None:
        labels = torch.randint(low=0,high=num_classes,size=([batch_size]))
    #attach label into noise
    one_hot_labels =  torch.nn.functional.one_hot(labels, num_classes = num_classes )
    random_latent_vectors = torch.normal(0, 3, size = (batch_size, latent_size))
    noise = torch.cat([random_latent_vectors, one_hot_labels], axis=1)
    noise = noise.to(device)
    labels = labels.to(device)
    return labels, noise

def generate_image(generator, num_classes, noise=None, batch_size = 64):
    if noise is None:
        noise = get_noise(num_classes = num_classes,  batch_size =  batch_size)[1]
    with torch.no_grad():
        noisev = noise
        samples = generator(noisev)
        samples = samples.view(batch_size, 3, generator.img_size, generator.img_size)
        normalized_samples = torch.round(samples*127.5 + 127.5).float()
    return normalized_samples