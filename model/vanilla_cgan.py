import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, generator_layer_size, z_size, img_size, class_num):
        super().__init__()
        
        self.z_size = z_size
        self.img_size = img_size
        
        self.label_emb = nn.Embedding(class_num, class_num)
        
        self.model = nn.Sequential(
            nn.Linear(self.z_size + class_num, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], 3*self.img_size * self.img_size),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        # Reshape z
        z = z.view(-1, self.z_size)
        # One-hot vector to embedding vector
        c = self.label_emb(labels)
        # Concat image & label
        x = torch.cat([z, c], 1)
        # Generator out
        out = self.model(x)
        return out.view(-1, 3, self.img_size, self.img_size)



class Discriminator(nn.Module):
    def __init__(self, discriminator_layer_size, img_size, class_num):
        super().__init__()
        self.label_emb = nn.Embedding(class_num, class_num)
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Linear(3*self.img_size * self.img_size + class_num, discriminator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[0], discriminator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[1], discriminator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[2], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        # Reshape fake image
        x = x.view(-1, 3*self.img_size * self.img_size)        
        # One-hot vector to embedding vector
        c = self.label_emb(labels)        
        # Concat image & label
        x = torch.cat([x, c], 1)        
        # Discriminator out
        out = self.model(x)        
        return out.squeeze()


