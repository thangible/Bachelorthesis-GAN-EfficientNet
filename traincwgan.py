import torch
import numpy as np 
import time
import torch.optim as optim
from parser_config import config_parser
from model.cwgan import *
import wandb
from segmentation_dataset import ClassificationDataset

class Trainer():

    def __init__(self, data_loader, class_size, embedding_dim, batch_size, latent_size=100, lr=0.0002, num_workers=1):

        # define device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #get dataset from directory. If not present, download to directory
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.class_size = class_size

        #define models
        self.latent_size = latent_size

        self.dis = Discriminator(class_size, embedding_dim).to(device)
        self.gen = Generator(latent_size,class_size, embedding_dim).to(device)

        self.loss_func = nn.BCELoss().to(device)

        self.optimizer_d = optim.RMSprop(self.dis.parameters(), lr=lr)
        self.optimizer_g = optim.RMSprop(self.gen.parameters(), lr=lr)

    def gradient_penalty(self, real_samples, fake_samples, labels):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.dis(interpolates, labels)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        fake.requires_grad = False
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(gradients[0].size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, epochs):
        saved_model_directory = './save_models/cwgan.'
        start_time = time.time()

        gen_loss_list = []
        dis_loss_list = []
        was_loss_list = []

        lmbda_gp = 10

        for epoch in range(epochs):
            gen_loss = 0
            dis_loss = 0
            cur_time = time.time()
            for images, labels, cat in self.data_loader:
                b_size = len(images)
                #train Discriminator with Wasserstein Loss
                self.optimizer_d.zero_grad()

                #fake loss
                z = torch.randn(b_size, self.latent_size).to(self.device)
                fake_images = self.gen(z, labels.to(self.device))
                fake_pred = self.dis(fake_images, labels.to(self.device))
                d_loss_fake = torch.mean(fake_pred)

                #real loss
                real_pred = self.dis(images.to(self.device), labels.to(self.device))
                d_loss_real = -torch.mean(real_pred)

                gp = self.gradient_penalty(images.to(self.device), fake_images, labels.to(self.device))
                d_loss = d_loss_fake - d_loss_real
                was_loss = (d_loss_fake + d_loss_real) + lmbda_gp*gp
                was_loss.backward()
                self.optimizer_d.step()
                dis_loss += d_loss.item()/b_size

                #train Generator
                self.optimizer_g.zero_grad()
                z = torch.randn(b_size, self.latent_size).to(self.device)
                fake_images = self.gen(z, labels.to(self.device))
                fake_pred = self.dis(fake_images, labels.to(self.device))
                g_loss = -torch.mean(fake_pred)
                g_loss.backward()
                self.optimizer_g.step()
                gen_loss += g_loss.item()/b_size
                
            ## LOGGING
            cur_time = time.time() - cur_time
            print('Epoch {},    Gen Loss: {:.4f},   Dis Loss: {:.4f},   Was Loss: {:.4f}'.format(epoch, gen_loss, dis_loss, was_loss))
            print('Time Taken: {:.4f} seconds. Estimated {:.4f} hours remaining'.format(cur_time, (epochs-epoch)*(cur_time)/3600))
            gen_loss_list.append(gen_loss)
            dis_loss_list.append(dis_loss)
            was_loss_list.append(was_loss)
            wandb.log({'gen_loss': gen_loss, 'dis_loss': dis_loss, 'was_loss': was_los})

            #show samples
            labels = torch.LongTensor(np.arange(10)).to(self.device)
            z = torch.randn(10, self.latent_size).to(self.device)
            sample_images = self.gen(z, labels)

            #save models to model_directory
            torch.save(self.gen.state_dict(), saved_model_directory + '/generator_{}.pt'.format(epoch))
            torch.save(self.dis.state_dict(), saved_model_directory + '/discriminator_{}.pt'.format(epoch))

            ### VISUALIZE
            if epoch % 50 == 99:
                image_grid = wandb.Image(sample_images, caption ='/epoch_{}_checkpoint.jpg'.format(epoch))
                wandb.log({'generated_pic': image_grid})

        finish_time = time.time() - start_time
        print('Training Finished. Took {:.4f} seconds or {:.4f} hours to complete.'.format(finish_time, finish_time/3600))
        return gen_loss_list, dis_loss_list

def main(args):
    ##SET UP DATASET
    full_dataset = ClassificationDataset(
        one_hot = False,
        augmentation= None,
        npz_path= args.npz_path,
        image_path= args.image_path,
        label_path= args.label_path,
        size = args.size)

    get_cat_from_label = full_dataset._get_cat_from_label
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_data, _ = torch.utils.data.random_split(full_dataset, [train_size, valid_size],
                                                                  generator=torch.Generator().manual_seed(0))    
    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    class_size = full_dataset._get_num_classes()

    gan = Trainer(data_loader = train_dataloader, 
                  class_size = class_size,
                  embedding_dim = args.embedding_dim,
                  latent_size= args.latent_size,
                  lr = args.lr,
                  num_workers= args.num_workers,
                  batch_size= args.batch_size)
    
    gen_loss_lost, dis_loss_list = gan.train(args.epochs)

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    run_name = 'TRAIN cWGAN'
    wandb.init(project="training conditional WGAN")
    wandb.run.name = run_name + ', lr: {}, epochs: {}, size: {}'.format(args.lr, args. epochs, args.size)
    wandb.config = {'epochs' : args.epochs, 
        'run_name' : run_name,
        'npz_path' :args.npz_path,
        'image_path' : args.image_path,
        'label_path' : args.label_path,
        'img_size' : args.size,
        'lr' : args.lr,
        'batch_size': args.batch_size}
    main(args)
