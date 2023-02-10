from model.vanilla_cgan import *
import numpy as np
import torch
import torchvision
from model.conwgan import *
from model.vanilla_cgan_utils import *
from torch.utils.data import DataLoader
from preprocess_dataset import ClassificationDataset
from model.conwgan_utils import *
import wandb
from tqdm import tqdm #te quiero demasio. taqadum
from config.parser_config import config_parser
from torch.autograd import Variable
import os



device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(data_loader,
          img_size,
          class_num,
          get_cat_from_label,
          run_name,
          epochs = 1,
          lr = 1e-3,
          batch_size = 64,
          generator_layer_size = [256, 512, 1024],
          discriminator_layer_size = [1024, 512, 256],
          z_size = 100,
          log_dir = './saved_models/vanilla_gan'):
    
    d_path = os.path.join(log_dir,'discriminator_{}.pt'.format(run_name))
    g_path = os.path.join(log_dir,'generator_{}.pt'.format(run_name))
    
    generator = Generator(generator_layer_size, z_size, img_size, class_num).to(device)
    discriminator = Discriminator(discriminator_layer_size, img_size, class_num).to(device)
    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    
    if os.path.exists(g_path):
        g_checkpoint = torch.load(g_path)
        generator.load_state_dict(g_checkpoint['model_state_dict'])
        g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
        epoch = g_checkpoint['epoch']
        g_loss = g_checkpoint['loss']
    
    if os.path.exists(d_path):
        d_checkpoint = torch.load(d_path)
        discriminator.load_state_dict(d_checkpoint['model_state_dict'])
        d_optimizer.load_state_dict(d_checkpoint['optimizer_state_dict'])
        epoch = d_checkpoint['epoch']
        d_loss = d_checkpoint['loss']
    
    for epoch in range(epochs):
        print('Starting epoch {}...'.format(epoch+1))
        for images, labels, cat in tqdm(data_loader, total=len(data_loader)):        
            # Train data
            real_images = Variable(images).to(device)
            labels = Variable(labels).to(device)        
            # Set generator train
            generator.train()        
            # Train discriminator
            d_loss = discriminator_train_step(device = device,
                                              batch_size = batch_size,
                                              discriminator = discriminator,
                                              generator = generator,
                                              d_optimizer = d_optimizer,
                                              criterion = criterion,
                                              real_images = real_images,                                      
                                              labels = labels,
                                              class_num = class_num,
                                              z_size = z_size)
            
            # Train generator
            g_loss = generator_train_step(device = device,
                                          batch_size = batch_size, 
                                          discriminator = discriminator, 
                                          generator = generator, 
                                          g_optimizer = g_optimizer, 
                                          criterion = criterion,
                                          class_num = class_num,
                                          z_size= z_size)    
        # Set generator eval
        generator.eval()    
        wandb.log({'g_loss': g_loss, 'd_loss': d_loss, 'epoch' : epoch})
        # Building z 
        if epoch % 50 == 0:
            num_of_pics = 64
            z = Variable(torch.randn(num_of_pics, z_size)).to(device)  
            # Labels 0 ~ 8
            labels = Variable(torch.LongTensor(np.arange(num_of_pics))).to(device)
            # Generating images
            sample_images = generator(z, labels).view(-1,3,256,256)
            for i in range(sample_images.shape[0]):
                cat = get_cat_from_label(i)
                img = sample_images[i,...]
                g_single_img = wandb.Image(img, caption= cat)
                wandb.log({'generated images': g_single_img, 'epoch': epoch} )
            # Show images
            grid = torchvision.utils.make_grid(sample_images)
            img_to_log = wandb.Image(grid, caption="conv1")
            wandb.log({'sample images': img_to_log, 'epoch': epoch} )
            #SAVE MODEL
            
            torch.save(
                {'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': g_optimizer.state_dict(),
                'loss': g_loss}, g_path)
            
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': d_optimizer.state_dict(),
                'loss': d_loss}, d_path)
            
            wandb.save(g_path)
            wandb.save(d_path)

    

def run(run_name, args):    
    #LOADING DATA
    full_dataset = ClassificationDataset(
        one_hot = False,
        augmentation= None,
        npz_path= args.npz_path,
        image_path= args.image_path,
        label_path= args.label_path,
        size = args.size,
        normalize = True)
    
    
    # get_cat_from_label = full_dataset._get_cat_from_label
    num_classes = full_dataset._get_num_classes()
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_data, _ = torch.utils.data.random_split(full_dataset, [train_size, valid_size],
                                                                generator=torch.Generator().manual_seed(0))
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)
    
    train_dataloader = DataLoader(train_data,
                                batch_size=args.batch_size, 
                                shuffle=True,
                                num_workers=args.num_workers)
    
    get_cat_from_label = full_dataset._get_cat_from_label
    
    # validation_dataloader = DataLoader(validation_data, 
    #                                 batch_size=args.batch_size, 
    #                                 shuffle=True,
    #                                 num_workers=args.num_workers)
    
    train(data_loader = train_dataloader,
          class_num = num_classes,
          batch_size= args.batch_size,
          epochs = args.epochs,
          lr = args.lr,
          img_size =args.size,
          z_size = args.latent_size,
          get_cat_from_label = get_cat_from_label,
          run_name = run_name)

if __name__ == "__main__":
    # wandb.init(project="training conditional WGAN")
    parser = config_parser()
    args = parser.parse_args()
    run_name = 'TRAIN cGAN'
    # wandb.init(mode="disabled") 
    wandb.init(project="train_vanilla_cgan") 
    wandb.run.name = run_name + ', lr: {}, epochs: {}, size: {}'.format(args.lr, args. epochs, args.size)
    wandb.config = {'epochs' : args.epochs, 
    'run_name' : run_name,
    'npz_path' :args.npz_path,
    'image_path' : args.image_path,
    'label_path' : args.label_path,
    'img_size' : args.size,
    'lr' : args.lr,
    'batch_size': args.batch_size}
    
    run(run_name, args)