import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm #te quiero demasio. taqadum
from torch.autograd import Variable
import os
import wandb
#FROM MODULES
from config.parser_config import config_parser
from dataset_utils import *
from model.vanilla_cgan_utils import *
from classification_dataset import ClassificationDataset
from model.vanilla_cgan import *
import albumentations as A



device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(data_loader,
          img_size,
          class_num,
          get_cat_from_label,
          run_name,
          model_dim = None,
          epochs = 1,
          lr = 1e-3,
          batch_size = 64,
          generator_layer_size = [256, 512, 1024],
          discriminator_layer_size = [1024, 512, 256],
          z_size = 100,
          log_dir = './saved_models/vanilla_gan',
          test_mode = False,
          edge_labels = None):
    
    if model_dim:
        generator_layer_size = [model_dim, model_dim*2, model_dim*4]
        discriminator_layer_size = [model_dim*4, model_dim*2, model_dim]
    d_path = os.path.join(log_dir,'discriminator_{}.pt'.format("_".join(run_name.split(" "))))
    g_path = os.path.join(log_dir,'generator_{}.pt'.format("_".join(run_name.split(" "))))
    
    generator = Generator(generator_layer_size, z_size, img_size, class_num).to(device)
    discriminator = Discriminator(discriminator_layer_size, img_size, class_num).to(device)
    criterion = nn.BCELoss().to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    
    
    if not(test_mode):
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
                                          edge_labels=edge_labels,
                                          z_size= z_size)  
            # print('g_loss', g_loss.cpu(), 'd_loss', d_loss.cpu())
              
        # Set generator eval
        generator.eval()    
        wandb.log({'g_loss': g_loss, 'epoch' : epoch})
        wandb.log({'d_loss': d_loss, 'epoch' : epoch})
        # Building z 
        if epoch % 200 == 199:
            z = Variable(torch.randn(len(edge_labels), z_size)).to(device)  
            # Labels 0 ~ 8
            labels = Variable(torch.LongTensor(edge_labels)).to(device)
            # Generating images
            sample_images_raw = generator(z, labels).view(-1,3,256,256)
            sample_images = torch.round(sample_images_raw*127.5 + 127.5).float()
            for i in range(sample_images.shape[0]):
                cat = get_cat_from_label(labels[i])
                img = sample_images[i,...]
                g_single_img = wandb.Image(img, caption= cat)
                wandb.log({'generated images': g_single_img, 'epoch': epoch} )
            # Show images
            grid = torchvision.utils.make_grid(sample_images)
            img_to_log = wandb.Image(grid, caption="samples")
            wandb.log({'sample images': img_to_log, 'epoch': epoch} )
            #SAVE MODEL
        if not(test_mode) and epoch % 200 == 199:
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
    for i in range(10):
        z = Variable(torch.randn(len(edge_labels), z_size)).to(device)  
        # Labels 0 ~ 8
        labels = Variable(torch.LongTensor(edge_labels)).to(device)
        # Generating images
        sample_images_raw = generator(z, labels).view(-1,3,256,256)
        sample_images = torch.round(sample_images_raw*127.5 + 127.5).float()
        for i in range(sample_images.shape[0]):
            cat = get_cat_from_label(labels[i])
            img = sample_images[i,...]
            g_single_img = wandb.Image(img, caption= cat)
            wandb.log({'end results images': g_single_img, 'label' : cat})

def run(run_name, args):    
    
    Rotation = A.Rotate(p=0.9)
    Flip = A.Flip(p=0.5)
    Augmentation = A.Compose([Rotation, Flip])
    
    #LOADING DATA
    full_dataset = ClassificationDataset(
        one_hot = False,
        augmentation= Augmentation,
        npz_path= args.npz_path,
        image_path= args.image_path,
        label_path= args.label_path,
        size = args.size,
        normalize=True)
   
    edge_classes = ["Gryllteiste","Schnatterente","Buchfink","unbestimmte Larusmöwe",
                        "Schmarotzer/Spatel/Falkenraubmöwe","Brandgans","Wasserlinie mit Großalgen",
                        "Feldlerche","Schmarotzerraubmöwe","Grosser Brachvogel","unbestimmte Raubmöwe",
                        "Turmfalke","Trauerseeschwalbe","unbestimmter Schwan",
                        "Sperber","Kiebitzregenpfeifer",
                        "Skua","Graugans","unbestimmte Krähe"]
    
    edge_labels = [full_dataset._get_label_from_cat(cat) for cat in edge_classes]
    
    edge_train_data, _ = edge_stratified_split(full_dataset, full_labels = full_dataset._labels, edge_labels = edge_labels,  fraction = 0.8, random_state = 0)                     
    train_dataloader = DataLoader(edge_train_data,
                                  batch_size=args.batch_size, 
                                  shuffle=True,
                                  num_workers=args.num_workers)

    
    get_cat_from_label = full_dataset._get_cat_from_label
    
    # validation_dataloader = DataLoader(validation_data, 
    #                                 batch_size=args.batch_size, 
    #                                 shuffle=True,
    #                                 num_workers=args.num_workers)
    
    train(data_loader = train_dataloader,
          class_num = full_dataset._get_num_classes(),
          batch_size= args.batch_size,
          epochs = args.epochs,
          lr = args.lr,
          img_size =args.size,
          z_size = args.latent_size,
          get_cat_from_label = get_cat_from_label,
          run_name = run_name,
          model_dim = args.model_dim,
          test_mode= args.test,
          edge_labels = edge_labels)

if __name__ == "__main__":
    # wandb.init(project="training conditional WGAN")
    parser = config_parser()
    args = parser.parse_args()
    run_name = 'TRAIN cGAN'+ ', dim:{}, lr: {}, epochs: {}, size: {}'.format(args.model_dim, args.lr, args. epochs, args.size)
    # wandb.init(mode="disabled") 
    wandb_mode = None
    if args.test:
        wandb_mode = 'disabled'
    wandb.init(project="train_vanilla_cgan", mode = wandb_mode) 
    wandb.run.name = run_name
    wandb.config = {'epochs' : args.epochs, 
    'run_name' : run_name,
    'npz_path' :args.npz_path,
    'image_path' : args.image_path,
    'label_path' : args.label_path,
    'img_size' : args.size,
    'lr' : args.lr,
    'batch_size': args.batch_size}
    
    run(run_name, args)