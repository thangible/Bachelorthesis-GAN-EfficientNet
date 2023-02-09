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



device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(data_loader,
          img_size,
          class_num,
          epochs = 1,
          lr = 1e-3,
          batch_size = 64,
          generator_layer_size = [256, 512, 1024],
          discriminator_layer_size = [1024, 512, 256],
          z_size = 100):
    generator = Generator(generator_layer_size, z_size, img_size, class_num).to(device)
    discriminator = Discriminator(discriminator_layer_size, img_size, class_num).to(device)
    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)


    for epoch in range(epochs):
        print('Starting epoch {}...'.format(epoch+1))
        for images, label, cat in tqdm(data_loader, total=len(data_loader)):        
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
            z = Variable(torch.randn(class_num-1, z_size)).to(device)
            # Labels 0 ~ 8
            labels = Variable(torch.LongTensor(np.arange(batch_size))).to(device)
            # Generating images
            sample_images = generator(z, labels).unsqueeze(1).data.cpu()
            # Show images
            grid = torchvision.utils.make_grid(sample_images, nrow=3, normalize=True).permute(1,2,0).numpy()
            img_to_log = wandb.Image(grid, caption="conv1")
            wandb.log({'sample images': img_to_log, 'epoch': epoch} )

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
    
    # validation_dataloader = DataLoader(validation_data, 
    #                                 batch_size=args.batch_size, 
    #                                 shuffle=True,
    #                                 num_workers=args.num_workers)
    
    train(data_loader = train_dataloader,
          class_num = num_classes,
          batch_size= args.batch_size,
          end_iter = args.epochs,
          lr = args.lr,
          img_size =args.size,
          z_size = args.latent_size)

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