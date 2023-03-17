"""
Training of DCGAN network with WGAN loss
Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.wasserstein_cgan import Discriminator, Generator, initialize_weights
from dataset_utils import edge_stratified_split
from classification_dataset import ClassificationDataset
import albumentations as A 
import wandb
from  config.parser_config import config_parser

# Hyperparameters etc



def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.size
    CHANNELS_IMG = 3
    Z_DIM = args.latent_size
    NUM_EPOCHS = args.epochs
    FEATURES_CRITIC = args.model_dim
    FEATURES_GEN = args.model_dim
    CRITIC_ITERATIONS = 5
    WEIGHT_CLIP = 0.01

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
        normalize= True)

    edge_classes = ["Gryllteiste","Schnatterente","Buchfink","unbestimmte Larusmöwe",
                        "Schmarotzer/Spatel/Falkenraubmöwe","Brandgans","Wasserlinie mit Großalgen",
                        "Feldlerche","Schmarotzerraubmöwe","Grosser Brachvogel","unbestimmte Raubmöwe",
                        "Turmfalke","Trauerseeschwalbe","unbestimmter Schwan",
                        "Sperber","Kiebitzregenpfeifer",
                        "Skua","Graugans","unbestimmte Krähe"]

    edge_labels = [full_dataset._get_label_from_cat(cat) for cat in edge_classes]

    edge_train_data, _ = edge_stratified_split(full_dataset, full_labels = full_dataset._labels, edge_labels = edge_labels,  fraction = 0.8, random_state = 0)                     
    loader = DataLoader(edge_train_data,
                                    batch_size=BATCH_SIZE, 
                                    shuffle=True,
                                    num_workers=args.num_workers)

    CLASS_NUM = full_dataset._get_num_classes()
    # initialize gen and disc/critic
    GEN = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, CLASS_NUM).to(device)
    CRITIC = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, CLASS_NUM).to(device)
    initialize_weights(GEN)
    initialize_weights(CRITIC)

    # initializate optimizer
    opt_gen = optim.RMSprop(GEN.parameters(), lr=LEARNING_RATE)
    opt_critic = optim.RMSprop(CRITIC.parameters(), lr=LEARNING_RATE)

    # for tensorboard plotting
    fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
    step = 0

    GEN.train()
    CRITIC.train()

    for epoch in range(NUM_EPOCHS):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (data, labels, cat) in enumerate(tqdm(loader)):
            data = data.to(device)
            labels = labels.to(device)
            cur_batch_size = data.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = GEN(noise, labels)
                critic_real = CRITIC(data, labels).reshape(-1)
                critic_fake = CRITIC(fake, labels).reshape(-1)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                CRITIC.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

                # clip critic weights between -0.01, 0.01
                for p in CRITIC.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = CRITIC(fake, labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            GEN.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            
            # Print losses occasionally and print to tensorboard
            if batch_idx % 50 == 0:
                GEN.eval()
                CRITIC.eval()
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = GEN(noise, labels)
                    # take out (up to) 32 examples
                    # data = torch.round(img_grid_real*127.5 + 127.5).float()
                    # fake = torch.round(fake*127.5 + 127.5).float()
                    img_grid_real = torchvision.utils.make_grid(
                        data[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )
                    img_to_log_real = wandb.Image(img_grid_real, caption="samples")
                    img_to_log_fake = wandb.Image(img_grid_fake, caption="samples")
                    
                    wandb.log({'Real Image': img_to_log_real, 'epoch':step})
                    wandb.log({'Fake Image': img_to_log_fake, 'epoch':step})

                step += 1
                GEN.train()
                CRITIC.train()
        
        wandb.log({"loss_critic": loss_critic, "loss_gen": loss_gen, 'epoch': epoch})        
        
                
if __name__ == "__main__":
    #CONFIG PARSER
    parser = config_parser()
    args  = parser.parse_args()
    run_name = 'TRAIN cGAN'+ ', dim:{}, lr: {}, epochs: {}, size: {}'.format(args.model_dim, args.lr, args. epochs, args.size)
    # wandb.init(mode="disabled") 
    wandb_mode = None
    if args.test:
        wandb_mode = 'disabled'
    wandb.init(project="NEW WASSERSTEIN GAN", mode = wandb_mode) 
    wandb.run.name = run_name
    wandb.config = {'epochs' : args.epochs, 
    'run_name' : run_name,
    'npz_path' :args.npz_path,
    'image_path' : args.image_path,
    'label_path' : args.label_path,
    'img_size' : args.size,
    'lr' : args.lr,
    'batch_size': args.batch_size,
    'model_dim': args.model_dim,
    'latent_size' : args.latent_size}
    train(args)