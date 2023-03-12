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

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)






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
        normalize=True)

    edge_classes = ["Gryllteiste","Schnatterente","Buchfink","unbestimmte Larusmöwe",
                        "Schmarotzer/Spatel/Falkenraubmöwe","Brandgans","Wasserlinie mit Großalgen",
                        "Feldlerche","Schmarotzerraubmöwe","Grosser Brachvogel","unbestimmte Raubmöwe",
                        "Turmfalke","Trauerseeschwalbe","unbestimmter Schwan",
                        "Sperber","Kiebitzregenpfeifer",
                        "Skua","Graugans","unbestimmte Krähe"]

    edge_labels = [full_dataset._get_label_from_cat(cat) for cat in edge_classes]

    edge_train_data, _ = edge_stratified_split(full_dataset, full_labels = full_dataset._labels, edge_labels = edge_labels,  fraction = 0.8, random_state = 0)                     
    loader = DataLoader(edge_train_data,
                                    batch_size=args.batch_size, 
                                    shuffle=True,
                                    num_workers=args.num_workers)

    # initialize gen and disc/critic
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # initializate optimizer
    opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
    opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

    # for tensorboard plotting
    fixed_noise = torch.randn(args.batch_size, Z_DIM, 1, 1).to(device)
    step = 0

    gen.train()
    critic.train()

    for epoch in range(NUM_EPOCHS):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(device)
            cur_batch_size = data.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(data).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

                # clip critic weights between -0.01, 0.01
                for p in critic.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            wandb.log({"loss_critic": loss_critic, "loss_gen": loss_gen, 'epoch': epoch})
            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0 and batch_idx > 0:
                gen.eval()
                critic.eval()
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        data[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )
                    wandb.log({'Real Image': img_grid_real, 'epoch':step})
                    wandb.log({'Real Image': img_grid_fake, 'epoch':step})

                step += 1
                gen.train()
                critic.train()
                
                
                
if __name__ == "__main__":
    #CONFIG PARSER
    parser = config_parser()
    args  = parser.parse_args()
    train(args)