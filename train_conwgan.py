
import numpy as np
import torch
import torchvision
from model.conwgan import *
from torch.utils.data import DataLoader
from classification_dataset import ClassificationDataset
from model.conwgan_utils import *
import wandb
from tqdm import tqdm #te quiero demasio. taqadum
from config.parser_config import config_parser
from dataset_utils import stratified_split, edge_stratified_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESTORE_MODE = False


def train(train_dataloader,
          validation_dataloader,
          num_classes,
          output_path = './saved_models/con_wgan',
          ACGAN_SCALE_G = 1.,
          ACGAN_SCALE = 1., 
          CRITIC_ITERS = 5,
          batch_size = 64,
          start_iter = 0, # starting iteration 
          GENER_ITERS = 1,
          end_iter = 1,
          lr = 1e-4,
          image_size = 256,
          latent_size = 100,
          model_dim = 8,
          edge_labels = None
    ) -> None:
    
    if RESTORE_MODE:
        GENERATOR = torch.load(output_path + "generator_with_num_class_{}.pt".format(num_classes))
        DISCRIMINATOR = torch.load(output_path + "discriminator_with_num_class_{}.pt".format(num_classes))
    else:
        GENERATOR = GoodGenerator(num_classes = num_classes, size=image_size, latent_size = latent_size, dim = model_dim)
        DISCRIMINATOR = GoodDiscriminator(num_classes = num_classes, size=image_size, dim = model_dim)
        GENERATOR.apply(weights_init)
        DISCRIMINATOR.apply(weights_init)
        
    optimizer_g = torch.optim.Adam(GENERATOR.parameters(), lr=lr, betas=(0,0.9))
    optimizer_d = torch.optim.Adam(DISCRIMINATOR.parameters(), lr=lr, betas=(0,0.9))
    aux_criterion = nn.CrossEntropyLoss() # nn.NLLLoss()

    one = torch.FloatTensor([1])
    mone = one * -1
    GENERATOR.to(device)
    DISCRIMINATOR.to(device)
    one = one.float().to(device)
    mone = mone.float().to(device)
    
    
    for iteration in tqdm(range(start_iter, end_iter)):
        #---------------------TRAIN G------------------------
        for p in DISCRIMINATOR.parameters():
            p.requires_grad_(False)  # freeze D
        gen_cost = None
        for i in range(GENER_ITERS):
            GENERATOR.zero_grad()
            f_labels, noise = get_noise(device = device, num_classes = num_classes, batch_size=batch_size, latent_size = latent_size)
            noise.requires_grad_(True)
            fake_data = GENERATOR(noise)
            gen_cost, gen_aux_output = DISCRIMINATOR(fake_data)
            gen_aux_output.to(device)
            aux_errG = aux_criterion(gen_aux_output, f_labels).mean()
            gen_cost = -gen_cost.mean()
            g_cost = ACGAN_SCALE_G*aux_errG + gen_cost
            g_cost.backward()
        optimizer_g.step()
        #---------------------TRAIN D------------------------
        for p in DISCRIMINATOR.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CRITIC_ITERS):
            DISCRIMINATOR.zero_grad()
            # gen fake data and load real data
            _, noise = get_noise(device = device, num_classes = num_classes, batch_size=batch_size)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            fake_data = GENERATOR(noisev).detach()
            dataiter = iter(train_dataloader)
            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(train_dataloader)
                batch = dataiter.next()
            real_data = batch[0].float() #batch[1] contains labels
            real_data.requires_grad_(True)
            real_label = batch[1]
            #print("r_label" + str(r_label))
            real_data = real_data.to(device)
            real_label = real_label.to(device)

            # train with real data
            disc_real, aux_output = DISCRIMINATOR(real_data)
            aux_errD_real = aux_criterion(aux_output, real_label)
            errD_real = aux_errD_real.mean()
            disc_real = disc_real.mean()


            # train with fake data
            disc_fake, aux_output = DISCRIMINATOR(fake_data)
            #aux_errD_fake = aux_criterion(aux_output, fake_label)
            #errD_fake = aux_errD_fake.mean()
            disc_fake = disc_fake.mean()

            #showMemoryUsage(0)
            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(device = device, 
                                                     discriminator = DISCRIMINATOR,
                                                     real_data = real_data, 
                                                     fake_data = fake_data,
                                                     batch_size = batch_size)
            #showMemoryUsage(0)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_acgan = errD_real #+ errD_fake
            (disc_cost + ACGAN_SCALE*disc_acgan).backward()
            w_dist = disc_fake  - disc_real
            optimizer_d.step()
            #------------------VISUALIZATION----------
            if i == CRITIC_ITERS-1:
                if iteration %200==199:
                    body_model = [i for i in DISCRIMINATOR.children()][0]
                    layer1 = body_model.conv
                    xyz = layer1.weight.data.clone()
                    tensor = xyz.cpu()
                    conv_img_to_log = torchvision.utils.make_grid(tensor, nrow=8,padding=1)
                    img_to_log = wandb.Image(conv_img_to_log, caption="conv1")
                    wandb.log({'D/conv1': img_to_log, 'epoch': iteration} )
                    
        wandb.log({'gen_cost': gen_cost, 'epoch': iteration})
    #----------------------Generate images-----------------
        wandb.log({'train_disc_cost': disc_cost.cpu().data.numpy(), 'epoch': iteration})
        wandb.log({'train_gen_cost': gen_cost.cpu().data.numpy(), 'epoch': iteration})
        wandb.log({'wasserstein_distance': w_dist.cpu().data.numpy(), 'epoch': iteration})
        if iteration % 50==0:
            dev_disc_costs = []
            for images, _, _ in validation_dataloader:
                imgs = torch.Tensor(images[0])
                imgs = imgs.to(device)
                with torch.no_grad():
                    imgs_v = imgs       
                output_wgan, _ = DISCRIMINATOR(imgs_v)
                fixed_labels =  torch.LongTensor(edge_labels[:batch_size])
                _, fixed_noise = get_noise(device = device,num_classes = num_classes, 
                                                        labels = fixed_labels, 
                                                        batch_size=batch_size)
                _dev_disc_cost = -output_wgan.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
                            
            gen_images = generate_image(GENERATOR, noise = fixed_noise,
                                        num_classes=num_classes, 
                                        batch_size = batch_size)
            log_imgs = torchvision.utils.make_grid(gen_images)
            grid_images = wandb.Image(log_imgs, caption="Fake images")
            wandb.log({"fake_image": grid_images} )
    #----------------------Save model----------------------
            torch.save(GENERATOR, output_path + "generator.pt")
            torch.save(DISCRIMINATOR, output_path + "discriminator.pt")
        
def run(run_name, args):    
    #LOADING DATA
    full_dataset = ClassificationDataset(
        one_hot = False,
        augmentation= None,
        npz_path= args.npz_path,
        image_path= args.image_path,
        label_path= args.label_path,
        size = args.size,
        normalize=True)

    get_cat_from_label = full_dataset._get_cat_from_label
    class_size = full_dataset._get_num_classes()   
    
    edge_classes = ["Gryllteiste","Schnatterente","Buchfink","unbestimmte Larusmöwe",
                        "Schmarotzer/Spatel/Falkenraubmöwe","Brandgans","Wasserlinie mit Großalgen",
                        "Feldlerche","Schmarotzerraubmöwe","Grosser Brachvogel","unbestimmte Raubmöwe",
                        "Turmfalke","Trauerseeschwalbe","unbestimmter Schwan",
                        "Sperber","Kiebitzregenpfeifer",
                        "Skua","Graugans","unbestimmte Krähe"]
    
    edge_labels = [full_dataset._get_label_from_cat(cat) for cat in edge_classes]
    
    edge_train_data, test_dataset = edge_stratified_split(full_dataset, full_labels = full_dataset._labels, edge_labels = edge_labels,  fraction = 0.8, random_state = 0)                     
    train_dataloader = DataLoader(edge_train_data,
                                  batch_size=args.batch_size, 
                                  shuffle=True,
                                  num_workers=args.num_workers)
    
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=args.batch_size, 
                                  shuffle=True,
                                  num_workers=args.num_workers)
    
    
    train(train_dataloader = train_dataloader,
          validation_dataloader = test_dataloader, 
          num_classes = full_dataset._get_num_classes(),
          batch_size= args.batch_size,
          end_iter = args.epochs,
          lr = args.lr,
          image_size=args.size,
          latent_size= args.latent_size,
          model_dim= args.model_dim,
          edge_labels = edge_labels)


if __name__ == "__main__":
    # wandb.init(project="training conditional WGAN")
    parser = config_parser()
    args = parser.parse_args()
    run_name = 'TRAIN cWGAN'
    # wandb.init(mode="disabled") 
    wandb_mode = None
    if args.test:
        wandb_mode = 'disabled'
    wandb.init(project="train_wgan", mode = wandb_mode) 
    wandb.run.name = run_name + ' ,lr: {}, epochs: {}, size: {}'.format(args.lr, args. epochs, args.size)
    wandb.config = {'epochs' : args.epochs, 
    'run_name' : run_name,
    'npz_path' :args.npz_path,
    'image_path' : args.image_path,
    'label_path' : args.label_path,
    'img_size' : args.size,
    'lr' : args.lr,
    'batch_size': args.batch_size,
    'model_dim': args.model_dim}
    
    run(run_name, args)
    
    