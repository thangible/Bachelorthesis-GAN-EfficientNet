
import numpy as np
import torch
import torchvision
from model.conwgan import *
from torch.utils.data import DataLoader
from segmentation_dataset import ClassificationDataset
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESTORE_MODE = False

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

def calc_gradient_penalty(netD, real_data, fake_data, batch_size = 64, LAMBDA = 10):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, netD.dim, netD.dim)
    alpha = alpha.to(device)

    fake_data = fake_data.view(batch_size, 3, netD.dim, netD.dim)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)   

    disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def gen_rand_noise_with_label(num_classes, label=None, batchsize = 64):
    if label is None:
        label = np.random.randint(0, num_classes, batchsize)
    #attach label into noise
    noise = np.random.normal(0, 1, (batchsize, 128))
    prefix = np.zeros((batchsize, num_classes))
    prefix[np.arange(batchsize), label] = 1
    noise[np.arange(batchsize), :num_classes] = prefix[np.arange(batchsize)]

    noise = torch.from_numpy(noise).float()
    noise = noise.to(device)

    return noise

def generate_image(netG, num_classes, noise=None, batch_size = 64):
    if noise is None:
        rand_label = np.random.randint(0, num_classes, batch_size)
        noise = gen_rand_noise_with_label(rand_label, batch_size, num_classes)
    with torch.no_grad():
        noisev = noise
    samples = netG(noisev)
    samples = samples.view(batch_size, 3, netG.dim, netG.dim)
    samples = samples * 0.5 + 0.5

    return samples



def train(train_dataloader,
          validation_dataloader,
          num_classes,
          output_path = './logs/',
          ACGAN_SCALE_G = 1.,
          ACGAN_SCALE = 1., 
          CRITIC_ITERS = 5,
          batch_size = 64,
          start_iter = 0, # starting iteration 
          GENER_ITERS = 1,
          end_iter = 10000,
          lr = 1e-4# How many iterations to train for
    ) -> None:
    
    if RESTORE_MODE:
        aG = torch.load(output_path + "generator.pt")
        aD = torch.load(output_path + "discriminator.pt")
    else:
        aG = GoodGenerator(64,64*64*3)
        aD = GoodDiscriminator(64, num_classes)
        aG.apply(weights_init)
        aD.apply(weights_init)
        
    optimizer_g = torch.optim.Adam(aG.parameters(), lr=lr, betas=(0,0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=lr, betas=(0,0.9))
    aux_criterion = nn.CrossEntropyLoss() # nn.NLLLoss()

    one = torch.FloatTensor([1])
    mone = one * -1
    aG = aG.to(device)
    aD = aD.to(device)
    one = one.to(device)
    mone = mone.to(device)
    
    
    #writer = SummaryWriter()
    dataiter = iter(train_dataloader)
    for iteration in range(start_iter, end_iter):
        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D
        gen_cost = None
        for i in range(GENER_ITERS):
            print("Generator iters: " + str(i))
            aG.zero_grad()
            f_label = np.random.randint(0, num_classes, batch_size)
            noise = gen_rand_noise_with_label(f_label)
            noise.requires_grad_(True)
            fake_data = aG(noise)
            gen_cost, gen_aux_output = aD(fake_data)
            aux_label = torch.from_numpy(f_label).long()
            aux_label = aux_label.to(device)
            aux_errG = aux_criterion(gen_aux_output, aux_label).mean()
            gen_cost = -gen_cost.mean()
            g_cost = ACGAN_SCALE_G*aux_errG + gen_cost
            g_cost.backward()
        optimizer_g.step()
        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CRITIC_ITERS):
            print("Critic iter: " + str(i))
            aD.zero_grad()

            # gen fake data and load real data
            f_label = np.random.randint(0, num_classes, batch_size)
            noise = gen_rand_noise_with_label(f_label)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            fake_data = aG(noisev).detach()
            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(train_dataloader)
                batch = dataiter.next()
            real_data = batch[0] #batch[1] contains labels
            real_data.requires_grad_(True)
            real_label = batch[1]
            #print("r_label" + str(r_label))
            real_data = real_data.to(device)
            real_label = real_label.to(device)

            # train with real data
            disc_real, aux_output = aD(real_data)
            aux_errD_real = aux_criterion(aux_output, real_label)
            errD_real = aux_errD_real.mean()
            disc_real = disc_real.mean()


            # train with fake data
            disc_fake, aux_output = aD(fake_data)
            #aux_errD_fake = aux_criterion(aux_output, fake_label)
            #errD_fake = aux_errD_fake.mean()
            disc_fake = disc_fake.mean()

            #showMemoryUsage(0)
            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)
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
                    body_model = [i for i in aD.children()][0]
                    layer1 = body_model.conv
                    xyz = layer1.weight.data.clone()
                    tensor = xyz.cpu()
                    img_to_log = torchvision.utils.make_grid(tensor, nrow=8,padding=1)
                    wandb.log({'D/conv1': img_to_log, 'epoch': iteration} )
                    
        wandb.log({'gen_cost': gen_cost, 'epoch': iteration})
    #----------------------Generate images-----------------
        wandb.log({'train_disc_cost': disc_cost.cpu().data.numpy()})
        wandb.log({'train_gen_cost': gen_cost.cpu().data.numpy()})
        wandb.log({'wasserstein_distance': w_dist.cpu().data.numpy()})
        if iteration % 200==0:
            dev_disc_costs = []
            for _, images in validation_dataloader:
                imgs = torch.Tensor(images[0])
                imgs = imgs.to(device)
                with torch.no_grad():
                    imgs_v = imgs                
                D,_ = aD(imgs_v)
                fixed_label = []
                for c in range(batch_size):
                    fixed_label.append(c%num_classes)
                fixed_noise = gen_rand_noise_with_label(fixed_label)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)            
            gen_images = generate_image(aG, fixed_noise = fixed_noise,
                                        num_classes=num_classes, 
                                        batch_size = batch_size)
            torchvision.utils.save_image(gen_images, output_path + 'samples_{}.png'.format(iteration), nrow=8, padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            wandb.log({'fake image': grid_images, 'epoch': iteration} )
    #----------------------Save model----------------------
            torch.save(aG, output_path + "generator.pt")
            torch.save(aD, output_path + "discriminator.pt")
        
def run(run_name, args):
    wandb.init(project="training conditional WGAN")
    wandb.run.name = run_name + ' ,lr: {}, epochs: {}, size: {}'.format(args.lr, args. epochs, args.size)
    wandb.config = {'epochs' : args.epochs, 
    'run_name' : run_name,
    'npz_path' :args.npz_path,
    'image_path' : args.image_path,
    'label_path' : args.label_path,
    'img_size' : args.size,
    'lr' : args.lr,
    'batchsize': args.batch_size}
    
    #LOADING DATA
    full_dataset = ClassificationDataset(
        one_hot = False,
        augmentation= 'no augment',
        npz_path= args.npz_path,
        image_path= args.image_path,
        label_path= args.label_path,
        size = args.size)
    
    
    # get_cat_from_label = full_dataset._get_cat_from_label
    num_classes = full_dataset._get_num_classes()
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_data, validation_data = torch.utils.data.random_split(full_dataset, [train_size, valid_size],
                                                                generator=torch.Generator().manual_seed(0))
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)
    
    train_dataloader = DataLoader(train_data,
                                batch_size=args.batch_size, 
                                shuffle=True,
                                num_workers=8)
    
    validation_dataloader = DataLoader(validation_data, 
                                    batch_size=args.batch_size, 
                                    shuffle=True,
                                    num_workers=8)
    
    train(train_dataloader,
          validation_dataloader, 
          num_classes = num_classes,
          batch_size= args.batch_size,
          ACGAN_SCALE_G = 1.,
          ACGAN_SCALE = 1.,
          CRITIC_ITERS = 5,
          end_iter = args.epochs,
          lr = args.lr )