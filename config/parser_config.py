import configargparse
import json

def config_parser():
    parser = configargparse.ArgumentParser()
    # training options
    parser.add_argument('--config', is_config_file=True,help='config file path')
    # parser.add_argument("--input_file", type=str,
    #                     default='./light_compressed.npz', help='input file')   
    parser.add_argument("--epochs", type=int, default=1,help='number of epochs')  
    parser.add_argument("--run_name", type=str, default='No aug',help='name of the run on wandb')  
    parser.add_argument("--image_path", type=str, default=None,help='Image Path')  
    parser.add_argument("--label_path", type=str, default=None,help='label path')  
    parser.add_argument("--npz_path", type=str, default=None,help='npz path')  
    parser.add_argument("--size", type=int, default=256,help='image size') 
    parser.add_argument("--lr", type=float, default=1e-4,help='learning rate') 
    parser.add_argument("--augment", type=str, default=None,help='augment method') 
    parser.add_argument("--batch_size", type=int,default=64,help='batchsize')
    parser.add_argument('--latent_size', type=int, default=100, help='size of gaussian noise vector')
    parser.add_argument('--num_workers', type=int, default=8, help='workers simultaneously putting data into RAM')
    parser.add_argument('--saved_image_directory', type=str, default='data/saved_images', help='directory to where image samples will be saved')
    parser.add_argument('--saved_model_directory', type=str, default='saved_models', help='directory to where model weights will be saved')
    parser.add_argument('--embedding_dim', type=int, default=10, help='size of embedding vector')
    parser.add_argument('--model_dim', type=int, default=64, help='size of dim in generator and discriminator')
    parser.add_argument('--test', type=bool, default=False, help='activate wandb')
    return parser