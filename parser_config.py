import configargparse
import json

def config_parser():
    parser = configargparse.ArgumentParser()
    # training options
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    # parser.add_argument("--input_file", type=str,
    #                     default='./light_compressed.npz', help='input file')   
    parser.add_argument("--epochs", type=int, default=100,
                        help='number of epochs')  
    parser.add_argument("--run_name", type=str, default='No aug',
                        help='name of the run on wandb')  
    parser.add_argument("--image_path", type=str, default=None,
                        help='Image Path')  
    parser.add_argument("--label_path", type=str, default=None,
                        help='label path')  
    parser.add_argument("--npz_path", type=str, default=None,
                        help='npz path')  
    parser.add_argument("--size", type=int, default=256,
                        help='image size') 
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='learning rate') 
    parser.add_argument("--augment", type=str, default=None,
                        help='augment method') 
    parser.add_argument("--batch_size", type=int, default=64,
                        help='batchsize') 
    # parser.add_argument('--unwanted_classes', type=json.loads)
    # parser.add_argument('--unwanted_pics', type=json.loads)
    return parser