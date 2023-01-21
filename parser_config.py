import configargparse
import json

def config_parser():
    parser = configargparse.ArgumentParser()
    # training options
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--input_file", type=str,
                        default='./light_compressed.npz', help='input file')   
    parser.add_argument("--epochs", type=int, default=100,
                        help='number of epochs')  
    parser.add_argument('--unwanted_classes', type=json.loads)
    parser.add_argument('--unwanted_pics', type=json.loads)
    return parser