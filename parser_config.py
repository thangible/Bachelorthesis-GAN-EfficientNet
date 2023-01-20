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
    parser.add_argument("--save_interval", type=int, default=5,
                        help='interval between save')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='batchsize')     
    parser.add_argument("--gen_nodes_num", type=int, default=128,
                        help='nodes number of the first layer in generator')
    parser.add_argument('--unwanted_classes', type=json.loads)
    parser.add_argument('--unwanted_pics', type=json.loads)
    return parser