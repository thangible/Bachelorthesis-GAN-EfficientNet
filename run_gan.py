import model.edge_classifier as classifier
from  config.parser_config import config_parser
import albumentations as A
import numpy as np




if __name__ == "__main__":
    #CONFIG PARSER
    parser = config_parser()
    args  = parser.parse_args()
    

    classifier.single_run(args, given_augment = 'no augment', run_name = 'GAN', project_name = 'OFFICIAL GOOD CLASSIFIER')

    # for run_name in HP:
    #     augmentation = HP[run_name]
    #     classifier.single_run(args, given_augment = augmentation, run_name = run_name)
        