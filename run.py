import classifier
import wandb
from  parser_config import config_parser
import albumentations as A




if __name__ == "__main__":
    #CONFIG PARSER
    parser = config_parser()
    args  = parser.parse_args()
    
    center_crop = A.Compose([A.CenterCrop(height = 200, width = 200, p=1.0), 
                         A.Resize(height = 500, width = 500, interpolation=1, p=1.0)
                         ])
    augmentations = [center_crop
                     ]
    run_names = ['center_crop']
    
    
    for i in range(len(augmentations)):
        classifier.single_run(args, given_augment = augmentations[i], run_name = run_names[i])
        