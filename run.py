import classifier
import wandb
from  parser_config import config_parser
import albumentations as A




if __name__ == "__main__":
    #CONFIG PARSER
    parser = config_parser()
    args  = parser.parse_args()
    
    wandb.init(project="classifier-efficientnet")  
    
    augmentations = [A.augmentations.geometric.resize.RandomScale (scale_limit=0.1, interpolation=1, always_apply=False, p=0.3),
                    A.augmentations.geometric.transforms.Affine (scale=None, 
                                                        translate_percent=0.3, 
                                                        translate_px=None,
                                                        rotate=None,
                                                        shear=0.0, 
                                                        interpolation=1, 
                                                        mask_interpolation=0,
                                                        cval=0, 
                                                        cval_mask=0, 
                                                        mode=0, 
                                                        fit_output=False, 
                                                        keep_ratio=True,
                                                        always_apply=True, p=0.3),
                    A.augmentations.geometric.transforms.Affine (scale=None, 
                                                        translate_percent=0.0, 
                                                        translate_px=None,
                                                        rotate=None,
                                                        shear=0.3, 
                                                        interpolation=1, 
                                                        mask_interpolation=0,
                                                        cval=0, 
                                                        cval_mask=0, 
                                                        mode=0, 
                                                        fit_output=False, 
                                                        keep_ratio=True,
                                                        always_apply=True, p=0.3) 
                     ]
    run_names = ['RandomScale 0,3',
                 'Translate 0.3'
                 'shear 0.3',
                 ]
    
    
    for i in len(augmentations):
        classifier.single_run(args, given_augment = augmentations[i], run_name = run_names[i])