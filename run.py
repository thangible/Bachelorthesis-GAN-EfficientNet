import model.classifier as classifier
from  config.parser_config import config_parser
import albumentations as A
import numpy as np




if __name__ == "__main__":
    #CONFIG PARSER
    parser = config_parser()
    args  = parser.parse_args()
    

    
    ##GEOMETRIC AUGMENTATION
    Resize = A.Resize(height = 500, width = 500, interpolation=1, p=1.0)
    CenterCrop = A.Compose([A.CenterCrop(height = 200, width = 200, p=1.0),Resize])
    CenterCrop_2 = A.Compose([A.CenterCrop(height = 100, width = 100, p=1.0),Resize])
    Rotation = A.Rotate(p=1.0)
    Flip = A.Flip(p=1.0)
    ##MOTION BLUR
    #COLOR AUGMENTATION
    CLAHE = A.CLAHE(clip_limit = 16, tile_grid_size=(4, 4), p=1.0)
    Sharpen = A.Sharpen(alpha= (0.6,0.8), lightness = (0.6,1.0), p=1.0)
    ChannelShuffle = A.ChannelShuffle(p =1)
    ColorJitter =  A.ColorJitter(brightness=0.8, hue = 0.42, contrast = 0.54, saturation = 0.65, p=1.0)
    ToGray = A.ToGray(p=1)
    ToSepia = A.ToSepia(p=1)
    GaussNoise = A.GaussNoise(p =1)
    Normalize = A.Normalize(mean = (0.184, 1.289, 0.661), std = (0.708, 0.338, 0.177), p = 1.0)
    ### SOLARIZE
    
    #CUTOUT
    GridDropout = A.GridDropout(ratio = 0.6, random_offset = True, p=1)
    Cutout = A.CoarseDropout(max_holes=1, p =1, max_height=50, max_width=50)
    
    
    aug_dict = {}
    aug_dict['no augment'] = 'no augment'
    aug_dict['CenterCrop_2_of_5'] = CenterCrop
    aug_dict['CenterCrop_1_of_5'] = CenterCrop_2
    aug_dict['Rotation'] = Rotation
    aug_dict['Flip'] = Flip
    #COLOR
    aug_dict['CLAHE'] = CLAHE
    aug_dict['Sharpen'] = Sharpen
    aug_dict['ChannelShuffle'] = ChannelShuffle
    aug_dict['ToGray'] = ToGray
    aug_dict['ToSepia'] = ToSepia
    aug_dict['GaussNoise'] = GaussNoise
    aug_dict['Normalize'] = Normalize
    #Cutout
    aug_dict['GridDropout'] = GridDropout
    aug_dict['Cutout'] = Cutout
    
    
    for run_name in aug_dict.keys():
        augmentation = aug_dict[run_name]
        classifier.single_run(args, given_augment = augmentation, run_name = run_name, project_name = 'GOOD CLASSIFIER')

    # for run_name in HP:
    #     augmentation = HP[run_name]
    #     classifier.single_run(args, given_augment = augmentation, run_name = run_name)
        