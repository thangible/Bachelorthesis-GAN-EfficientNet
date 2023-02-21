import model.classifier as classifier
from  config.parser_config import config_parser
import albumentations as A
import numpy as np




if __name__ == "__main__":
    #CONFIG PARSER
    parser = config_parser()
    args  = parser.parse_args()
    
    Resize = A.Resize(height = 500, width = 500, interpolation=1, p=1.0)
    CenterCrop = A.Compose([A.CenterCrop(height = 200, width = 200, p=1.0),Resize])
    CenterCrop_2 = A.Compose([A.CenterCrop(height = 100, width = 100, p=1.0),Resize])
    MotionBlur =  A.MotionBlur(blur_limit =31,  p=1.0)
    Clahe = A.CLAHE(p=1.0)
    Sharpen = A.Sharpen(p=1.0)
    ColorJitter =  A.ColorJitter( p=1.0)
    Solarize = A.Solarize(p=1)
    ToGray = A.ToGray(p=1)
    RandomGridShuffle = A.RandomGridShuffle(grid=(3, 3), p=1.0)
    Perspective = A.Perspective(scale = (0.2,0.1),  p=1)
    Posterize = A.Posterize(p= 1)
    CropThenSharpen = A.Compose([CenterCrop, Sharpen])
    CropThenCLAHE = A.Compose([CenterCrop, Clahe])
    ChannelShuffle = A.ChannelShuffle(p =1)
    Equalize = A.Equalize( p =1)
    GaussNoise = A.GaussNoise(p =1)
    Normalize = A.Normalize(p=1)
    RandomConstrast = A.RandomContrast(limit=(-0.9, 0.9),p=1)
    ToSepia = A.ToSepia(p=1)
    Cutout = A.CoarseDropout(p =1, max_height=50, max_width=50)
    GridDropout = A.GridDropout(p=1)
    
    def RandAugment(image, n = 3, m = 10):
    # Define a set of possible image transformations
        transforms_list = [CenterCrop,
                           GaussNoise,
                           Sharpen,
                           Cutout,
                           ChannelShuffle,
                           ColorJitter,
                           Solarize,
                           ToSepia]
        # Apply a random sequence of n transformations with magnitude m
        aug = A.Compose([transforms_list[i] for i in np.random.choice(len(transforms_list), n)])
        aug_image = aug(image = image.astype(np.uint8))['image']
        output = {}
        output['image'] = aug_image
         #FOR VISUALISATION
        output['augs'] = [str(x).partition('(')[0] for x in aug]
        return output
    
    #HYPERPARAMETER TUNING
    ClaheClip4Tile20 = A.CLAHE(clip_limit = 4, tile_grid_size=(20, 20), p=1.0)
    ClaheClip12Tile8 = A.CLAHE(clip_limit = 12, tile_grid_size=(8, 8), p=1.0)
    SharpenAlpha510Lightness510 = A.Sharpen(alpha=(0.5, 1), lightness = (0.2, 0.5), p=1.0)
    SharpenAlpha25Lightness810 = A.Sharpen(alpha=(0.5, 1), lightness = (0.8, 1) ,p=1.0)
    
    
    
    
    ColoJitter_HP = {}
    for i in range(4):
        brightness = np.random.uniform(0.2, 0.8)
        contrast = np.random.uniform(0.2, 0.8)
        saturation = np.random.uniform(0.2, 0.8)
        hue = np.random.uniform(0.2, 0.8)
        key = 'ColorJitter brightness:{:.2f}, contrast:{:.2f}, saturation:{:.2f}, hue:{:.2f}'.format(brightness, contrast, saturation, hue)
        ColoJitter_HP[key](A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue , p=1.0))
    
    Solarize_HP = {}
    for threshold in  [64, 128, 192]:
        key = 'Solarize Threshold: {}'.format(threshold)
        Solarize_HP[key] = A.Solarize(threshold = threshold, p = 1.0)
        
    Normalize_HP  = {}
    for i in range(4):
        means = tuple(np.random.normal(loc = 0.4, scale = 0.5, size =(3)))
        stds = tuple(np.random.uniform(0.1, 0.8, size = (3)))
        key = 'Normalize means: ({:.3f}, {:.3f}, {:.3f}), std = {:.3f}, {:.3f}, {:.3f})'.format(*means,*stds)
        Solarize_HP[key] = A.Normalize(mean = means, std = stds, p = 1.0)
        
    
    HP = {**ColoJitter_HP, **Solarize_HP,**Normalize_HP}
    # augmentations = ['no augment']
    # run_names = ['New Baseline - No Augment'] 
    
    
    # for i in range(len(augmentations)):
    #     classifier.single_run(args, given_augment = augmentations[i], run_name = run_names[i])
    for run_name in HP:
        augmentation = HP[run_name]
        classifier.single_run(args, given_augment = augmentation, run_name = run_name)
        