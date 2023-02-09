import model.classifier as classifier
from  EfficientNet.config.parser_config import config_parser
import albumentations as A




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
    
    augmentations = [CropThenCLAHE, GaussNoise, Normalize, ChannelShuffle, RandomConstrast, ToSepia]
    run_names = ['CropThenCLAHE','GaussNoise','Normalize', 'ChannelShuffle', 'RandomConstrast limit = 0.9','ToSepia']
    
    
    for i in range(len(augmentations)):
        classifier.single_run(args, given_augment = augmentations[i], run_name = run_names[i])
        