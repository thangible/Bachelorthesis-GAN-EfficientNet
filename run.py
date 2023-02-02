import classifier
from  parser_config import config_parser
import albumentations as A




if __name__ == "__main__":
    #CONFIG PARSER
    parser = config_parser()
    args  = parser.parse_args()
    
    CenterCrop = A.Compose([A.CenterCrop(height = 200, width = 200, p=1.0), 
                         A.Resize(height = 500, width = 500, interpolation=1, p=1.0)])
    MotionBlur =  A.MotionBlur(blur_limit =31,  p=1.0)
    Clahe = A.CLAHE(p=1.0)
    Sharpen = A.Sharpen(alpha = (0.2, 0.5), lightness = (0.5, 1.0), p=1.0)
    ColorJitter =  A.ColorJitter( p=1.0)
    Solarize = A.Solarize(p=1)
    ToGray = A.ToGray(p=1)
    
    augmentations = ['no augment', Sharpen, ColorJitter]
    run_names = ['no augment',
                 'Sharpen - alpha (0.2, 0.5), lightness (0.5,1)',
                 'ColorJitter - Default',
                 'Solarize - Default',
                 'ToGray']
    
    
    for i in range(len(augmentations)):
        classifier.single_run(args, given_augment = augmentations[i], run_name = run_names[i])
        