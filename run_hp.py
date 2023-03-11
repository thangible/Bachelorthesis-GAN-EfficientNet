import model.classifier as classifier
from  config.parser_config import config_parser
import albumentations as A
import numpy as np




if __name__ == "__main__":
    #CONFIG PARSER
    parser = config_parser()
    args  = parser.parse_args()
    
    # Perspective_HP = {}
    # for scale in [0.2,0.3,0.4,0.5,0.6,0.7]:
    #     name = 'Perspective scale_{}'.format(scale)
    #     Perspective_HP[name] = A.Perspective(p=1.0, scale=scale)
    
    # MotionBlur_HP = {}
    # for blur_limit in [11, 21 , 31, 41, 51, 61, 71]:
    #     name = 'MotionBlur_ blur_limit_{}'.format(blur_limit)
    #     MotionBlur_HP[name] = A.MotionBlur(p=1.0, blur_limit=blur_limit)
    
    # Solarize_HP = {}
    # for thresold in [64, 128, 192]:
    #     name = 'Solarize thresold_{}'.format(thresold)
    #     Solarize_HP[name] = A.Solarize(p=1.0, threshold=thresold)
    
    
    
    # ColoJitter_HP = {}
    # for i in range(8):
    #     brightness = np.random.uniform(0.2, 0.8)
    #     contrast = np.random.uniform(0.2, 0.8)
    #     saturation = np.random.uniform(0.2, 0.8)
    #     hue = np.random.uniform(0.2, 0.8)
    #     key = 'ColorJitter brightness:{:.2f}, contrast:{:.2f}, saturation:{:.2f}, hue:{:.2f}'.format(brightness, contrast, saturation, hue)
    #     ColoJitter_HP[key] = A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue , p=1.0)
        
    # Resize = A.Resize(height = 500, width = 500, interpolation=1, p=1.0)
    # MotionBlur = A.MotionBlur(blur_limit = 11, p = 1.0)
    # Rotation = A.Rotate(p=1.0)
    # Flip = A.Flip(p=1.0)
    # Sharpen = A.Sharpen(alpha= (0.6,0.8), lightness = (0.6,1.0), p=1.0)
    # CenterCrop = A.Compose([A.CenterCrop(height = 200, width = 200, p=1.0),Resize])
    # GaussNoise = A.GaussNoise(p =1)
    # GridDropout = A.GridDropout(ratio = 0.6, random_offset = True, p=1)
    
    # def Rand_HP(n):
    #     def RandAugment(image, n = n, m = 10):
    #     # Define a set of possible image transformations
    #         transforms_list = [Rotation, 
    #                         Flip,
    #                         MotionBlur,
    #                         Sharpen,
    #                         CenterCrop,
    #                         GaussNoise,
    #                         GridDropout
    #                         ]
    #         # Apply a random sequence of n transformations with magnitude m
    #         aug = A.Compose([transforms_list[i] for i in np.random.choice(len(transforms_list), n)])
    #         aug_image = aug(image = image.astype(np.uint8))['image']
    #         output = {}
    #         output['image'] = aug_image
    #         #FOR VISUALISATION
    #         output['augs'] = [str(x).partition('(')[0] for x in aug]
    #         return output
    #     return RandAugment
    
    # for n in range(2, 5):
    #     augmentation = Rand_HP(n)
    #     classifier.single_run(args, given_augment = augmentation, run_name = 'RandAugment', project_name= 'GOOD HP TUNING')

    # for run_name in ColoJitter_HP:
    #     augmentation = ColoJitter_HP[run_name]
    #     classifier.single_run(args, given_augment = augmentation, run_name = run_name, project_name= 'GOOD HP TUNING')
    
    classifier.single_run(args, given_augment = 'no augment', run_name = 'no augment', project_name= 'GOOD HP TUNING')

        