import model.classifier as classifier
from  config.parser_config import config_parser
import albumentations as A
import numpy as np




if __name__ == "__main__":
    #CONFIG PARSER
    parser = config_parser()
    args  = parser.parse_args()
    
    Perspective_HP = {}
    for scale in [0.2,0.3,0.4,0.5,0.6,0.7]:
        name = 'Perspective scale_{}'.format(scale)
        Perspective_HP[scale] = A.Perspective(p=1.0, scale=scale)
    
    MotionBlur_HP = {}
    for blur_limit in [11, 21 , 31, 41, 51, 61, 71]:
        name = 'MotionBlur_ blur_limit_{}'.format(blur_limit)
        MotionBlur_HP[name] = A.MotionBlur(p=1.0, blur_limit=blur_limit)
    
    Solarize_HP = {}
    for thresold in [64, 128, 192]:
        name = 'Solarize thresold_{}'.format(thresold)
        Solarize_HP[name] = A.Solarize(p=1.0, threshold=thresold)
        
    # def RandAugment(image, n = 3, m = 10):
    # # Define a set of possible image transformations
    #     transforms_list = [CenterCrop,
    #                        GaussNoise,
    #                        Sharpen,
    #                        ChannelShuffle,
    #                        ColorJitter,
    #                        Solarize,
    #                        ToSepia]
    #     # Apply a random sequence of n transformations with magnitude m
    #     aug = A.Compose([transforms_list[i] for i in np.random.choice(len(transforms_list), n)])
    #     aug_image = aug(image = image.astype(np.uint8))['image']
    #     output = {}
    #     output['image'] = aug_image
    #      #FOR VISUALISATION
    #     output['augs'] = [str(x).partition('(')[0] for x in aug]
    #     return output
    
    HP = {**Perspective_HP, **MotionBlur_HP, **Solarize_HP}

    for run_name in HP:
        augmentation = HP[run_name]
        classifier.single_run(args, given_augment = augmentation, run_name = run_name, project_name= 'GOOD HP TUNING')

        