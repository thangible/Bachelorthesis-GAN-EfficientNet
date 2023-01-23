import albumentations as A

def aug_transfrom(
    scale = 0.5
    # hsv_h = 0.0,
    # hsv_s = 0.0,
    # hsv_v = 0.0,
    # translate = 0.0,
    # scale = 0.0,
    # shear = 0.0, 
    # perspective = 0.0,
    # flipud = 0.0,
    # fliplr = 0.0,
    # mosaic = 0.0,
    # mixup = 0.0,
    # copy_paste = 0.0,
    ):
        
    # trans = A.Compose([
    #         A.VerticalFlip(p=0.5),
    #         A.RandomScale(scale_limit=0.2, interpolation=1,
    #                     always_apply=False, p=0.5),
    #         A.Rotate(limit=180, p=0.4),
    #         A.CLAHE(p=0.8),
    #         A.RandomBrightnessContrast(p=0.3),
    #         A.RandomGamma(p=0.3),
    #         A.HueSaturationValue(p=0.1)
    #     ])
    
    trans =  A.Compose(A.augmentations.geometric.resize.RandomScale (scale_limit=0.1, 
                                                                     interpolation=1, 
                                                                     always_apply=False,
                                                                     p=0.5))
    return trans



