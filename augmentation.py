import albumentations as A

def aug_transfrom(flip = 0.5):
    trans = A.Compose([
            A.VerticalFlip(p=0.5),
            A.RandomScale(scale_limit=0.2, interpolation=1,
                        always_apply=False, p=0.5),
            A.Rotate(limit=180, p=0.4),
            A.CLAHE(p=0.8),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.HueSaturationValue(p=0.1)
        ])
    return trans



