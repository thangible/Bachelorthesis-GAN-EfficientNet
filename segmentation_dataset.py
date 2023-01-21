from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import PurePath
import torch.nn.functional as Functional
import numpy as np
import torch
from torchvision.io import read_image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


class ClassificationDataset(Dataset):
    def __init__(
        self,
        unwanted_pics: list = [],
        unwanted_classes: list = [],
        npz_path: PurePath = None,
        image_path: PurePath = None,
        label_path: PurePath = None,
        size: int = None,
        one_hot: bool = True,
        augmentation = None,
        ):
        
        self._image_path = image_path
        self._label_path = label_path
        self._npz_path = npz_path
        self._unwanted_classes = unwanted_classes
        self._unwanted_pics = unwanted_pics
        self._size = size
        self._augmentation = augmentation
        self._one_hot = one_hot
        
        #LOADER
        if self._npz_path:
            with np.load(self._npz_path, mmap_mode='r') as data:
                self._images = data['x']
                self._labels = data['y']
                self._categories = data['z']
        else:
            self._image_names, self._labels, self._categories = self._get_names_and_labels_categories()

        self._num_classes = self._get_num_classes()
        
        
        
        

        
        
    def __getitem__(self, index: int) -> tuple:
        ##LOADER
        img = self._get_image(index)
        label = self._get_label(index)
        return img, label
    
    def __len__(self) -> int:
        return self._labels.shape[0]
    
    
    def _get_names_and_labels_categories(self):
        #PREPROCESSING
        all_data = pd.read_csv(self._label_path,index_col=False)
        data = all_data[['tp','name','file']].copy()
        data.reset_index(drop=True, inplace=True)
        unwanted_classes = self._unwanted_classes
        no_labelled_pics = self._unwanted_pics
        data = data[~data['name'].isin(unwanted_classes)]
        data = data[~data.file.isin(no_labelled_pics)]
        
        ##GET LABELS FROM CATEGOGRIES
        le = preprocessing.LabelEncoder()
        name = data['name']
        le.fit(name)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        data['label'] = le.transform(name)
        data = data.sort_values(by=['file'][:-4],ascending=False)
        
        #ASSIGNING
        image_names = data.file.array
        labels = data.label.array
        categories = data.name.array
        return image_names, labels, categories
        
    def _get_image(self, index: int):
        if self._npz_path:
            image =  self._images[index]
        else:
          image_name = self._image_names[index]
          path = PurePath(self._image_path, image_name)
          image = read_image(str(path))
        #RESIZE
        if self._size != image.shape[0]:
            image = self._resize(image)
        #AUGMENTATION
        if self._augmentation:
            image = self._augmentation(image)
            
        return image
        
    def _get_label(self, index: int):
        label = self._labels[index]
        if self._one_hot:
            label = self._one_hot_transform(label)
        return label
    
    def _get_category(self, index: int):
        category = self._categories[index]
        return category
        
    def _one_hot_transform(self, label):
        one_hot_transform = transforms.Compose([
        lambda x: torch.as_tensor(x),
        lambda x: Functional.one_hot(x.to(torch.int64), self._num_classes)
    ])
        return one_hot_transform(label)
        
        
    def _resize(self, image):
        transform_resize = transforms.Resize([self._size, self._size])
        return transform_resize(image)   
    
    def _show_pic(self, index: int):
        image = self._get_image(index)
        label = self._labels[index]
        category = self._get_category(index)
        fig = plt.figure()
        plt.imshow(image.permute(1, 2, 0))
        plt.title('Category: {}, Label: {}'.format(label, category))
    
    def _get_num_classes(self):
        return len(np.unique(self._labels))
    