from torch.utils.data import Dataset
from pathlib import PurePath
import numpy as np
import cv2
import pandas as pd
from sklearn import preprocessing


class ClassificationDataset(Dataset):
    def __init__(
        self,
        unwanted_pics: list,
        unwanted_classes: list,
        npz_path: PurePath = None,
        image_path: PurePath = None,
        label_path: PurePath = None,
        ):

        self._image_path = image_path
        self._label_path = label_path
        self._npz_path = npz_path
        self._unwanted_classes = unwanted_classes
        self._unwanted_pics = unwanted_pics
        
        
        #LOADER
        if self._npz_path:
            with np.load(self._npz_path, mmap_mode='r') as data:
                self._images = data['x']
                self._labels = data['y']
        else:
            self._image_names = self._get_image_names()
            self._labels = self._get_labels()
        
        image_path.glob()
        
        
    def __getitem__(self, index: int) -> tuple:
        ##LOADER
        img = self._get_image(index)
        label = self._labels[index]
            
        ##AUGMENTATION
        return img, label
    
    def __len__(self) -> int:
        return self.labels.shape[0]
    
    
    def _get_names_and_labels(self) -> tuple(pd.Series, pd.Series):
        all_data = pd.read_csv(self._label_path,index_col=False)
        data = all_data[['tp','name','file']].copy()
        data.reset_index(drop=True, inplace=True)
        unwanted_classes = self._unwanted_classes
        no_labelled_pics = self._unwanted_pics
        data = data[~data['name'].isin(unwanted_classes)]
        data = data[~data.file.isin(no_labelled_pics)]
        ## get labels
        le = preprocessing.LabelEncoder()
        label = data['name']
        le.fit(label)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        data['label'] = le.transform(label)
        data = data.sort_values(by=['file'][:-4],ascending=False)
        
        ##
        names = data.file
        labels = data.label
        return names, labels
        
    def _get_image(self, index):
        image_name = self._images[index]
        path = PurePath(self._image_path, image_name + '.jpg')
        return cv2.imread(path)
        
        