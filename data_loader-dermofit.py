import numpy as np
import torch.utils.data as data
import os
import sys
import errno
from PIL import Image
import random
import torch

class Dermofit(data.Dataset):
    raw_folder = 'raw'
    raw_train = 'dermofit_train_new'
    raw_test = 'dermofit_test'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    
    def __init__(self, root, train=True, transform=None, target_transform=None, process=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        
        if process:
            if(os.path.exists(os.path.join(self.root, self.raw_folder, self.raw_train)) and \
               os.path.exists(os.path.join(self.root, self.raw_folder, self.raw_test))):
                self.process()
            else:
                raise RuntimeError('Raw data not found. Please make sure raw data is there in raw folder')
                
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use process=True to process the already placed raw data')
        
        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
                    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))
    
    def __getitem__(self, index):
        
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
            
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    def process(self):
        
        if self._check_exists():
            return
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        print("\nProcessing...\n")
        
        train_data_links = images_and_labels_path(os.path.join(self.root, self.raw_folder, self.raw_train))
        test_data_links = images_and_labels_path(os.path.join(self.root, self.raw_folder, self.raw_test))
        train_img_and_lbl_dirs = get_labels_and_files(train_data_links)
        test_img_and_lbl_dirs = get_labels_and_files(test_data_links)
        random.shuffle(train_img_and_lbl_dirs)
        random.shuffle(test_img_and_lbl_dirs)
        train_images, train_labels = get_arrays(train_img_and_lbl_dirs, 'train')
        test_images, test_labels = get_arrays(test_img_and_lbl_dirs, 'test')
        training_set = (train_images, train_labels)
        test_set = (test_images , test_labels)
        
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        print("Done..!")
        
def images_and_labels_path(directory):
    data = []
    data_class = []
    img_dir = list(np.sort(os.listdir(directory)))
    for i in img_dir:
        class_dir = list(os.listdir(os.path.join(directory, i)))
        for j in class_dir:
            data_class.append(os.path.join(directory, i, j))
        data.append(data_class)
        data_class = []
    return data

def get_labels_and_files(links):
    label_and_files = []
    
    for class_lb, class_links in enumerate(links):
        for link in class_links:
            tmp = (class_lb, link)
            label_and_files.append(tmp)
    return label_and_files

def get_arrays(img_and_lbls, name):
    images = []
    labels = []
    for i in range(len(img_and_lbls)):
        
        #display progress
        if(i%100 == 0):
            sys.stdout.write("\r%d%% complete" % ((i * 100)/len(img_and_lbls)))
            sys.stdout.flush()
        
        filename = img_and_lbls[i][1]
        try:
            f = Image.open(filename)
            resized_img = f.resize((224, 224), resample = Image.LANCZOS)
            image = np.asarray(resized_img)
            images.append(image)
            labels.append(img_and_lbls[i][0])
        except:
            #if there is some sort of garbage valuse rather then image class and path
            print("\nCan't read image file " + filename)
            
    count = len(images)
    
    image_data = np.zeros((count,224,224, 3), dtype=np.uint8)
    label_data = np.zeros(count, dtype = np.int_)
    
    for i in range(len(img_and_lbls)):
        image_data[i] = images[i]
        label_data[i] = labels[i]
    
    print("\n")
    
    print(name, "processing 100% complete\n")
    return image_data, label_data