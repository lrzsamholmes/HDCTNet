import os
import cv2
import torch
import config
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation

import imgaug as ia
from imgaug import augmenters as iaa

class RangeNormaliziation(object):
    def __call__(self, sample):
        img, lbl = sample
        return img / 255.0 * 3.2 - 1.6, lbl

class ToTensor(object):
    def __call__(self, sample):
        img, lbl = sample
        lbl = torch.from_numpy(lbl).long()
        img = torch.from_numpy(np.array(img, np.float32).transpose(2, 0, 1))
        return img, lbl

# preprocesses ground-truth labelmap (data expected to already provide tubules border as label 7)
def preprocessingGT(lbl):
    structure = np.zeros((3, 3), dtype=np.int)
    structure[1, :] = 1
    structure[:, 1] = 1

    # add glomeruli border only for almost touching glomeruli
    allGlomeruli = np.logical_or(lbl == 2, lbl == 3)
    labeledGlom, numberGlom = label(np.asarray(allGlomeruli, np.uint8), structure)
    temp = np.zeros(lbl.shape)
    for i in range(1, numberGlom + 1):
        temp += binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(labeledGlom == i)))))))
    glomBorder = np.logical_and(temp > 1, np.logical_not(allGlomeruli))
    lbl[binary_dilation(glomBorder)] = 7

    # add arterial border only for almost touching arteries
    allArteries = np.logical_or(lbl == 5, lbl == 6)
    labeledGlom, numberGlom = label(np.asarray(allArteries, np.uint8), structure)
    temp = np.zeros(lbl.shape)
    for i in range(1, numberGlom + 1):
        temp += binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(labeledGlom == i)))))
    glomBorder = np.logical_and(temp > 1, np.logical_not(allArteries))
    lbl[binary_dilation(glomBorder)] = 7

class CustomDataSetRAM(Dataset):
    def __init__(self, datasetType, logger):
        self.transformIMG = None
        self.transformLBL = None
        self.transform_WhenNoAugm = transforms.Compose([RangeNormaliziation(), ToTensor()])

        self.data = []
        self.lblShape = 0

        self.useAug = config.useAug and datasetType=='train'
        if self.useAug:
            self.transformIMG, self.transformLBL = get_Augmentation_Transf()
            logger.info('use data augmentation')

        assert datasetType in ['train', 'val', 'test'], '### ERROR: WRONG DATASET TYPE '+datasetType+' ! ###'

        image_dir_base = config.image_dir_base

        if datasetType == 'train':
            image_dir = image_dir_base + '/Train'
        elif datasetType == 'val':
            image_dir = image_dir_base + '/Val'
        elif datasetType == 'test':
            image_dir = image_dir_base + '/Test'

        # here we expect labels to be stored in same directory as respective images with ending '-labels.png' instead of '.png'
        label_dir = image_dir
        files = sorted(list(filter(lambda x: ').png' in x, os.listdir(image_dir))))

        logger.info('Loading dataset with size: '+str(len(files)))
        for k, fname in enumerate(files):
            imagePath = os.path.join(image_dir, fname)
            labelPath = os.path.join(label_dir, fname.replace('.png', '-labels.png'))

            if config.useSmallImg:
                img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                img = img[:640, :640, :]    
                img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
                lbl = np.array(Image.open(labelPath).resize((258, 258), Image.ANTIALIAS))
            else:
                img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                img = img[:640, :640, :]
                lbl = np.array(Image.open(labelPath))

            # fix bug in the dataset
            for row in range(0, lbl.shape[0]):
                for col in range(0, lbl.shape[1]):
                    if lbl[row][col] == 7:
                        lbl[row][col] = 8
                    elif lbl[row][col] == 8:
                        lbl[row][col] = 7

            # preprocess ground truth
            preprocessingGT(lbl)

            logger.info("Load data with index " + str(k) + " : " + fname + ", ImgShape: " + str(img.shape) + " " + str(img.dtype)
                        + ", LabelShape: " + str(lbl.shape) + " " + str(lbl.dtype) + " (max: " + str(lbl.max()) + ", min: " + str(lbl.min()) + ")")

            self.lblShape = lbl.shape
            # most likely, shapes are not equal, then pad label map to same size as images with values of 8 (which will be ignored for loss computation),
            # providing equal sizes simplifies the appliacation of data augmentation transformation 
            if img.shape[:2] != lbl.shape:
                lbl = np.pad(lbl, ((img.shape[0]-lbl.shape[0])//2,(img.shape[1]-lbl.shape[1])//2), 'constant', constant_values=(8,8))

            self.data.append((img, lbl))

        assert len(files) > 0, 'No files found in ' + image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.useAug:
            # get different augmentation transformation for each sample within minibatch
            ia.seed(np.random.get_state()[1][0])

            img, lbl = self.data[index]

            seq_img_d = self.transformIMG.to_deterministic()
            seq_lbl_d = self.transformLBL.to_deterministic()

            # apply almost equal transformation for label maps (however using nearest neighbor interpolation)
            seq_lbl_d = seq_lbl_d.copy_random_state(seq_img_d, matching="name")

            # after applying the transformation, center crop label map back to its original size
            augmentedIMG = seq_img_d.augment_image(img)
            augmentedLBL = seq_lbl_d.augment_image(lbl)[(img.shape[0]-self.lblShape[0])//2:(img.shape[0]-self.lblShape[0])//2+self.lblShape[0],
                                                        (img.shape[1]-self.lblShape[1])//2:(img.shape[1]-self.lblShape[1])//2+self.lblShape[1]]

            return self.transform_WhenNoAugm((augmentedIMG, augmentedLBL.copy()))
        else:
            img, lbl = self.data[index]
            return self.transform_WhenNoAugm((img, lbl[(img.shape[0]-self.lblShape[0])//2:(img.shape[0]-self.lblShape[0])//2+self.lblShape[0],
                                                       (img.shape[1]-self.lblShape[1])//2:(img.shape[1]-self.lblShape[1])//2+self.lblShape[1]]))

def get_Augmentation_Transf():
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug, name="Random1")
    sometimes2 = lambda aug: iaa.Sometimes(0.2, aug, name="Random2")
    sometimes3 = lambda aug: iaa.Sometimes(0.9, aug, name="Random3")
    sometimes4 = lambda aug: iaa.Sometimes(0.9, aug, name="Random4")
    sometimes5 = lambda aug: iaa.Sometimes(0.9, aug, name="Random5")

    # specify DATA AUGMENTATION TRANSFORMATION
    seq_img = iaa.Sequential([
        iaa.AddToHueAndSaturation(value=(-13, 13), name="MyHSV"),
        sometimes2(iaa.GammaContrast(gamma=(0.85, 1.15), name="MyGamma")),
        iaa.Fliplr(0.5, name="MyFlipLR"),
        iaa.Flipud(0.5, name="MyFlipUD"),
        sometimes(iaa.Rot90(k=1, keep_size=True, name="MyRot90")),
        iaa.OneOf([
            sometimes3(iaa.PiecewiseAffine(scale=(0.015, 0.02), cval=0, name="MyPiece")),
            sometimes4(iaa.ElasticTransformation(alpha=(100, 200), sigma=20, cval=0, name="MyElastic")),
            sometimes5(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, rotate=(-45, 45), shear=(-4, 4), cval=0, name="MyAffine"))
        ], name="MyOneOf")
    ], name="MyAug")

    seq_lbl = iaa.Sequential([
        iaa.Fliplr(0.5, name="MyFlipLR"),
        iaa.Flipud(0.5, name="MyFlipUD"),
        sometimes(iaa.Rot90(k=1, keep_size=True, name="MyRot90")),
        iaa.OneOf([
            sometimes3(iaa.PiecewiseAffine(scale=(0.015, 0.02), cval=8, order=0, name="MyPiece")),
            sometimes4(iaa.ElasticTransformation(alpha=(100, 200), sigma=20, cval=8, order=0, name="MyElastic")),
            sometimes5(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, rotate=(-45, 45), shear=(-4, 4), cval=8, order=0, name="MyAffine"))
        ], name="MyOneOf")
    ], name="MyAug")

    return seq_img, seq_lbl