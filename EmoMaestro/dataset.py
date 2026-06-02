import os
from abc import ABC
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

class FaceDataset(ABC):
    def __init__(
        self,
        dataset_path,
        name,
        width,
        height,
        channels,
        enc,
    ):
        self.__test = None
        self.__train = None
        self.__n_test_case = None
        self.__n_train_case = None
        self.__dataset_path = dataset_path

        self.__width = width
        self.__height = height
        self.__channels = channels

        self.__enc = sorted(enc)

        self.__x_train = None
        self.__y_train = None

        self.__x_test = None
        self.__y_test = None

        self.__batch_size = None

        self.__name = name

        self.__train_dir = "train"
        self.__test_dir = "test"
        self.__val_dir = "val"

    def load_data(self, batch_size, target_w, target_h, target_c):
        print(f"Loading {self.__name} dataset")

        transform_ops = []

        if self.__width != target_w or self.__height != target_h:
            transform_ops.append(transforms.Resize((target_w, target_h)))

        if self.__channels == 1:
            transform_ops.append(transforms.Grayscale(num_output_channels = target_c))

        transform_ops.extend([transforms.ToTensor(), transforms.Normalize(mean = 0.5, std = 0.5)])

        transform = transforms.Compose(transform_ops)

        train_ds = datasets.ImageFolder(os.path.join(self.__dataset_path, self.__train_dir), transform = transform)
        test_ds = datasets.ImageFolder(os.path.join(self.__dataset_path, self.__test_dir), transform = transform)
        val_ds = datasets.ImageFolder(os.path.join(self.__dataset_path, self.__val_dir), transform = transform)

        # Validation data treated as additional test data for now

        norm_train_classes = {s.casefold() for s in set(train_ds.classes)}
        norm_test_classes = {s.casefold() for s in set(test_ds.classes)}
        norm_val_classes = {s.casefold() for s in set(val_ds.classes)}
        labels_set = {s.casefold() for s in set(self.__enc)}

        if norm_train_classes != labels_set:
            raise Exception("Classes do not match in training data")

        if norm_test_classes != labels_set != norm_val_classes:
            raise Exception("Classes do not match in testing data")

        test_ds = ConcatDataset([test_ds, val_ds])

        self.__n_train_case = len(train_ds)
        self.__n_test_case = len(test_ds)

        self.__batch_size = batch_size

        self.__train = DataLoader(train_ds, batch_size = self.__batch_size, shuffle = True)
        self.__test = DataLoader(test_ds, batch_size = self.__batch_size, shuffle = False)

        print(f"Successfully loaded {self.__name} dataset")

    def get_train_data(self):
        return self.__train

    def get_test_data(self):
        return self.__test

    def get_enc(self):
        return self.__enc

    def get_train_size(self):
        return self.__n_train_case

    def get_test_size(self):
        return self.__n_test_case

# Real Datasets
class AffectNet(FaceDataset):
    name = "AffectNet"

    dataset_path = "datasets/affectnet"

    enc = ("anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")

    width = 112
    height = 112
    channels = 3

    def __init__(self):
        super().__init__(
            dataset_path = AffectNet.dataset_path,
            name = AffectNet.name,
            width = AffectNet.width,
            height = AffectNet.height,
            channels = AffectNet.channels,
            enc = AffectNet.enc,
        )

class CKPlus(FaceDataset):
    name = "CK+"

    dataset_path = "datasets/ckplus"

    enc = ("anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")

    width = 75
    height = 75
    channels = 1

    def __init__(self):
        super().__init__(
            dataset_path = CKPlus.dataset_path,
            name = CKPlus.name,
            width = CKPlus.width,
            height = CKPlus.height,
            channels = CKPlus.channels,
            enc = CKPlus.enc,
        )

class FERPlus(FaceDataset):
    name = "FERPlus"

    dataset_path = "datasets/ferplus"

    enc = ("angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")

    width = 112
    height = 112
    channels = 1

    def __init__(self):
        super().__init__(
            dataset_path = FERPlus.dataset_path,
            name = FERPlus.name,
            width = FERPlus.width,
            height = FERPlus.height,
            channels = FERPlus.channels,
            enc = FERPlus.enc,
        )