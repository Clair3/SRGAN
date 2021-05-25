from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from torchvision.transforms.functional import crop

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.tif', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),  # Convert a PIL Image or numpy.ndarray to tensor.
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

def train_hr_transform_with_lr(img, crop_windows):
    i, j, h, w = crop_windows
    img = crop(img, i, j, h, w)
    return ToTensor()(img)

def train_lr_transform_with_lr(img, crop_size, upscale_factor, crop_windows):
    i, j, h, w = crop_windows
    img = crop(img, i, j, h, w)
    img = Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC)(img)
    return ToTensor()(img)


def Randomcrop(imageHR, crop_size):
    return RandomCrop.get_params(imageHR, output_size=(crop_size, crop_size))


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        #hr_image = CenterCrop(176)(hr_image)
        hr_image = self.hr_transform(hr_image)
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class TrainDatasetFromFolderWithLR(Dataset):
    def __init__(self, datasetHR_dir, datasetLR_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolderWithLR, self).__init__()
        self.imageHR_filenames = [join(datasetHR_dir, x) for x in sorted(listdir(datasetHR_dir)) if is_image_file(x)] # list des images HR
        self.imageLR_filenames = [join(datasetLR_dir, x) for x in sorted(listdir(datasetLR_dir)) if is_image_file(x)]
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.upscale_factor = upscale_factor
        #self.hr_transform = train_hr_transform_with_lr(crop_size)
        #self.lr_transform = train_lr_transform_with_lr(crop_size, upscale_factor)

    def __getitem__(self, index):
        imageHR = Image.open(self.imageHR_filenames[index])
        imageLR = Image.open(self.imageLR_filenames[index])

        imageHR = CenterCrop(164)(imageHR)
        imageLR = CenterCrop(164)(imageLR)

        crop_windows = Randomcrop(imageHR, self.crop_size)
        hr_image = train_hr_transform_with_lr(imageHR,crop_windows)
        lr_image = train_lr_transform_with_lr(imageLR, self.crop_size, self.upscale_factor, crop_windows)
        return lr_image, hr_image

    def __len__(self):
        return len(self.imageHR_filenames)


class ValDatasetFromFolderWithLR(Dataset):
    def __init__(self, datasetHR_dir, datasetLR_dir, upscale_factor):
        super(ValDatasetFromFolderWithLR, self).__init__()
        self.upscale_factor = upscale_factor
        self.imageHR_filenames = [join(datasetHR_dir, x) for x in listdir(datasetHR_dir) if is_image_file(x)]  # The join() method takes all items in an iterable and joins them into one string.
        self.imageLR_filenames = [join(datasetLR_dir, x) for x in listdir(datasetLR_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.imageHR_filenames[index])
        lr_image = Image.open(self.imageLR_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(lr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.imageHR_filenames)




class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

class ValDatasetFromFolderRealSize(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolderRealSize, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        hr_image = CenterCrop(176)(hr_image)
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
