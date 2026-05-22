
"""
Data Loading, Augmentation, Scheduling, and Logging Utilities,

This file provides all supporting infrastructure required for training
vision models. It does not define model architectures, but instead
controls how data is loaded, preprocessed, weighted, scheduled, and logged.

Main Responsibilities:
----------------------
1. Load datasets (EcoSet or ImageNet) and create PyTorch DataLoaders.
2. Apply configurable image augmentations (resize, crop, blur, flip, etc.).
3. Compute class weights to compensate for class imbalance.
4. Define a custom learning rate scheduler based on linear trend fitting.
5. Create logging directories for saving metrics and model checkpoints.

Scientific Role:
----------------
This module defines the experimental environment under which models
are trained. While the architecture determines representational
capacity, this file determines the training protocol and data
conditions that shape learned representations.

In summary:
-----------
This file controls how data enters the model and how training is
managed, but it does not implement the neural network itself.

ssh lemoehlenkam@hpc3.rz.uos.de
"""


##################
### Importing required packages
##################

from torch.utils.data import Subset
import torch
import torchvision.transforms as transforms
import numpy as np
import h5py
import os
from sklearn.linear_model import LinearRegression
import random
from torchvision import datasets, transforms
from collections import Counter
from torchvision import transforms

class Ecoset(torch.utils.data.Dataset):
    #Import Ecoset as a Dataset splitwise

    def __init__(self, split, dataset_path, in_memory=False, transform=None):
        """
        Args:
            dataset_path (string): Path to the .h5 file
            transform (callable, optional): Optional transforms to be applied
                on a sample.
            in_memory: Should we pre-load the dataset?
        """
        self.root_dir = dataset_path
        self.transform = transform
        self.split = split
        self.in_memory = in_memory

        if self.in_memory:
            with h5py.File(dataset_path, "r") as f:
                self.images = torch.from_numpy(f[split]['data'][()]).permute((0, 3, 1, 2)) # to match the CHW expectation of pytorch
                self.labels = torch.from_numpy(f[split]['labels'][()].astype(np.int64))
            self._len = len(self.labels)
        else:
            # Only read the length here; actual data is read lazily per-worker
            with h5py.File(dataset_path, "r") as f:
                self._len = len(f[split]['labels'])
            self._h5_file = None
            self.images = None
            self.labels = None

    def _open_h5(self):
        """Lazily open the HDF5 file in the current worker process.
        Limit the chunk cache to 32 MB to prevent RAM from ballooning
        when many DataLoader workers each hold their own file handle."""
        self._h5_file = h5py.File(
            self.root_dir, "r",
            rdcc_nbytes=32 * 1024 * 1024,   # 32 MB chunk cache per worker
            rdcc_nslots=10007,               # prime number of hash slots
        )
        self.images = self._h5_file[self.split]['data']
        self.labels = self._h5_file[self.split]['labels']
        self._access_count = 0
        # Open a separate fd to advise the OS about page cache
        try:
            self._advise_fd = os.open(self.root_dir, os.O_RDONLY)
            # Tell kernel not to readahead (DataLoader shuffles randomly)
            os.posix_fadvise(self._advise_fd, 0, 0, os.POSIX_FADV_RANDOM)
        except (AttributeError, OSError):
            self._advise_fd = None  # Not on Linux or not supported

    def __len__(self):
        return self._len

    def __del__(self):
        # Clean up file descriptors
        if hasattr(self, '_advise_fd') and self._advise_fd is not None:
            try:
                os.close(self._advise_fd)
            except OSError:
                pass

    def __getitem__(self, idx): # accepts ids and returns the images and labels transformed to the Dataloader
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.in_memory:
            imgs = self.images[idx]
            labels = self.labels[idx]
        else:
            if self.images is None:
                self._open_h5()
            imgs = torch.from_numpy(np.asarray(self.images[idx])).permute((2,0,1))    
            labels = torch.from_numpy(np.asarray(self.labels[idx].astype(np.int64)))

            # Evict cached file pages from system RAM on every read.
            # The ~300GB H5 file on NFS fills the cgroup page cache
            # and triggers OOM if pages aren't actively evicted.
            self._access_count += 1
            if getattr(self, '_advise_fd', None) is not None:
                try:
                    os.posix_fadvise(self._advise_fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass

        if self.transform:
            imgs = self.transform(imgs)

        return imgs, labels

##############################
## Loading the dataset loaders
##############################



def get_Dataset_loaders(hyp, splits):

    import torch
    import numpy as np

    dataset_mode = hyp.get("dataset_mode", 0)

    # ==========================================================
    # MODE 1 — FakeData (pure debugging, no real learning)
    # ==========================================================
    if dataset_mode == 1:
        print("Using FakeData dataset")

        from torchvision.datasets import FakeData
        from torchvision import transforms
        from torch.utils.data import DataLoader

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        train_data = FakeData(
            size=200,
            image_size=(3, 224, 224),
            num_classes=100,
            transform=transform
        )

        val_data = FakeData(
            size=50,
            image_size=(3, 224, 224),
            num_classes=100,
            transform=transform
        )

        hyp['dataset']['n_classes'] = 100
        hyp['dataset']['class_weights'] = None

        train_loader = DataLoader(
            train_data,
            batch_size=hyp['optimizer']['batch_size'],
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_data,
            batch_size=hyp['misc']['batch_size_val_test'],
            num_workers=0
        )

        return train_loader, val_loader, None, hyp


    # ==========================================================
    # MODE 2 — CIFAR100 (real small dataset, local experiments)
    # ==========================================================
    if dataset_mode == 2:
        print("Using CIFAR100 dataset")

        from torchvision.datasets import CIFAR100
        from torch.utils.data import DataLoader
        from torchvision import transforms

        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        transform_val_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # TRAIN
        if 'train' in splits:
            train_data = CIFAR100(
                root='./data',
                train=True,
                download=True,
                transform=transform_train
            )

            train_loader = DataLoader(
                train_data,
                batch_size=hyp['optimizer']['batch_size'],
                shuffle=True,
                num_workers=hyp['optimizer']['dataloader']['num_workers_train']
            )
        else:
            train_loader = None

        # VALIDATION
        if 'val' in splits:
            val_data = CIFAR100(
                root='./data',
                train=False,
                download=True,
                transform=transform_val_test
            )

            val_loader = DataLoader(
                val_data,
                batch_size=hyp['misc']['batch_size_val_test'],
                shuffle=False,
                num_workers=hyp['optimizer']['dataloader']['num_workers_val_test']
            )
        else:
            val_loader = None

        # TEST (separate loader, same split)
        if 'test' in splits:
            test_data = CIFAR100(
                root='./data',
                train=False,
                download=True,
                transform=transform_val_test
            )

            test_loader = DataLoader(
                test_data,
                batch_size=hyp['misc']['batch_size_val_test'],
                shuffle=False,
                num_workers=hyp['optimizer']['dataloader']['num_workers_val_test']
            )
        else:
            test_loader = None

        hyp['dataset']['n_classes'] = 100
        hyp['dataset']['class_weights'] = None

        print(f"Number of classes: {hyp['dataset']['n_classes']}")

        return train_loader, val_loader, test_loader, hyp


    # ==========================================================
    # MODE 0 — Default EcoSet
    # ==========================================================
    if hyp['dataset']['name'] in ['ecoset', 'miniecoset']:

        if hyp['dataset']['name'] == 'miniecoset':
            print('Getting MiniEcoSet ready!')
            dataset_path = "/share/klab/datasets/optimized_datasets/miniecoset.h5"

        else:
            print('Getting Ecoset ready!')
            dataset_path = (
                hyp['dataset']['dataset_path']
                + hyp['dataset']['name']
                + '_square256_proper_chunks.h5'
            )

        import h5py

        with h5py.File(dataset_path, "r") as f:
            hyp['dataset']['n_classes'] = np.max(f['val']['labels'][()]) + 1

        hyp['dataset']['class_weights'] = None

        transform = get_transform(hyp['dataset']['augment'], hyp)
        transform_val_test = get_transform(hyp['dataset']['augment_val_test'], hyp)

        if 'train' in splits:
            train_data = Ecoset(
                'train',
                dataset_path=dataset_path,
                in_memory=0,
                transform=transform
            )

        if 'val' in splits:
            val_data = Ecoset(
                'val',
                dataset_path=dataset_path,
                in_memory=0,
                transform=transform_val_test
            )

        if 'test' in splits:
            test_data = Ecoset(
                'test',
                dataset_path=dataset_path,
                in_memory=0,
                transform=transform_val_test
            )

        # ---------------------------------
        # DEBUG: limit EcoSet size
        # ---------------------------------
        if hyp.get("ecoset_debug_subset", False):

            debug_size = hyp.get("ecoset_debug_size", 500)

            if 'train' in splits:
                train_data = torch.utils.data.Subset(train_data, range(debug_size))

            if 'val' in splits:
                val_data = torch.utils.data.Subset(val_data, range(min(debug_size, len(val_data))))

            if 'test' in splits:
                test_data = torch.utils.data.Subset(test_data, range(min(debug_size, len(test_data))))

            print(f"⚠ Using EcoSet DEBUG subset of size {debug_size}")


    # ==========================================================
    # IMAGENET (if ever needed)
    # ==========================================================
    elif hyp['dataset']['name'] == 'imagenet':

        from torchvision import datasets
        from helpers.helper_funcs import calculate_class_weights_from_imagefolder

        dataset_path = hyp['dataset']['dataset_path'] + 'imagenet'

        print('Getting Imagenet ready!')

        transform = get_transform(hyp['dataset']['augment'], hyp)
        transform_val_test = get_transform(hyp['dataset']['augment_val_test'], hyp)

        if 'train' in splits:
            train_data = datasets.ImageFolder(
                root=dataset_path + '/train',
                transform=transform
            )
            hyp['dataset']['class_weights'] = calculate_class_weights_from_imagefolder(train_data)

        if 'val' in splits:
            val_data = datasets.ImageFolder(
                root=dataset_path + '/val',
                transform=transform_val_test
            )

        if 'test' in splits:
            test_data = datasets.ImageFolder(
                root=dataset_path + '/val',
                transform=transform_val_test
            )

        hyp['dataset']['n_classes'] = 1000

    else:
        print('Dataset not found!')
        return


    print(dataset_path)
    print(f'Number of classes: {hyp["dataset"]["n_classes"]}')


    # ==========================================================
    # Create DataLoaders
    # ==========================================================
    ra_reps = hyp.get('augmentation', {}).get('ra_reps', 0)
    mixup_alpha = hyp.get('augmentation', {}).get('mixup_alpha', 0.0)
    cutmix_alpha = hyp.get('augmentation', {}).get('cutmix_alpha', 0.0)

    train_sampler = None
    train_collate_fn = None

    if 'train' in splits:
        if ra_reps > 0:
            train_sampler = RepeatedAugmentationSampler(train_data, num_repeats=ra_reps)
            print(f'Using RepeatedAugmentation sampler with {ra_reps} repeats')
        if mixup_alpha > 0.0 or cutmix_alpha > 0.0:
            train_collate_fn = get_mixup_cutmix_collate_fn(mixup_alpha, cutmix_alpha, hyp['dataset']['n_classes'])
            print(f'Using MixUp(alpha={mixup_alpha}) + CutMix(alpha={cutmix_alpha})')

        _num_workers_train = hyp['optimizer']['dataloader']['num_workers_train']
        _train_dl_kwargs = dict(
            batch_size=hyp['optimizer']['batch_size'],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=_num_workers_train,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=train_collate_fn
        )
        if _num_workers_train > 0:
            _train_dl_kwargs['prefetch_factor'] = hyp['optimizer']['dataloader']['prefetch_factor_train']
        train_loader = torch.utils.data.DataLoader(train_data, **_train_dl_kwargs)
    else:
        train_loader = None

    if 'val' in splits:
        _num_workers_vt = hyp['optimizer']['dataloader']['num_workers_val_test']
        _val_dl_kwargs = dict(
            batch_size=hyp['misc']['batch_size_val_test'],
            num_workers=_num_workers_vt,
            pin_memory=False,
            persistent_workers=False
        )
        if _num_workers_vt > 0:
            _val_dl_kwargs['prefetch_factor'] = hyp['optimizer']['dataloader']['prefetch_factor_val_test']
        val_loader = torch.utils.data.DataLoader(val_data, **_val_dl_kwargs)
    else:
        val_loader = None

    if 'test' in splits:
        _num_workers_vt = hyp['optimizer']['dataloader']['num_workers_val_test']
        _test_dl_kwargs = dict(
            batch_size=hyp['misc']['batch_size_val_test'],
            num_workers=_num_workers_vt,
            pin_memory=False,
            persistent_workers=False
        )
        if _num_workers_vt > 0:
            _test_dl_kwargs['prefetch_factor'] = hyp['optimizer']['dataloader']['prefetch_factor_val_test']
        test_loader = torch.utils.data.DataLoader(test_data, **_test_dl_kwargs)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader, hyp

    
def calculate_class_weights_from_h5(labels):
    """
    Calculate class weights for CrossEntropyLoss based on EcoSet labels
    and print the min and max counts per class.

    Args:
        labels (numpy.ndarray): Array of labels for the EcoSet dataset.

    Returns:
        torch.Tensor: Tensor of class weights to use with CrossEntropyLoss.
    """
    # Count occurrences of each class
    class_counts = Counter(labels)

    # Get total number of samples
    total_samples = len(labels)

    # Calculate class weights: inverse proportional to class frequency
    num_classes = len(class_counts)
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]

    # Print min and max counts
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    print(f"Minimum count per class: {min_count}")
    print(f"Maximum count per class: {max_count}")

    # Normalize weights
    class_weights = np.array(class_weights) / sum(class_weights)

    # Convert to a tensor for use in PyTorch
    return torch.tensor(class_weights, dtype=torch.float)

def calculate_class_weights_from_imagefolder(dataset):
    """
    Calculate class weights for CrossEntropyLoss based on the dataset loaded with ImageFolder.

    Args:
        dataset (torchvision.datasets.ImageFolder): Dataset loaded using ImageFolder.

    Returns:
        torch.Tensor: Tensor of class weights to use with CrossEntropyLoss.
    """
    # Get the list of labels for all samples in the dataset
    labels = [sample[1] for sample in dataset.samples]

    # Count occurrences of each class
    class_counts = Counter(labels)

    # Get total number of samples
    total_samples = sum(class_counts.values())

    # Calculate class weights: inverse proportional to class frequency
    num_classes = len(class_counts)
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]

    # Print min and max counts
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    print(f"Minimum count per class: {min_count}")
    print(f"Maximum count per class: {max_count}")

    # Normalize weights (optional, depends on preference)
    class_weights = np.array(class_weights) / sum(class_weights)

    # Convert to a tensor for use in PyTorch
    return torch.tensor(class_weights, dtype=torch.float)
    
##############################
## Transform functions
##############################

def get_transform(aug_str,hyp=None):
    # Returns a transform compose function given the transforms listed in "aug_str"

    transform_list = []
    if 'randomresizedcrop_176' in aug_str:
        transform_list.append(transforms.RandomResizedCrop(176, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True))
    if 'resize_232' in aug_str:
        transform_list.append(transforms.Resize(232, antialias=True))
    if 'resize_224' in aug_str:
        transform_list.append(transforms.Resize(224, antialias=True))
    if 'crop_224' in aug_str:
        transform_list.append(transforms.RandomCrop(224))
    if 'centercrop_224' in aug_str:
        transform_list.append(transforms.CenterCrop(224))
    if 'resize_128' in aug_str:
        transform_list.append(transforms.Resize(128, antialias=True))
    if 'blurring' in aug_str:
        max_kernel_size = 224//8 - 1
        transform_list.append(RandomGaussianBlur(p=0.5, kernel_size=(1,max_kernel_size), sigma=(0.1,max_kernel_size*1./2))) # apply random gaussian blur "p" of time
    if 'hflip' in aug_str:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if 'trivialaug' in aug_str:
        transform_list.append(transforms.TrivialAugmentWide())
    if 'randaug' in aug_str:
        transform_list.append(transforms.RandAugment())
    if hyp['dataset']['name'] == 'imagenet':
        transform_list.append(transforms.ToTensor())
    else:
        transform_list.append(transforms.ConvertImageDtype(torch.float))
    if 'imagenet_normalize' in aug_str:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if 'normalize' in aug_str:
        # transform_list.append(transforms.Lambda(lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1))  # Scale to [-1, 1]
        transform_list.append(transforms.Lambda(lambda x: 2*x - 1))  # to_float, etc. makes images go between [0,1] - the other thing doesn't work as well!
    if 'random_erasing' in aug_str:
        transform_list.append(transforms.RandomErasing(p=0.1))

    transform = transforms.Compose(transform_list)
    
    return transform

class RandomGaussianBlur(transforms.GaussianBlur):
    def __init__(self, p, kernel_size, sigma=None):
        super().__init__(kernel_size, sigma)
        self.prob = p

    def __call__(self, img):
        if random.random() < self.prob:  # apply blur if...
            return super().__call__(img)
        return img


class RepeatedAugmentationSampler(torch.utils.data.Sampler):
    """Sampler that repeats each sample multiple times for repeated augmentation (RA).
    Each index appears num_repeats times per epoch, with different augmentations each time."""

    def __init__(self, dataset, num_repeats=4, shuffle=True):
        self.dataset = dataset
        self.num_repeats = num_repeats
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = len(dataset) * num_repeats

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        # Repeat each index num_repeats times
        indices = [idx for idx in indices for _ in range(self.num_repeats)]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_mixup_cutmix_collate_fn(mixup_alpha, cutmix_alpha, num_classes):
    """Returns a collate_fn that applies MixUp and/or CutMix to each batch."""
    from torchvision.transforms import v2

    transforms_list = []
    if mixup_alpha > 0.0:
        transforms_list.append(v2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0.0:
        transforms_list.append(v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))
    mixupcutmix = v2.RandomChoice(transforms_list)

    def collate_fn(batch):
        return mixupcutmix(*torch.utils.data.dataloader.default_collate(batch))

    return collate_fn

    
##############################
## LR scheduler
##############################
    
class LinearFitScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, factor=1./2, min_lr=1e-8, min_percent_change=1.0, mode='min', patience=5, last_epoch=-1, verbose=False):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            num_epochs (int): Number of epochs to use for the linear fit.
            factor (float): Factor by which the learning rate will be reduced. Default: 0.1.
            min_lr (float): Minimum learning rate. Default: 1e-6.
            min_percent_change (float): Minimum absolute percentage change in the metric to not trigger a reduction. Default: 1.0.
            mode (str): One of 'min' or 'max'. 'min' will reduce the LR if the metric has not decreased by min_percent_change,
                        'max' will reduce the LR if the metric has not increased by min_percent_change. Default: 'min'.
            patience (int): Number of epochs with no improvement after which learning rate will be reduced. Default: 0.
            last_epoch (int): The index of the last epoch. Default: -1.
            verbose (bool): If True, prints a message to stdout for each update. Default: False.
        """
        self.num_epochs = num_epochs
        self.factor = factor
        self.min_lr = min_lr
        self.min_percent_change = min_percent_change
        self.mode = mode
        self.patience = patience
        self.num_bad_epochs = 0  # Track the number of epochs without improvement
        self.verbose = verbose
        self.metric_history = []
        super(LinearFitScheduler, self).__init__(optimizer, last_epoch=last_epoch)

    def step(self, metric=None):
        """
        Step should be called after each epoch. Can be called without 'metric' during initialization.
        
        Args:
            metric (float, optional): Current epoch's metric. Default is None.
        """
        # Increment the last_epoch attribute from the base class
        self.last_epoch += 1
        
        if metric is not None:
            # Update metric history
            self.metric_history.append(metric)
            
            # Only perform the check if we have enough history
            if len(self.metric_history) >= self.num_epochs:
                # Perform linear fit
                epochs = np.arange(self.num_epochs).reshape(-1, 1)
                metrics = np.array(self.metric_history[-self.num_epochs:]).reshape(-1, 1)
                
                reg = LinearRegression().fit(epochs, metrics)
                slope = reg.coef_[0, 0]
                intercept = reg.intercept_[0]
                
                # Calculate the predicted metrics
                predicted_start = intercept
                predicted_end = slope * (self.num_epochs - 1) + intercept
                
                # Calculate percent change based on the magnitude of the start value
                if predicted_start != 0:
                    percent_change = 100 * (predicted_end - predicted_start) / abs(predicted_start)
                else:
                    percent_change = float('inf')  # Avoid division by zero
                if self.verbose:
                    print(f"Percent_change in metric: {percent_change:.2f}%")
                    
                # Determine if we should adjust the learning rate based on the mode and percent change
                if self.mode == 'min' and percent_change > -self.min_percent_change:
                    self.num_bad_epochs += 1
                elif self.mode == 'max' and percent_change < self.min_percent_change:
                    self.num_bad_epochs += 1
                else:
                    self.num_bad_epochs = 0  # Reset counter if improvement is observed
                
                # Check if we have hit the patience threshold
                if self.num_bad_epochs > self.patience:
                    self.reduce_lr(percent_change)
                    self.metric_history = []  # Reset history after reducing LR
                    self.num_bad_epochs = 0  # Reset bad epoch count after reducing LR

    def reduce_lr(self, percent_change):
        """Reduce the learning rate according to the factor and min_lr constraints and print verbose message."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = max(param_group['lr'] * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            if self.verbose:
                print(f"Reducing learning rate of group {i} to {new_lr:.4e}. Percent change: {percent_change:.2f}%. Patience exceeded.")
    
##############################
## Logging functions
##############################
    
def create_folders_logging(net_name, create_folders=True):

    print('Accessing log folders...')

    log_folder = 'logs/perf_logs'
    net_folder = 'logs/net_params'

    isExist = os.path.exists(log_folder)
    if not isExist and create_folders:
        os.makedirs(log_folder)
        print('Log folder is created!')
    isExist = os.path.exists(net_folder)
    if not isExist and create_folders:
        os.makedirs(net_folder)
        print('Net folder is created!')

    log_folder_name = log_folder+f'/{net_name}'
    net_folder_name = net_folder+f'/{net_name}'

    isExist = os.path.exists(log_folder_name)
    if not isExist and create_folders:
        os.makedirs(log_folder_name)
        print('Specific log folder is created!')
    isExist = os.path.exists(net_folder_name)
    if not isExist and create_folders:
        os.makedirs(net_folder_name)
        print('Specific net folder is created!')

    return log_folder_name, net_folder_name 


def compute_first_signal(bottlenecks, skip_connections):
    """Compute earliest possible signal arrival per area based on network connections.

    With bio_unroll, the feedforward path is:
        Retina(t=0) -> LGN(t=1) -> V1(t=2) -> V2(t=3) -> V3(t=4) -> V4(t=5) -> LOC(t=6)

    Skip connections (e.g. V1->V4_skip) allow signals to arrive earlier.
    The skip signal arrives one timestep after the source area first has activity.
    Changes propagate through the feedforward chain (e.g. V4 earlier -> LOC earlier).
    """
    # Feedforward chain: each area receives from the previous one
    ff_chain = [
        ("Retina", "LGN"),
        ("LGN", "V1"),
        ("V1", "V2"),
        ("V2", "V3"),
        ("V3", "V4"),
        ("V4", "LOC"),
    ]

    first_signal = {
        "Retina": 0,
        "LGN": 1,
        "V1": 2,
        "V2": 3,
        "V3": 4,
        "V4": 5,
        "LOC": 6,
    }

    if not skip_connections:
        return first_signal

    # Parse skip connections from bottleneck config (explicit "_skip" keys)
    skip_edges = []
    for edge in bottlenecks:
        if "->" in edge and edge.endswith("_skip"):
            base_edge = edge[:-5]  # strip "_skip"
            src, dst = base_edge.split("->", 1)
            skip_edges.append((src, dst))

    # If skip_connections=1 but no explicit skip edges in bottlenecks,
    # fall back to the hardcoded architecture skips: V1->V4 (bottom-up) and V4->V1 (top-down)
    if not skip_edges:
        skip_edges = [("V1", "V4"), ("V4", "V1")]

    # Combine feedforward + skip edges, then relax until stable
    all_edges = ff_chain + skip_edges

    changed = True
    while changed:
        changed = False
        for src, dst in all_edges:
            if src not in first_signal or dst not in first_signal:
                continue
            new_timing = first_signal[src] + 1
            if new_timing < first_signal[dst]:
                first_signal[dst] = new_timing
                changed = True

    return first_signal