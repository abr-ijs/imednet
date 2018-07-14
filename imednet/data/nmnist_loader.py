"""
Load the n-MNIST (noisy MNIST) dataset.

The n-MNIST dataset (short for noisy MNIST) is created using the MNIST dataset
of handwritten digits by adding -

(1) additive white gaussian noise,
(2) motion blur and
(3) a combination of additive white gaussian noise and reduced contrast to the
MNIST dataset.
"""
from __future__ import print_function

import os
import errno
import scipy.io as sio
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.datasets.utils import download_url


class NMNIST(MNIST):
    """`n-MNIST <http://www.csc.lsu.edu/~saikat/n-mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``mnist-with-awgn.mat``,
            ``mnist-with-motion-blur.mat`` and
            ``mnist-with-reduced-contrast-and-awgn.mat`` exist.
        train (bool, optional): If True, loads training data, otherwise loads
            test data.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        dataset (string, optional): Select the particular n-MNIST dataset to load:
            "awgn", "motion-blur", "contrast-and-awgn" or "all".
    """
    urls = {
            'awgn': 'http://www.csc.lsu.edu/~saikat/n-mnist/data/mnist-with-awgn.gz',
            'motion-blur': 'http://www.csc.lsu.edu/~saikat/n-mnist/data/mnist-with-motion-blur.gz',
            'contrast-and-awgn': 'http://www.csc.lsu.edu/~saikat/n-mnist/data/mnist-with-reduced-contrast-and-awgn.gz',
    }
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, dataset='awgn'):
        self.dataset = dataset

        if self.dataset != 'all':
            self.urls = {self.dataset: self.urls[self.dataset]}

        self.gzip_files = []
        self.mat_files = []
        for _, url in self.urls.items():
            self.gzip_files.append(os.path.basename(url))
            self.mat_files.append(os.path.splitext(os.path.basename(url))[0] + '.mat')

        self.training_file = self.dataset + '-training.pt'
        self.test_file = self.dataset + '-test.pt'

        super(NMNIST, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def _check_gzips_exists(self):
        for gzip_file in self.gzip_files:
            if not os.path.exists(os.path.join(self.root, self.raw_folder, gzip_file)):
                return False
        return True

    def _check_mats_exists(self):
        for mat_file in self.mat_files:
            if not os.path.exists(os.path.join(self.root, self.raw_folder, mat_file)):
                return False
        return True

    def download(self):
        """Download the n-MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if not self._check_mats_exists():
            for _, url in self.urls.items():
                filename = url.rpartition('/')[2]
                file_path = os.path.join(self.root, self.raw_folder, filename)
                if not self._check_gzips_exists():
                    download_url(url, root=os.path.join(self.root, self.raw_folder),
                                 filename=filename, md5=None)
                with open(file_path.replace('.gz', '.mat'), 'wb') as out_f:
                    tar = tarfile.open(file_path, 'r:gz')
                    zip_f = tar.extractfile(os.path.basename(file_path.replace('.gz', '.mat')))
                    out_f.write(zip_f.read())
                    os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        def read_images(mat_data, split):
            length = mat_data[split].shape[0]
            num_rows = np.uint8(np.sqrt(mat_data[split].shape[1]))
            num_cols = num_rows
            return torch.from_numpy(mat_data[split]).view(length, num_rows, num_cols)

        def read_labels(mat_data, split):
            length = mat_data[split].shape[0]
            labels = np.asarray([np.where(r==1)[0][0] for r in mat_data[split]])
            return torch.from_numpy(labels).view(length).long()

        data = sio.loadmat(os.path.join(self.root, self.raw_folder, self.mat_files[0]))
        if len(self.mat_files) > 1:
            for mat_file in self.mat_files[1:]:
                mat_data = sio.loadmat(os.path.join(self.root, self.raw_folder, mat_file))
                data['train_x'] = np.concatenate((data['train_x'], mat_data['train_x']), axis=0)
                data['train_y'] = np.concatenate((data['train_y'], mat_data['train_y']), axis=0)
                data['test_x'] = np.concatenate((data['test_x'], mat_data['test_x']), axis=0)
                data['test_y'] = np.concatenate((data['test_y'], mat_data['test_y']), axis=0)

        training_set = (
            read_images(data, 'train_x'),
            read_labels(data, 'train_y')
        )
        test_set = (
            read_images(data, 'test_x'),
            read_labels(data, 'test_y')
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')
