import os
import sys
import time
import pickle
import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from sklearn.utils import shuffle
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator


class HybridTrainPipe_CIFAR(Pipeline):
    def __init__(self,
                 batch_size,
                 num_threads,
                 device_id,
                 crop=32,
                 input_iterator=None):
        super(HybridTrainPipe_CIFAR, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        assert input_iterator is not None
        self.iterator = iter(input_iterator)

        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()

        # Augmentation operations
        self.uniform = ops.Uniform(range=(0., 1.))
        self.coin = ops.CoinFlip(probability=0.5)

        self.pad = ops.Paste(device="gpu", ratio=1.25, fill_value=0)
        self.crop = ops.Crop(device="gpu", crop_h=crop, crop_w=crop)
        self.cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.FLOAT, output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.],
                                            std=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.])

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)

    def define_graph(self):
        rng = self.coin()
        self.jpegs = self.input()
        self.labels = self.input_label()
        output = self.jpegs
        output = self.pad(output.gpu())
        output = self.crop(output, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        output = self.cmnp(output, mirror=rng)
        return [output, self.labels]


class HybridValPipe_CIFAR(Pipeline):
    def __init__(self,
                 batch_size,
                 num_threads,
                 device_id,
                 input_iterator = None):

        super(HybridValPipe_CIFAR, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        assert input_iterator is not None
        self.iterator = iter(input_iterator)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.],
                                            std=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
                                            )

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images, layout=types.NHWC)  # can only in HWC order
        self.feed_input(self.labels, labels)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        output = self.jpegs
        output = self.cmnp(output.gpu())
        return [output, self.labels]


class CifarInputIterator:
    def __init__(self, batch_size, type='train', root='/userhome/memory_data/cifar10', indices=None):
        self.root = root
        self.batch_size = batch_size
        self.train = (type == 'train')
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.targets = np.vstack(self.targets)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if indices is not None:
            self.data = self.data[indices]
            self.targets = self.targets[indices]

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        self.data, self.targets = shuffle(self.data, self.targets, random_state=0)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            if self.train and self.i % self.n == 0:
                self.data, self.targets = shuffle(self.data, self.targets, random_state=0)
            img, label = self.data[self.i], self.targets[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    next = __next__


class Cifar10InputIterator(CifarInputIterator):
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, *args, **kwargs):
        super(Cifar10InputIterator, self).__init__(*args, **kwargs)


class Cifar100InputIterator(CifarInputIterator):
    base_folder = 'cifar-100-python'
    train_list = [['train', '']]
    test_list = [['test', '']]

    def __init__(self, *args, **kwargs):
        super(Cifar100InputIterator, self).__init__(*args, **kwargs)


def get_cifar_iter_dali(
        type,
        image_dir,
        batch_size,
        num_threads,
        val_ratio=0,
        local_rank=0,
        version="cifar10"
):

    InputIterator = Cifar10InputIterator if version == "cifar10" else Cifar100InputIterator
    if type == 'train':
        if val_ratio != 0:
            all_idxs = np.random.RandomState(seed=42).permutation(50000)
            num_tr_samples = int(50000*(1-val_ratio))
            tr_idxs = all_idxs[:num_tr_samples]
            val_idxs = all_idxs[num_tr_samples:]
        else:
            tr_idxs = range(50000)
            val_idxs = []

        pip_train = HybridTrainPipe_CIFAR(batch_size=batch_size, num_threads=num_threads,
                                          device_id=local_rank, crop=32,
                                          input_iterator=InputIterator(batch_size, root=image_dir, indices=tr_idxs))
        pip_train.build()
        dali_iter_train = DALIClassificationIterator(pip_train, size=len(tr_idxs))

        dali_iter_validation = None
        if len(val_idxs):
            pip_val = HybridValPipe_CIFAR(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                          input_iterator=InputIterator(batch_size, root=image_dir, indices=val_idxs))
            pip_val.build()
            dali_iter_validation = DALIClassificationIterator(pip_val, size=len(val_idxs))

        return dali_iter_train, dali_iter_validation

    elif type == 'test':
        pip_test = HybridValPipe_CIFAR(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                       input_iterator=InputIterator(batch_size, type="test", root=image_dir))
        pip_test.build()
        dali_iter_test = DALIClassificationIterator(pip_test, size=10000)
        return dali_iter_test
    else:
        exit(f"type={type} is not valid")


if __name__ == '__main__':
    from distill.plot_batch import plot_batch

    train_loader, val_loader = get_cifar_iter_dali(type='train',
                                                   image_dir='/home/vladimir/workspace/distilation_comparison/data',
                                                   batch_size=256, num_threads=4, version="cifar100", local_rank=2,
                                                   val_ratio=.2)
    print('start iterate')
    n_tr_samples = 0
    start = time.time()
    for i, data in enumerate(train_loader):
        images = data[0]["data"].cuda(non_blocking=True)
        labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        n_tr_samples+=labels.size(0)
    end = time.time()
    print(f"Number of training samples: {n_tr_samples}")
    print('end iterate')
    print('dali iterate time: %fs' % (end - start))

    print('start iterate')
    n_val_samples = 0
    start = time.time()
    for i, data in enumerate(val_loader):
        images = data[0]["data"].cuda(non_blocking=True)
        labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        n_val_samples+=labels.size(0)
    end = time.time()
    print(f"number of validation samples: {n_val_samples}")
    print('end iterate')
    print('dali iterate time: %fs' % (end - start))
