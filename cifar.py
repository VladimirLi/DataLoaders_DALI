import os
import pickle
import sys

import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from sklearn.utils import shuffle


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
    def __init__(self, batch_size, file_name, root):
        self.root = root
        self.batch_size = batch_size

        self.data = []
        self.targets = []
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
        self.targets = np.vstack(self.targets).astype(np.uint8)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        self.data, self.targets = shuffle(self.data, self.targets, random_state=0)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            if self.i % self.n == 0:
                self.data, self.targets = shuffle(self.data, self.targets, random_state=0)
                # return (batch, labels)
            img, label = self.data[self.i], self.targets[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    next = __next__


# class Cifar10InputIterator(CifarInputIterator):
#     base_folder = 'cifar-10-batches-py'
#     train_list = [
#         ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
#         ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
#         ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
#         ['data_batch_4', '634d18415352ddfa80567beed471001a'],
#         ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
#     ]
#
#     test_list = [
#         ['test_batch', '40351d587109b95175f43aff81a1287e'],
#     ]
#
#     def __init__(self, *args, **kwargs):
#         super(Cifar10InputIterator, self).__init__(*args, **kwargs)


class Cifar100InputIterator(CifarInputIterator):
    base_folder = 'cifar-100-python'
    train_list = [['train', '']]
    test_list = [['test', '']]

    def __init__(self, *args, **kwargs):
        super(Cifar100InputIterator, self).__init__(*args, **kwargs)


def get_cifar_iter_dali(
        type_,
        image_dir,
        batch_size,
        num_threads,
        version="cifar10"
):
    assert version == "cifar100", f"Cifar10 is not yet available"
    InputIterator = Cifar100InputIterator
    # InputIterator = Cifar10InputIterator if version == "cifar10" else Cifar100InputIterator

    assert type_ in ["train", "test", "sub_train", "sub_val"]

    if "train" in type_:
        input_iterator = InputIterator(batch_size, file_name=type_, root=image_dir)
        pipe_train = HybridTrainPipe_CIFAR(batch_size=batch_size, num_threads=num_threads,
                                           device_id=0, crop=32,
                                           input_iterator=input_iterator)
        pipe_train.build()
        dali_iter_train = DALIClassificationIterator(pipe_train, size=len(input_iterator), fill_last_batch=False,
                                                     last_batch_padded=True)
        return dali_iter_train
    else:
        input_iterator = InputIterator(batch_size, file_name=type_, root=image_dir)
        pipe_test = HybridValPipe_CIFAR(batch_size=batch_size, num_threads=num_threads,
                                        device_id=0, input_iterator=input_iterator)
        pipe_test.build()
        dali_iter_test = DALIClassificationIterator(pipe_test, size=len(input_iterator), fill_last_batch=False,
                                                    last_batch_padded=True)
        return dali_iter_test
