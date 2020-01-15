import os
import pickle
import sys

import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator

from sklearn.utils import shuffle


mean = {
    'cifar10':  (0.4914 * 255., 0.4822 *255., 0.4465 *255.),
    'cifar100': (0.5071 * 255., 0.4867 *255., 0.4408 *255.),
}

std = {
    'cifar10':  (0.2023 *255., 0.1994 *255., 0.2010 *255.),
    'cifar100': (0.2675 *255., 0.2565 *255., 0.2761 *255.),
}


class BasePipe(Pipeline):
    def __init__(self,
                 batch_size,
                 num_threads,
                 device_id,
                 mean,
                 std,
                 input_iterator=None,
                 transform = True,
                 ):
        super(BasePipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        assert input_iterator is not None
        self.transform = transform
        self.iterator = iter(input_iterator)

        self.input_image = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.input_index = ops.ExternalSource()
        self.input_pos_index = ops.ExternalSource()
        self.input_neg_index = ops.ExternalSource()

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=mean,
                                            std=std)

    def iter_setup(self):
        (images, labels, index, pos_idx, neg_idx) = self.iterator.next()
        self.feed_input(self.jpegs, images, layout=types.NHWC)  # can only in HWC order
        self.feed_input(self.labels, labels)
        self.feed_input(self.index, index)
        self.feed_input(self.pos_index, pos_idx)
        self.feed_input(self.neg_index, neg_idx)

    def input_transform(self, images):
        return self.cmnp(images)

    def define_graph(self):
        self.jpegs = self.input_image()
        self.labels = self.input_label()
        self.index = self.input_index()
        self.pos_index = self.input_pos_index()
        self.neg_index = self.input_neg_index()


        if self.transform:
            output = self.input_transform(self.jpegs.gpu())
        else:
            output = self.jpegs.gpu()
        return [output, self.labels, self.index, self.pos_index, self.neg_index]


class HybridTrainPipeCIFAR(BasePipe):
    def __init__( self, crop=32, **kwargs):
        super(HybridTrainPipeCIFAR, self).__init__(**kwargs)

        # Augmentation operations
        self.uniform = ops.Uniform(range=(0., 1.))
        self.coin = ops.CoinFlip(probability=0.5)

        self.pad = ops.Paste(device="gpu", ratio=1.25, fill_value=0)
        self.crop = ops.Crop(device="gpu", crop_h=crop, crop_w=crop)

    def input_transform(self, images):
        output = self.pad(images)
        output = self.crop(output, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        output = self.cmnp(output, mirror=self.coin())
        return output


class HybridValPipeCIFAR(BasePipe):
    def __init__(self, **kwargs):
        super(HybridValPipeCIFAR, self).__init__(**kwargs)


class CifarInputIterator:
    def __init__(self, batch_size, file_name, root, percent=1.0, k=None, mode=None):
        self.root = root
        self.batch_size = batch_size
        self.k = k
        self.mode = mode

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
        self.index = np.array(range(len(self.data)))
        self.targets = np.vstack(self.targets).astype(np.uint8)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
        num_classes = len(np.unique(self.targets))
        num_samples = len(self.data)
        label = self.targets

        # >> Copy from https://github.com/HobbitLong/RepDistiller/blob/master/dataset/cifar100.py
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[int(label[i])].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)
        # <<

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
        idx = []
        pos_idxs = []
        neg_idxs = []
        for _ in range(self.batch_size):
            if self.i % self.n == 0:
                self.data, self.targets, self.index = shuffle(self.data, self.targets, self.index, random_state=0)
                # return (batch, labels)
            img, label, index = self.data[self.i], self.targets[self.i], self.index[self.i]

            batch.append(img)
            labels.append(label)
            idx.append(index)

            # >> Copy from https://github.com/HobbitLong/RepDistiller/blob/master/dataset/cifar100.py
            if self.k is not None and self.k!=0:
                if self.mode == 'exact':
                    pos = index
                elif self.mode == 'relax':
                    pos = np.random.choice(self.cls_positive[int(label)], 1)
                    pos = pos[0]
                else:
                    raise NotImplementedError(self.mode)
                replace = True if self.k > len(self.cls_negative[int(label)]) else False
                neg = np.random.choice(self.cls_negative[int(label)], self.k, replace=replace)
                pos_idxs.append(pos)
                neg_idxs.append(neg)
            # <<
            else:
                pos_idxs.append(np.array(-1))
                neg_idxs.append(np.array(-1))



            self.i = (self.i + 1) % self.n

        return batch, labels, idx, pos_idxs, neg_idxs

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
        version="cifar10",
        transform=True,
        k = None,
        mode = None,
):
    assert version == "cifar100", f"Cifar10 is not yet available"
    InputIterator = Cifar100InputIterator
    # InputIterator = Cifar10InputIterator if version == "cifar10" else Cifar100InputIterator

    assert type_ in ["train", "test", "sub_train", "sub_val"]

    if "train" in type_:
        input_iterator = InputIterator(batch_size, k = k, mode = mode, file_name=type_, root=image_dir)
        pipe_train = HybridTrainPipeCIFAR(crop=32, 
                batch_size=batch_size, num_threads=num_threads, device_id=0, mean=mean[version], 
                std=std[version], input_iterator=input_iterator, transform=transform)
        pipe_train.build()
        dali_iter_train = DALIGenericIterator(pipe_train, output_map=["data", "label", "index", "positive_index", "negative_index"], size=len(input_iterator), 
                fill_last_batch=False, last_batch_padded=True)
        return dali_iter_train
    else:
        input_iterator = InputIterator(batch_size, file_name=type_, root=image_dir)
        pipe_test = HybridValPipeCIFAR(
                batch_size=batch_size, num_threads=num_threads, device_id=0, mean=mean[version], 
                std=std[version], input_iterator=input_iterator, transform=transform)
        pipe_test.build()
        dali_iter_test = DALIGenericIterator(pipe_test, output_map=["data", "label", "index", "positive_index", "negative_index"], size=len(input_iterator), 
                fill_last_batch=False, last_batch_padded=True)
        return dali_iter_test
