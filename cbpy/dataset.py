import gzip
import os
import random
import sys
import tarfile
from urllib.request import urlretrieve

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd


__all__ = ['create_xor_dataloader',
           'create_toy_dataloader',
           'create_cifar10_dataloader']


def create_xor_dataloader(delta=.1):
    """
    create trainloader for the XOR problem.

    :param delta: target value for (0, 0) and (1, 1)
    :return: trainloader
    """
    data = torch.FloatTensor([[1., 1.],
                         [1., 0.],
                         [0., 1.],
                         [0., 0.]])
    target = torch.FloatTensor([[delta], [1-delta], [1-delta], [delta]])
    trainloader = DataLoader(TensorDataset(data, target), 4)
    return trainloader


def create_toy_dataloader(toy_name, batch_size=None, **params):
    """
    create the pytorch-style trainloader for toy datasets.

    :param toy_name: dataset name
    :param params:
        shuffle: shuffle the order of the samples
        onehot: use onehot to code the targets or labels of the samples
        norm: use MinMaxScaler() to normalize the train data into [0,1]
    :return:
        trainloader
    """
    if toy_name in ['iris', 'wine', 'digits']:
        data, target = sk_toy_dataset(name=toy_name, **params)
    elif toy_name in ['blood', 'titanic', 'twonorm', 'breast']:
        data, target = uci_and_delve_toy_dataset(name=toy_name, **params)
    else:
        raise ValueError(f'Unknown dataset name {toy_name}.')

    if batch_size is None:
        batch_size = data.shape[0]
    trainloader = DataLoader(TensorDataset(data, target), batch_size)
    return trainloader


def sk_toy_dataset(name, shuffle=True, onehot=False, norm=True):
    '''
    three toy datasets (classification) in sklearn (https://scikit-learn.org/).

    :param name: dataset name in sklearn
    :param shuffle: shuffle the order of the samples
    :param onehot: use onehot to code the targets or labels of the samples
    :param norm: use MinMaxScaler() to normalize the train data into [0,1]
    :return:
        data: training data
        target: labels for data
    '''
    if name == 'iris':
        dset = datasets.load_iris()
    elif name == 'wine':
        dset = datasets.load_wine()
    elif name == 'digits':
        dset = datasets.load_digits()
    else:
        raise ValueError(f'Unknown dataset name {name}.'
                         'Current available name: "iris", "wine",'
                         '"digits".')

    data, target = _my_preprocess(dset.data,
                                  dset.target,
                                  onehot=onehot,
                                  shuffle=shuffle,
                                  norm=norm)
    return data, target


def uci_and_delve_toy_dataset(name, shuffle=True, onehot=False, norm=True):
    """
    four toy datasets (classification) in UCI repository (http://archive.ics.uci.edu/ml/)
    or DELVE repository (http://www.cs.toronto.edu/~delve/data/).

    :param name: dataset name
    :param shuffle: shuffle the order of the samples
    :param onehot: use onehot to code the targets or labels of the samples
    :param norm: use MinMaxScaler() to normalize the train data into [0,1]
    :return:
        data: training data
        target: labels for data
    """
    if name == 'blood':
        data, target = _blood_dataset()
    elif name == 'breast':
        data, target = _breast_dataset()
    elif name == 'titanic':
        data, target = _titanic_dataset()
    elif name == 'twonorm':
        data, target = _twonorm_dataset()
    else:
        raise ValueError(f'Unknown name {name}.')
    data, target = _my_preprocess(data,
                                  target,
                                  onehot=onehot,
                                  shuffle=shuffle,
                                  norm=norm)
    return data, target


def unzip_gz(file_name):
    """ unzip .gz file """
    f_name = file_name.replace(".gz", "")
    g_file = gzip.GzipFile(file_name, mode='rb')
    with open(f_name, "wb") as f:
        f.write(g_file.read())
    g_file.close()


def _blood_dataset():
    """ blood dataset """
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
    cache_dir = sys.modules['cbpy'].__path__[0] + '/data/'

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    data_file = os.path.basename(url)
    full_path = cache_dir + data_file
    try:
        dset = pd.read_csv(full_path).to_numpy()
    except:
        urlretrieve(url, cache_dir + data_file)
        dset = pd.read_csv(full_path).to_numpy()
    data = dset[:, :4]
    target = dset[:, 4].astype('int')
    return data, target


def _breast_dataset():
    """ breast cancer dataset """
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    cache_dir = sys.modules['cbpy'].__path__[0] + '/data/'

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    data_file = os.path.basename(url)
    full_path = cache_dir + data_file
    try:
        dset = pd.read_csv(full_path, header=None)
    except:
        urlretrieve(url, cache_dir + data_file)
        dset = pd.read_csv(full_path, header=None)

    use_index = list(range(1, 11))
    del use_index[5] # missing value
    dset = dset.iloc[:,use_index].to_numpy()
    data = dset[:, :8]
    target = dset[:, 8].astype('int')
    target[target == 2] = 0
    target[target == 4] = 1
    return data, target


def _titanic_dataset():
    """ titanic dataset """
    url = 'ftp://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/titanic.tar.gz'
    cache_dir = sys.modules['cbpy'].__path__[0] + '/data/'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    data_path = cache_dir + 'titanic/Source/titanic.dat'
    full_path = cache_dir + os.path.basename(url)
    try:
        dset = pd.read_csv(data_path, header=None, sep='\s+').to_numpy()
    except:
        urlretrieve(url, full_path)
        with tarfile.open(full_path) as t:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(t, path=cache_dir)
        dset = pd.read_csv(data_path, header=None, sep='\s+').to_numpy()
    data = dset[:, :3]
    target = dset[:, 3].astype('int')
    return data, target


def _twonorm_dataset():
    """ twonorm dataset """
    url = 'ftp://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/twonorm.tar.gz'
    cache_dir = sys.modules['cbpy'].__path__[0] + '/data/'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    data_path = cache_dir + 'twonorm/Dataset.data'
    full_path = cache_dir + os.path.basename(url)
    try:
        dset = pd.read_csv(data_path, header=None, sep='\s+').to_numpy()
    except:
        urlretrieve(url, full_path)
        with tarfile.open(full_path) as t:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(t, path=cache_dir)
        unzip_gz(data_path + '.gz')
        dset = pd.read_csv(data_path, header=None, sep='\s+').to_numpy()
    data = dset[:, :20]
    target = dset[:, 20].astype('int')
    return data, target


def _my_preprocess(data, target, shuffle=True, onehot=False, norm=True):
    """ preprocess the training samples and corresponding targets.

    :param data: training samples
    :param target: corresponding targets to samples
    :param shuffle: shuffle the order of the samples
    :param onehot: use onehot to code the targets or labels of the samples
    :param norm: use MinMaxScaler() to normalize the train data into [0,1]
    :return:
        data and target after preprocessing
    """
    if onehot:
        ohe = OneHotEncoder()
        target = ohe.fit_transform(target.reshape(-1, 1)).toarray()

    if shuffle:
        index = list(range(data.shape[0]))
        random.shuffle(index)
        data = data[index]
        target = target[index]

    if norm:
        scale = MinMaxScaler()
        data = scale.fit_transform(data)

    data = torch.Tensor(data)
    if onehot:
        target = torch.Tensor(target)
    else:
        target = torch.LongTensor(target)
    return data, target


def get_kfold_data(k, i, X, y):
    """ get the data of the K-fold cross validation.

    :param k: number of K-fold
    :param i: i th fold
    :param X: training samples
    :param y: targets
    :return:
        X_train: training samples
        y_train: training targets
        X_valid: validated samples
        y_valid: validated targets
    """
    assert k > 1 and i < k

    fold_size = X.shape[0] // k
    val_start = i * fold_size
    if i < k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = torch.cat((X[0:val_start], X[val_end:]), dim=0)
        y_train = torch.cat((y[0:val_start], y[val_end:]), dim=0)
    else:
        X_valid, y_valid = X[val_start:], y[val_start:]
        X_train = X[0:val_start]
        y_train = y[0:val_start]

    return X_train, y_train, X_valid, y_valid


def create_cifar10_dataloader(train_batch_size, test_batch_size,
                              num_workers=0, download=False):
    """ create the trainloader and testloader for cifar10 dataset.

    :param train_batch_size: batchsize for training samples
    :param test_batch_size: batchsize for testing samples
    :param num_workers: how many subprocesses to use for data loading
    :param download: download the dataset or not
    :return: trainloader and testloader
    """
    transform = transforms.Compose([transforms.ToTensor()])
    path = sys.modules['cbpy'].__path__[0] + '/data/'

    try:
        trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                                download=download, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                               download=download, transform=transform)
    except:
        download = True
        trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                                download=download, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                               download=download, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=False, num_workers=num_workers)
    return trainloader, testloader



