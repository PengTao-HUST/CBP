from copy import deepcopy
import time
import logging
import os

import torch
from tqdm import tqdm
import numpy as np

from . import chaos_optim


__all__ = ['train_with_chaos',
           'evaluate',
           'evaluate_per_class',
           'debug_chaos']


def train_with_chaos(model,
                     trainloader,
                     testloader,
                     loss_func,
                     zs,
                     cbp_epoch=1000,
                     max_epoch=None,
                     cbp_lr=0.1,
                     bp_lr=None,
                     record_weight=False,
                     record_first_output=False,
                     best_loss=1e20,
                     beta=0.999,
                     I0=0.65,
                     onehot=False,
                     logdir='./results/',
                     record_acc=None,
                     save_best_loss_model=False,
                     save_best_acc_model=False,
                     flatten_input=False,
                     whole_weight=False,
                     bp_momentum=0,
                     bp_adam=False,
                     gpu=False,
                     clip_cbp_grad=False,
                     clip_cbp_value=0.01,
                     **kws
                     ):
    """ The main training function used in CBP.

    :param model: neural network model
    :param trainloader: dataloader for training samples
    :param testloader: dataloader for testing samples
    :param loss_func: loss function
    :param zs: initial chaotic intensities for each layer
    :param cbp_epoch: number of epoch trained with CBP
    :param max_epoch: maximal epoch in the whole training
    :param cbp_lr: learning rate for CBP
    :param bp_lr: learning rate for BP
    :param record_weight: record the weight of the model during the training
    :param record_first_output: record the output of the first sample
    :param best_loss: minimal loss duiring the training
    :param beta: annealing constant
    :param I0: constant
    :param onehot: use onehot to code the targets or labels of the samples
    :param logdir: the directory where the results will be saved
    :param record_acc: record the time required to reach the accuracy
    :param save_best_loss_model: save the model with the minimal loss
    :param save_best_acc_model: save the model with the maximal accuracy
    :param flatten_input: flatten the features of the sample
    :param whole_weight: record the whole weight or not
    :param bp_momentum: the size of the momentum in BP
    :param bp_adam: whether use adam in BP
    :param gpu: use GPU
    :param clip_cbp_grad: whether clip the gradient of loss_chaos
    :param clip_cbp_value: the cutoff value for the gradient of loss_chaos
    :return:
        tl_list: loss list
        ta_list: accuray list
        ws_list: weight list
        os_list: output list
    """
    # make log directory
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # define logger
    logfile = logdir + "train.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(logfile)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    # whether use GPU
    if gpu:
        if torch.cuda.is_available():
            use_gpu = True
        else:
            logger.warning('No available GPU, use CPU instead.')
            use_gpu = False
    else:
        use_gpu = False

    if use_gpu:
        device = torch.device("cuda:0")
        model = model.to(device)
    else:
        device = torch.device('cpu')

    # record accuracy
    if record_acc is not None:
        _record_acc = True
        if isinstance(record_acc, float):
            n_record = 1
            record_acc_time = [True]
            acc_time = [None]
            record_acc = [record_acc]
        elif isinstance(record_acc, list):
            n_record = len(record_acc)
            record_acc_time = [True for _ in range(n_record)]
            acc_time = [None for _ in range(n_record)]
        else:
            logger.fatal(f'Unrecognized record_acc: {record_acc}, must be float or list')
            raise ValueError
    else:
        _record_acc = False

    # define unsigned params
    if max_epoch is None:
        max_epoch = cbp_epoch
    assert cbp_epoch <= max_epoch

    if bp_lr is None:
        bp_lr = cbp_lr

    # print input train parameters
    logger.info('-' * 50)
    logger.info(f'Train Params:')
    logger.info(f'  zs: {zs}')
    logger.info(f'  beta: {beta}')
    logger.info(f'  cbp_epoch: {cbp_epoch}')
    logger.info(f'  max_epoch: {max_epoch}')
    logger.info(f'  cbp_lr: {cbp_lr}')
    logger.info(f'  bp_lr: {bp_lr}')
    logger.info(f'  bp_momentum: {bp_momentum}')
    logger.info(f'  bp_adam: {bp_adam}')
    logger.info(f'  logfile: {logfile}')
    logger.info('-' * 50)

    stime = time.time()  # start time
    logger.info('=' * 20 + ' start training ' + '=' * 20)
    _best_loss = best_loss
    best_loss_epoch = 0
    best_acc = 0
    best_acc_epoch = 0

    tl_list = []  # loss
    ta_list = []  # acc
    ws_list = []  # weight
    os_list = []  # output

    # CBP
    if cbp_epoch > 0:
        # check the zs
        num_p = len([p for p in model.parameters()])
        try:
            float(zs)
            zs = np.repeat(zs, num_p).astype(float)
        except (ValueError, TypeError):
            try:
                zs = np.asarray(zs, dtype=float)
                assert zs.ndim == 1
                if len(zs) != num_p:
                    zs = np.repeat(zs, 2)
                assert len(zs) == num_p
            except:
                logger.fatal(f'Unrecognized zs: {zs}, must be float or list-type')

        optimizer = chaos_optim.SGD(model.parameters(), lr=cbp_lr)
        for epoch in range(cbp_epoch):
            train_loss, _best_loss, weights, outputs = train_single_cbp_epoch(
                model, trainloader, optimizer, loss_func, zs=zs, best_loss=_best_loss,
                record_weight=record_weight, record_first_output=record_first_output,
                beta=beta, I0=I0, logdir=logdir, save_best_loss_model=save_best_loss_model,
                flatten_input=flatten_input, whole_weight=whole_weight, device=device, 
                clip_cbp_grad=clip_cbp_grad, clip_cbp_value=clip_cbp_value, **kws)
            predicts, acc = evaluate(model, testloader, device=device, onehot=onehot, flatten_input=flatten_input)
            mean_loss = np.mean(train_loss)
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_loss_epoch = epoch + 1
            if best_acc < acc:
                best_acc = acc
                best_acc_epoch = epoch + 1
                if save_best_acc_model:
                    torch.save(model.state_dict(), f'{logdir}best_acc_cbp_weights.pth')

            if _record_acc:
                for n in range(n_record):
                    if epoch > 0 and last_acc < record_acc[n] and acc >= record_acc[n] and record_acc_time[n]:
                        acc_time[n] = time.time() - stime
                        record_acc_time[n] = False
            tl_list += train_loss
            ta_list.append(acc)
            ws_list += weights
            os_list += outputs
            last_acc = acc
            logger.info(f'EPOCH: {epoch + 1:5d} | CBP | train loss: {mean_loss:.4f} | test acc: {acc:.4f}')

    # swith to BP
    if bp_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=bp_lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=bp_lr, momentum=bp_momentum)
    for epoch in range(cbp_epoch, max_epoch):
        train_loss, _best_loss, weights, outputs = train_single_bp_epoch(
            model, trainloader, optimizer, loss_func, best_loss=_best_loss,
            record_weight=record_weight, logdir=logdir, record_first_output=record_first_output,
            save_best_loss_model=save_best_loss_model, flatten_input=flatten_input,
            whole_weight=whole_weight, device=device, **kws
        )
        predicts, acc = evaluate(model, testloader, device=device, onehot=onehot, flatten_input=flatten_input)
        mean_loss = np.mean(train_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_loss_epoch = epoch + 1
        if best_acc < acc:
            best_acc = acc
            best_acc_epoch = epoch + 1
            if save_best_acc_model:
                torch.save(model.state_dict(), f'{logdir}best_acc_bp_weights.pth')

        if _record_acc:
            for n in range(n_record):
                if epoch > 0 and last_acc < record_acc[n] and acc >= record_acc[n] and record_acc_time[n]:
                    acc_time[n] = time.time() - stime
                    record_acc_time[n] = False
        tl_list += train_loss
        ta_list.append(acc)
        ws_list += weights
        os_list += outputs
        last_acc = acc
        logger.info(f'EPOCH: {epoch + 1:5d} | BP | train loss: {mean_loss:.4f} | test acc: {acc:.4f}')

    train_time = time.time() - stime  # finish time
    logger.info('=' * 20 + ' finish training ' + '=' * 20)
    logger.info(f'  training time: {train_time:.2f} s')
    logger.info(f'  reach best train loss {best_loss:.4f} at epoch {best_loss_epoch}')
    logger.info(f'  reach best test acc {best_acc:.4f} at epoch {best_acc_epoch}')

    if _record_acc:
        for n in range(n_record):
            if acc_time[n] is None:
                logger.warning(f'  do NOT reach {record_acc[n]:.2f} acc during {max_epoch} epoches')
            else:
                logger.info(f'  reach {record_acc[n]:.2f} acc in {acc_time[n]:.2f} s')

    logger.removeHandler(handler)
    logger.removeHandler(console)
    return tl_list, ta_list, ws_list, os_list


def train_single_cbp_epoch(model,
                           trainloader,
                           optimizer,
                           loss_func,
                           zs,
                           logdir=None,
                           scheduler=None,
                           best_loss=1e20,
                           beta=0.999,
                           I0=0.65,
                           flatten_input=False,
                           record_weight=True,
                           record_first_output=True,
                           whole_weight=False,
                           save_best_loss_model=False,
                           device=torch.device('cpu'),
                           clip_bp_grad=False,
                           clip_cbp_grad=False,
                           clip_bp_value=1.,
                           clip_cbp_value=0.01,
                           ):
    """ training with CBP in single epoch. """

    train_loss = []
    weights = []
    outputs = []
    best_model = None
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        if flatten_input:
            inputs = torch.flatten(inputs, start_dim=1)

        hids, out = model(inputs, return_hid=True)  # return_hid should be True in CBP
        loss = loss_func(out, labels)
        optimizer.zero_grad()
        loss.backward()

        if clip_bp_grad:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_bp_value)
        optimizer.step(zs, hids, I=I0, clip=clip_cbp_grad, clip_value=clip_cbp_value)  # the difference between CBP and BP

        cur_loss = loss.item()
        train_loss.append(cur_loss)

        if scheduler is not None:
            scheduler.step()

        zs *= beta

        if cur_loss < best_loss:
            best_loss = cur_loss
            if save_best_loss_model:
                best_model = deepcopy(model.state_dict())

        if record_weight:
            weight = []
            if whole_weight:
                for w in model.parameters():
                    weight.append(torch.clone(w.detach().cpu()).numpy())
            else:
                for w in model.parameters():
                    if w.ndim == 2:
                        weight.append(torch.clone(w[:10, 0].detach().cpu()).numpy())
                    elif w.ndim == 4:
                        weight.append(torch.clone(w[:10, 0, 0, 0].detach().cpu()).numpy())
                    else:
                        continue
            weights.append(weight)

        if record_first_output:
            outputs.append(out[0].detach().cpu().numpy())

    if save_best_loss_model and logdir is not None and best_model is not None:
        torch.save(best_model, f'{logdir}best_cbp_weights.pth')
    return train_loss, best_loss, weights, outputs


def train_single_bp_epoch(model,
                          trainloader,
                          optimizer,
                          loss_func,
                          logdir=None,
                          scheduler=None,
                          best_loss=1e10,
                          record_weight=True,
                          record_first_output=True,
                          save_best_loss_model=False,
                          whole_weight=False,
                          flatten_input=False,
                          device=torch.device('cpu'),
                          clip_bp_grad=False,
                          clip_bp_value=1.
                          ):
    """ training with BP in single epoch. """

    train_loss = []
    weights = []
    outputs = []
    best_model = None
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        if flatten_input:
            inputs = torch.flatten(inputs, start_dim=1)

        _, out = model(inputs)
        loss = loss_func(out, labels)
        optimizer.zero_grad()
        loss.backward()

        if clip_bp_grad:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_bp_value)
        optimizer.step()

        cur_loss = loss.item()
        train_loss.append(cur_loss)

        if scheduler is not None:
            scheduler.step()

        if cur_loss < best_loss:
            best_loss = cur_loss
            if save_best_loss_model:
                best_model = deepcopy(model.state_dict())

        if record_weight:
            weight = []
            if whole_weight:
                for w in model.parameters():
                    weight.append(torch.clone(w.detach().cpu()).numpy())
            else:
                for w in model.parameters():
                    if w.ndim == 2:
                        weight.append(torch.clone(w[:10, 0].detach().cpu()).numpy())
                    elif w.ndim == 4:
                        weight.append(torch.clone(w[:10, 0, 0, 0].detach().cpu()).numpy())
                    else:
                        continue
            weights.append(weight)

        if record_first_output:
            outputs.append(out[0].detach().cpu().numpy())

    if save_best_loss_model and logdir is not None and best_model is not None:
        torch.save(best_model, f'{logdir}best_bp_weights.pth')
    return train_loss, best_loss, weights, outputs


def debug_chaos(model,
                trainloader,
                z=20,
                loss_func=torch.nn.CrossEntropyLoss(),
                max_iter=800,
                max_epoch=200,
                I0=0.65,
                beta=0.99,
                flatten_input=False,
                whole_weight=False,
                device=torch.device('cpu'),
                **kws
                ):
    """ an auxiliary function to determine z. """
    num_p = len([p for p in model.parameters()])
    zs = np.asarray([z] * num_p, dtype=np.float)
    _state_dict = deepcopy(model.state_dict())
    optimizer = chaos_optim.SGD(model.parameters(), lr=0.0)
    model = model.to(device)

    _best_loss = 1e10
    ws_lists = []
    for _ in tqdm(range(max_iter)):
        model.load_state_dict(_state_dict)
        ws_list = []
        for epoch in range(max_epoch):
            train_loss, _best_loss, weights, outputs = train_single_cbp_epoch(
                model, trainloader, optimizer, loss_func, zs=zs, best_loss=_best_loss,
                record_weight=True, record_first_output=False, beta=1, I0=I0,
                save_best_loss_model=False, flatten_input=flatten_input,
                whole_weight=whole_weight, device=device, **kws)
            ws_list += weights
        zs *= beta
        ws_lists.append(ws_list)
    return ws_lists


def evaluate(model,
             testloader,
             device=torch.device('cpu'),
             onehot=False,
             flatten_input=False):
    """ evaluate the accuracy on the testloader. """
    correct = 0
    total = 0
    predicts = []
    with torch.no_grad():
        for (inputs, labels) in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if flatten_input:
                inputs = torch.flatten(inputs, start_dim=1)

            # calculate outputs by running images through the network
            _, outputs = model(inputs)

            # the class with the highest energy is what we choose as prediction
            if outputs.shape[-1] > 1:
                _, predicted = torch.max(outputs.data, 1)
            else:
                predicted = torch.round(outputs.data.flatten()).int()
            predicts.append(predicted.cpu().numpy())

            total += labels.size(0)
            if onehot:
                _, labels = torch.max(labels.data, 1)
            if labels.ndim > 1:
                labels = torch.round(labels.flatten()).int()
            correct += (predicted == labels).sum().item()

    predicts = np.hstack(predicts)
    acc = correct / total
    return predicts, acc


def evaluate_per_class(model,
                       testloader,
                       classes,
                       device=torch.device('cpu'),
                       flatten_input=False):
    """ evaluate the accuracy for each class. """
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for (inputs, labels) in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if flatten_input:
                inputs = torch.flatten(inputs, start_dim=1)
            _, outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    acc_per_class = []
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
        acc_per_class.append(accuracy)
    return acc_per_class
