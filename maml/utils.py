import copy
import torch
import torch.nn as nn

from maml.model import ConvNet, BasicBlock, BasicBlockWithoutResidual, ResNet
from torchmeta.datasets.helpers import (miniimagenet, tieredimagenet, cifar_fs, fc100,
                                        cub, vgg_flower, aircraft, traffic_sign, svhn, cars)
from collections import OrderedDict
from torchmeta.modules import MetaModule

def load_dataset(args, mode):
    folder = args.folder
    ways = args.num_ways
    shots = args.num_shots
    test_shots = 15
    download = args.download
    shuffle = True
    
    if mode == 'meta_train':
        args.meta_train = True
        args.meta_val = False
        args.meta_test = False
    elif mode == 'meta_valid':
        args.meta_train = False
        args.meta_val = True
        args.meta_test = False
    elif mode == 'meta_test':
        args.meta_train = False
        args.meta_val = False
        args.meta_test = True  
    
    if args.dataset == 'miniimagenet':
        dataset = miniimagenet(folder=folder,
                               shots=shots,
                               ways=ways,
                               shuffle=shuffle,
                               test_shots=test_shots,
                               meta_train=args.meta_train,
                               meta_val=args.meta_val,
                               meta_test=args.meta_test,
                               download=download)
    elif args.dataset == 'tieredimagenet':
        dataset = tieredimagenet(folder=folder,
                                 shots=shots,
                                 ways=ways,
                                 shuffle=shuffle,
                                 test_shots=test_shots,
                                 meta_train=args.meta_train,
                                 meta_val=args.meta_val,
                                 meta_test=args.meta_test,
                                 download=download)
    elif args.dataset == 'cifar_fs':
        dataset = cifar_fs(folder=folder,
                           shots=shots,
                           ways=ways,
                           shuffle=shuffle,
                           test_shots=test_shots,
                           meta_train=args.meta_train,
                           meta_val=args.meta_val,
                           meta_test=args.meta_test,
                           download=download)
    elif args.dataset == 'fc100':
        dataset = fc100(folder=folder,
                        shots=shots,
                        ways=ways,
                        shuffle=shuffle,
                        test_shots=test_shots,
                        meta_train=args.meta_train,
                        meta_val=args.meta_val,
                        meta_test=args.meta_test,
                        download=download)
    elif args.dataset == 'cub':
        dataset = cub(folder=folder,
                      shots=shots,
                      ways=ways,
                      shuffle=shuffle,
                      test_shots=test_shots,
                      meta_train=args.meta_train,
                      meta_val=args.meta_val,
                      meta_test=args.meta_test,
                      download=download)
    elif args.dataset == 'vgg_flower':
        dataset = vgg_flower(folder=folder,
                             shots=shots,
                             ways=ways,
                             shuffle=shuffle,
                             test_shots=test_shots,
                             meta_train=args.meta_train,
                             meta_val=args.meta_val,
                             meta_test=args.meta_test,
                             download=download)
    elif args.dataset == 'aircraft':
        dataset = aircraft(folder=folder,
                           shots=shots,
                           ways=ways,
                           shuffle=shuffle,
                           test_shots=test_shots,
                           meta_train=args.meta_train,
                           meta_val=args.meta_val,
                           meta_test=args.meta_test,
                           download=download)
    elif args.dataset == 'traffic_sign':
        dataset = traffic_sign(folder=folder,
                               shots=shots,
                               ways=ways,
                               shuffle=shuffle,
                               test_shots=test_shots,
                               meta_train=args.meta_train,
                               meta_val=args.meta_val,
                               meta_test=args.meta_test,
                               download=download)
    elif args.dataset == 'svhn':
        dataset = svhn(folder=folder,
                               shots=shots,
                               ways=ways,
                               shuffle=shuffle,
                               test_shots=test_shots,
                               meta_train=args.meta_train,
                               meta_val=args.meta_val,
                               meta_test=args.meta_test,
                               download=download)
    elif args.dataset == 'cars':
        dataset = cars(folder=folder,
                               shots=shots,
                               ways=ways,
                               shuffle=shuffle,
                               test_shots=test_shots,
                               meta_train=args.meta_train,
                               meta_val=args.meta_val,
                               meta_test=args.meta_test,
                               download=download)
        
    return dataset

def load_model(args):
    if args.dataset == 'miniimagenet' or args.dataset == 'tieredimagenet' or args.dataset == 'cub' or args.dataset == 'cars':
        wh_size = 5
    elif args.dataset == 'cifar_fs' or args.dataset == 'fc100' or args.dataset == 'vgg_flower' or args.dataset == 'aircraft' or args.dataset == 'traffic_sign' or args.dataset == 'svhn':
        wh_size = 2
        
    if args.model == '4conv':
        model = ConvNet(in_channels=3, out_features=args.num_ways, hidden_size=args.hidden_size, wh_size=wh_size)
    elif args.model == 'resnet':
        if args.blocks_type == 'a':
            blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlock]
        elif args.blocks_type == 'b':
            blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlockWithoutResidual]
        elif args.blocks_type == 'c':
            blocks = [BasicBlock, BasicBlock, BasicBlockWithoutResidual, BasicBlockWithoutResidual]
        elif args.blocks_type == 'd':
            blocks = [BasicBlock, BasicBlockWithoutResidual, BasicBlockWithoutResidual, BasicBlockWithoutResidual]
        elif args.blocks_type == 'e':
            blocks = [BasicBlockWithoutResidual, BasicBlockWithoutResidual, BasicBlockWithoutResidual, BasicBlockWithoutResidual]
        
        model = ResNet(blocks=blocks, keep_prob=1.0, avg_pool=True, drop_rate=0.0, out_features=args.num_ways, wh_size=1)
    return model

def update_parameters(model,
                      loss,
                      extractor_step_size,
                      classifier_step_size,
                      params=None,
                      first_order=False):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.
    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.
    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.
    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.
    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.
    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the inner-loss.
    """
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)

    updated_params = OrderedDict()
    
    
    step_size=0.0 # not OrderedDict type
    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            if 'classifier' in name: # To control inner update parameter
                updated_params[name] = param - classifier_step_size * grad
            else:
                updated_params[name] = param - extractor_step_size * grad

    return updated_params

def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points

    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(num_examples,)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())