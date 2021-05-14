import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict

from torchmeta.utils.data import BatchMetaDataLoader
from maml.utils import load_dataset, load_model, update_parameters, get_accuracy

def main(args, mode, iteration=None):
    dataset = load_dataset(args, mode)
    dataloader = BatchMetaDataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    model.to(device=args.device)
    model.train()
    
    # To control update parameter
    head_params = [p for name, p in model.named_parameters() if 'classifier' in name]
    body_params = [p for name, p in model.named_parameters() if 'classifier' not in name]
    
    # inner update
    step_size = OrderedDict()
    
    # For ablation study (Appendix K)
    if args.model=='resnet':
        for name, _ in model.named_parameters():
            if 'classifier' in name:
                step_size[name] = args.classifier_step_size
            else:
                if args.save_name in ['MAML_b', 'BOIL_b', 'ANIL_b']:
                    step_size[name] = args.extractor_step_size
                else:
                    name_list = []
                    for i in args.save_name.split('_')[-1]:
                        name_list.append("layer" +  i)
                    if name[:6] in name_list:
                        step_size[name] = args.extractor_step_size
                    else:
                        step_size[name] = 0.0
    elif args.model == '4conv_sep':
        for name, _ in model.named_parameters():
            if 'classifier' in name:
                step_size[name] = args.classifier_step_size
            else:
                if 'conv'+args.save_name[-1] in name:
                    step_size[name] = args.extractor_step_size
                else:
                    step_size[name] = 0.0
    else:
        for name, _ in model.named_parameters():
            if 'classifier' in name:
                step_size[name] = args.classifier_step_size
            else:
                step_size[name] = args.extractor_step_size
        
    # outer update (Appendix L)
    if args.outer_fix:
        meta_optimizer = torch.optim.Adam([{'params': head_params, 'lr': 0},
                                           {'params': body_params, 'lr': args.meta_lr}])
    else:
        meta_optimizer = torch.optim.Adam([{'params': head_params, 'lr': args.meta_lr},
                                           {'params': body_params, 'lr': args.meta_lr}])
    
    if args.meta_train:
        total = args.train_batches
    elif args.meta_val:
        total = args.valid_batches
    elif args.meta_test:
        total = args.test_batches
        
    loss_logs, accuracy_logs = [], []
    
    # Training loop
    with tqdm(dataloader, total=total, leave=False) as pbar:
        for batch_idx, batch in enumerate(pbar):
            if args.centering:
                fc_weight_mean = torch.mean(model.classifier.weight.data, dim=0)
                model.classifier.weight.data -= fc_weight_mean
                            
            support_inputs, support_targets = batch['train']
            support_inputs = support_inputs.to(device=args.device)
            support_targets = support_targets.to(device=args.device)

            query_inputs, query_targets = batch['test']
            query_inputs = query_inputs.to(device=args.device)
            query_targets = query_targets.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
                
            for task_idx, (support_input, support_target, query_input, query_target) in enumerate(zip(support_inputs, support_targets, query_inputs, query_targets)):
                params = None
                for _ in range(args.inner_update_num):
                    support_features, support_logit = model(support_input, params=params)
                    inner_loss = F.cross_entropy(support_logit, support_target)
                    
                    model.zero_grad()
                    params = update_parameters(model=model,
                                               loss=inner_loss,
                                               params=params,
                                               step_size=step_size,
                                               first_order=args.first_order)
                
                query_features, query_logit = model(query_input, params=params)
                outer_loss += F.cross_entropy(query_logit, query_target)
                
                with torch.no_grad():
                    accuracy += get_accuracy(query_logit, query_target)
                    
            outer_loss.div_(args.batch_size)
            accuracy.div_(args.batch_size)
            loss_logs.append(outer_loss.item())
            accuracy_logs.append(accuracy.item())
            
            if args.meta_train:
                meta_optimizer.zero_grad()
                outer_loss.backward()
                meta_optimizer.step()
                
            postfix = {'mode': mode, 'iter': iteration, 'acc': round(accuracy.item(), 5)}
            pbar.set_postfix(postfix)
            if batch_idx+1 == total:
                break

    # Save model
    if args.meta_train:
        filename = os.path.join(args.output_folder, args.save_dir, 'models', 'epochs_{}.pt'.format((iteration+1)*total))
        if (iteration+1)*total % 5000 == 0:
            with open(filename, 'wb') as f:
                state_dict = model.state_dict()
                torch.save(state_dict, f)


    # Save best model
    if args.meta_val:
        filename = os.path.join(args.output_folder, args.save_dir, 'logs', 'logs.csv')
        valid_logs = list(pd.read_csv(filename)['valid_accuracy'])
        
        max_acc = max(valid_logs)
        curr_acc = np.mean(accuracy_logs)
        
        if max_acc < curr_acc:
            filename = os.path.join(args.output_folder, args.save_dir, 'models', 'best_val_acc_model.pt')
            with open(filename, 'wb') as f:
                state_dict = model.state_dict()
                torch.save(state_dict, f)
                
    return loss_logs, accuracy_logs

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    
    parser.add_argument('--folder', type=str, help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str, help='Dataset: miniimagenet, tieredimagenet, cub, cars, cifar_fs, fc100, aircraft, vgg_flower')
    parser.add_argument('--model', type=str, help='Model: 4conv, resnet')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
    parser.add_argument('--download', action='store_true', help='Download the dataset in the data folder.')
    parser.add_argument('--num-shots', type=int, default=5, help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate of meta optimizer.')

    parser.add_argument('--first-order', action='store_true', help='Use the first-order approximation of MAML.')
    parser.add_argument('--inner-update-num', type=int, default=1, help='The number of inner updates (default: 1).')
    parser.add_argument('--extractor-step-size', type=float, default=0.5, help='Extractor step-size for the gradient step for adaptation (default: 0.5).')
    parser.add_argument('--classifier-step-size', type=float, default=0.5, help='Classifier step-size for the gradient step for adaptation (default: 0.5).')
    parser.add_argument('--hidden-size', type=int, default=64, help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--blocks-type', type=str, default=None, help='Resnet block type (optional).')
    
    parser.add_argument('--output-folder', type=str, default='./output/', help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--save-name', type=str, default=None, help='Name of model (optional).')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--batch-iter', type=int, default=300, help='Number of times to repeat train batches (i.e., total epochs = batch_iter * train_batches) (default: 300).')
    parser.add_argument('--train-batches', type=int, default=100, help='Number of batches the model is trained over (i.e., validation save steps) (default: 100).')
    parser.add_argument('--valid-batches', type=int, default=25, help='Number of batches the model is validated over (default: 25).')
    parser.add_argument('--test-batches', type=int, default=2500, help='Number of batches the model is tested over (default: 2500).')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading (default: 1).')
    
    parser.add_argument('--centering', action='store_true', help='Parallel shift operation in the head.')
    parser.add_argument('--ortho-init', action='store_true', help='Use the head from the orthononal model.')
    parser.add_argument('--outer-fix', action='store_true', help='Fix the head during outer updates.')
    
    args = parser.parse_args()
    args.save_dir = '{}_{}shot_{}_{}'.format(args.dataset,
                                             args.num_shots,
                                             args.model,
                                             args.save_name)
    os.makedirs(os.path.join(args.output_folder, args.save_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, args.save_dir, 'models'), exist_ok=True)
    
    arguments_txt = "" 
    for k, v in args.__dict__.items():
        arguments_txt += "{}: {}\n".format(str(k), str(v))
    filename = os.path.join(args.output_folder, args.save_dir, 'logs', 'arguments.txt')
    with open(filename, 'w') as f:
        f.write(arguments_txt[:-1])
    
    args.device = torch.device(args.device)  
    model = load_model(args)
        
    if args.ortho_init:
        X = np.random.randn(5, 1600)
        Q = gs(X)
        
        model.classifier.weight.data = nn.Parameter(Q)
    
    log_pd = pd.DataFrame(np.zeros([args.batch_iter*args.train_batches, 6]),
                          columns=['train_error', 'train_accuracy', 'valid_error', 'valid_accuracy', 'test_error', 'test_accuracy'])
    filename = os.path.join(args.output_folder, args.save_dir, 'logs', 'logs.csv')
    log_pd.to_csv(filename, index=False)
    
    for iteration in tqdm(range(args.batch_iter)):
        meta_train_loss_logs, meta_train_accuracy_logs = main(args=args, mode='meta_train', iteration=iteration)
        meta_valid_loss_logs, meta_valid_accuracy_logs = main(args=args, mode='meta_valid', iteration=iteration)
        log_pd['train_error'][iteration*args.train_batches:(iteration+1)*args.train_batches] = meta_train_loss_logs
        log_pd['train_accuracy'][iteration*args.train_batches:(iteration+1)*args.train_batches] = meta_train_accuracy_logs
        log_pd['valid_error'][(iteration+1)*args.train_batches-1] = np.mean(meta_valid_loss_logs)
        log_pd['valid_accuracy'][(iteration+1)*args.train_batches-1] = np.mean(meta_valid_accuracy_logs)
        filename = os.path.join(args.output_folder, args.save_dir, 'logs', 'logs.csv')
        log_pd.to_csv(filename, index=False)
        
    meta_test_loss_logs, meta_test_accuracy_logs = main(args=args, mode='meta_test')
    log_pd['test_error'][args.batch_iter*args.train_batches-1] = np.mean(meta_test_loss_logs)
    log_pd['test_accuracy'][args.batch_iter*args.train_batches-1] = np.mean(meta_test_accuracy_logs)
    filename = os.path.join(args.output_folder, args.save_dir, 'logs', 'logs.csv')
    log_pd.to_csv(filename, index=False)