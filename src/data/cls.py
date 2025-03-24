import os
import torch
from data.imagenet import ImageNetDatasetPair
from utils.cls import PresetTrain, PresetEval


def load_data_pair(opt):
    use_trainset = opt.get('train', False)

    # data path
    train_hq_dir = os.path.join(opt['data']['hq_path'], "train")
    train_lq_dir = os.path.join(opt['data']['lq_path'], "train")
    val_hq_dir = os.path.join(opt['data']['hq_path'], "val")
    val_lq_dir = os.path.join(opt['data']['lq_path'], "val")
        
    # transforms
    transform_train = PresetTrain()
    transform_test = PresetEval()
    
    # datasets
    dataset_train = None
    if use_trainset:
        dataset_train = ImageNetDatasetPair(train_hq_dir, train_lq_dir, transform_train)
    dataset_test = ImageNetDatasetPair(val_hq_dir, val_lq_dir, transform_test)
    
    # distributed training
    train_sampler = None
    if opt['dist']:
        if use_trainset:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        if use_trainset:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    # data loader
    data_loader_train = None
    if opt.get('train', False):
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=opt['train']['batch_size'], sampler=train_sampler, num_workers=opt['num_threads'], pin_memory=True, collate_fn=None)
        
    data_loader_test = None
    if opt.get('test', False):
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=opt['test']['batch_size'], sampler=test_sampler, num_workers=opt['num_threads'], pin_memory=True, collate_fn=None)
                
    return data_loader_train, data_loader_test, train_sampler, test_sampler
