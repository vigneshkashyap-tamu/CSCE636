import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset
from dataset.randaugment import RandomAugment

from dataset.open_clip_data import SharedEpoch, detshuffle2, tarfile_to_samples_nothrow, log_and_continue, filter_no_caption_or_no_image, expand_urls
import webdataset as wds
import math


def get_transform(image_res, is_train):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
     
    if is_train:
        transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_res, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2, 7, isPIL=True, augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ]) 
    else:
        transform = transforms.Compose([
            transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])   
    return transform


def create_train_dataset(dataset, args):

    train_transform = get_transform(args.image_res, is_train=True)
    
    if dataset=='re':          
        train_dataset = re_train_dataset([args.train_file], train_transform, args.train_image_root)          
        return train_dataset

    else:
        assert 0, dataset + " is not supported."


def create_val_dataset(dataset, args, val_file, val_image_root, test_file=None):

    test_transform = get_transform(args.image_res, is_train=False)
    
    if dataset=='re':
        val_dataset = re_eval_dataset(val_file, test_transform, val_image_root)  

        if test_file is not None:
            test_dataset = re_eval_dataset(test_file, test_transform, val_image_root)                
            return val_dataset, test_dataset
        else:
            return val_dataset

    else:
        assert 0, dataset + " is not supported."


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_train_loader(dataset, sampler, batch_size, num_workers, collate_fn):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                      sampler=sampler, shuffle=(sampler is None), collate_fn=collate_fn, drop_last=True, prefetch_factor=4)              


def create_val_loader(datasets, samplers, batch_size, num_workers, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, collate_fn in zip(datasets, samplers, batch_size, num_workers, collate_fns):
        shuffle = False
        drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            prefetch_factor=12
        )              
        loaders.append(loader)
    return loaders  


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def json_parse_key(json_dict) -> int:
    return int(json_dict["key"])


# Adapted from https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
def get_wds_dataset(train_data, train_num_samples, seed, batch_size, workers, world_size, preprocess, image_res, epoch=0, floor=False, tokenize=None):
    print(train_data, flush=True)
    input_shards = train_data
    num_shards = None
    num_samples = train_num_samples
    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    pipeline.extend([
        detshuffle2(
            bufsize=_SHARD_SHUFFLE_SIZE,
            initial=_SHARD_SHUFFLE_INITIAL,
            seed=seed,
            epoch=shared_epoch,
        ),
        wds.split_by_node,
        wds.split_by_worker,
    ])
    pipeline.extend([
        # at this point, we have an iterator over the shards assigned to each worker at each node
        tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
        wds.shuffle(
            bufsize=_SAMPLE_SHUFFLE_SIZE,
            initial=_SAMPLE_SHUFFLE_INITIAL,
        ),
    ])

    if preprocess is not None:
        train_transform = preprocess
    else:
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_res, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2, 7, isPIL=True, augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ]) 

    # here we also load the key of data
    rename = wds.rename(image="jpg;png;jpeg;webp", text="txt", key="json")
    if tokenize is not None:
        map_dict = wds.map_dict(image=train_transform, text=tokenize, key=json_parse_key)
    else:
        map_dict = wds.map_dict(image=train_transform, key=json_parse_key)
    to_tuple = wds.to_tuple("image", "text", "key", "key")
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        rename, map_dict, to_tuple,
        wds.batched(batch_size, partial=False)
    ])

    dataset = wds.DataPipeline(*pipeline)

    num_shards = num_shards or len(expand_urls(input_shards)[0])
    assert num_shards >= workers * world_size, 'number of shards must be >= total workers'
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = batch_size * world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor = 4,
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return dataloader
