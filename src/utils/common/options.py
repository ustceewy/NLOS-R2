import os
import yaml
import random
import argparse
from collections import OrderedDict
from .misc import set_random_seed


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()

    # parse yml to dict
    opt = yaml_load(args.opt)
    
    # random seed
    seed = opt.get('manual_seed', None)
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
        # print('random seed: {:05d}'.format(seed))
    set_random_seed(seed)
    
    # set task
    opt['task'] = 'cls'
        
    # test_only
    if args.test_only:
        opt['test_only'] = True
    if opt.get('test_only', False):
        if opt.get('train', False):
            opt.pop('train')
    
    return opt, args


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])
