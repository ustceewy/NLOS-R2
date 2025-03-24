import torch
import utils.common
import torch.utils.data
from models import make_model
from data.cls import load_data_pair


def main():
    # load opt and args from yaml
    opt, args = utils.common.parse_options()
    opt = utils.common.init_distributed_mode(opt)
    
    # deterministic option for reproduction
    if opt.get('deterministic', False):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        
    # make model
    model = make_model(opt)
    
    # prepare data loader
    data_loader_train, data_loader_test, train_sampler, test_sampler = load_data_pair(opt)

    # training
    if opt.get('train', False):
        model.init_training_settings(data_loader_train)
        start_epoch, end_epoch = 1, opt['train']['epoch']
        model.text_logger.write("Start training")
        
        for epoch in range(start_epoch, end_epoch+1):
            model.train_one_epoch(data_loader_train, train_sampler, epoch)
            model.evaluate(data_loader_test, epoch)
            model.save(epoch)

    else:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model.evaluate(data_loader_test, test_sampler)


if __name__ == "__main__":
    main()
