import os
import torch
import warnings
from losses import build_loss
from archs import build_network
from utils.cls import MetricLogger, calculate_accuracy
from utils.common import save_on_master, quantize, reduce_across_processes, calculate_psnr_batch, calculate_lpips_batch
from .base_model import BaseModel
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def make_model(opt):
    return SEfeatureClassificationModel(opt)


class SEModule(torch.nn.Module):
    def __init__(self, channels, ratio=1/16):
        super().__init__()
        self._r = ratio
        self._avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        hidden_channels = int(channels * self._r)
        self._fc1 = torch.nn.Linear(channels, hidden_channels, bias=False)
        self._relu = torch.nn.ReLU(inplace=True)
        self._fc2 = torch.nn.Linear(hidden_channels, channels, bias=False)
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):  # torch.Size([batch_size, 1024, 7, 7])
        b, c, _, _ = inputs.size()
        x = self._avg_pool(inputs).squeeze()
        x = self._relu(self._fc1(x))
        x = self._sigmoid(self._fc2(x))  # torch.Size([batch_size, 1024])
        return inputs * x.view(b, c, 1, 1).expand_as(inputs)  # torch.Size([batch_size, 1024, 7, 7])


class Fusion_Network(torch.nn.Module):
    def __init__(self, net_cls_sr, net_cls_lr):
        super(Fusion_Network, self).__init__()
        self.net_cls_sr = net_cls_sr
        self.net_cls_lr = net_cls_lr
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(1024, 42)
        self.se_module = SEModule(channels=1024)

    def forward(self, img_sr, img_lr):
        _, feat_sr = self.net_cls_sr(img_sr, return_feats=True)
        _, feat_lr = self.net_cls_lr(img_lr, return_feats=True)
        # feat_sr['l1'].shape = torch.Size([batch_size, 64, 56, 56])
        # feat_sr['l2'].shape = torch.Size([batch_size, 128, 28, 28])
        # feat_sr['l3'].shape = torch.Size([batch_size, 256, 14, 14])
        # feat_sr['l4'].shape = torch.Size([batch_size, 512, 7, 7])
        feat = torch.cat((feat_sr['l4'], feat_lr['l4']), 1)  # torch.Size([batch_size, 1024, 7, 7])
        feat = self.se_module(feat)
        x = self.avgpool(feat)  # torch.Size([batch_size, 1024, 1, 1])
        x = torch.flatten(x, 1)  # torch.Size([batch_size, 1024])
        x = self.fc(x)
        return x


class SEfeatureClassificationModel(BaseModel):
    """Super-Resolution model for Image Classification."""

    def __init__(self, opt):
        super().__init__(opt)
                
        # define network sr
        self.net_sr = build_network(opt['network_sr'], self.text_logger, tag='net_sr')
        self.load_network(self.net_sr, name='network_sr', tag='net_sr')
        self.net_sr = self.model_to_device(self.net_sr, is_trainable=True)
        self.print_network(self.net_sr, tag='net_sr')
        
        # define network cls
        if not self.opt.get('test_only'):
            # define network cls
            self.net_cls_sr = build_network(opt['network_cls_sr'], self.text_logger, task=self.task, tag='net_cls_sr')
            self.load_network(self.net_cls_sr, name='network_cls_sr', tag='net_cls_sr')
            
            self.net_cls_lr = build_network(opt['network_cls_lr'], self.text_logger, task=self.task, tag='net_cls_lr')
            self.load_network(self.net_cls_lr, name='network_cls_lr', tag='net_cls_lr')

            self.net_cls = Fusion_Network(self.net_cls_sr, self.net_cls_lr)
            self.net_cls = self.model_to_device(self.net_cls, is_trainable=True)
            self.print_network(self.net_cls, tag='net_cls')
        else:
            # define network cls
            self.net_cls_sr = build_network(opt['network_cls_sr'], self.text_logger, task=self.task, tag='net_cls')
            self.net_cls_lr = build_network(opt['network_cls_lr'], self.text_logger, task=self.task, tag='net_cls')
            self.net_cls = Fusion_Network(self.net_cls_sr, self.net_cls_lr)

            self.load_network(self.net_cls, name='network_cls', tag='net_cls')
            self.net_cls = self.model_to_device(self.net_cls, is_trainable=True)
            self.print_network(self.net_cls, tag='net_cls')

    def set_mode(self, mode):
        if mode == 'train':
            self.net_sr.train()
            self.net_cls.train()
        elif mode == 'eval':
            self.net_sr.eval()
            self.net_cls.eval()
        else:
            raise NotImplementedError(f"mode {mode} is not supported")
        
    def init_training_settings(self, data_loader_train):
        self.set_mode(mode='train')
        train_opt = self.opt['train']

        # phase 1
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt'], self.text_logger).to(self.device)
            
        if train_opt.get('tdp_opt'):
            self.cri_tdp = build_loss(train_opt['tdp_opt'], self.text_logger).to(self.device)
            
        # phase 2
        if train_opt.get('ce_opt'):
            self.cri_ce = build_loss(train_opt['ce_opt'], self.text_logger).to(self.device)
        
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers(len(data_loader_train), name='sr', optimizer=self.optimizer_sr)
        self.setup_schedulers(len(data_loader_train), name='cls', optimizer=self.optimizer_cls)
        
        # set up saving directories
        os.makedirs(os.path.join(self.exp_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        
        # eval freq
        self.eval_freq = train_opt.get('eval_freq', 1)

        # warmup epoch
        self.warmup_epoch = train_opt.get('warmup_epoch', -1)
        self.text_logger.write("NOTICE: total epoch: {}, warmup epoch: {}".format(train_opt['epoch'], self.warmup_epoch))

    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        # optimizer sr
        optim_type = train_opt['optim_sr'].pop('type')
        self.optimizer_sr = self.get_optimizer(optim_type, self.net_sr.parameters(), **train_opt['optim_sr'])
        self.optimizers.append(self.optimizer_sr)
        
        # optimizer cls
        optim_type = train_opt['optim_cls'].pop('type')
        self.optimizer_cls = self.get_optimizer(optim_type, self.net_cls.parameters(), **train_opt['optim_cls'])
        self.optimizers.append(self.optimizer_cls)
        
    def train_one_epoch(self, data_loader_train, train_sampler, epoch):
        self.set_mode(mode='train')
        print(f'lr_sr = {round(self.optimizer_sr.param_groups[0]["lr"], 8)}, lr_cls = {round(self.optimizer_cls.param_groups[0]["lr"], 8)}')
        
        if self.dist:
            train_sampler.set_epoch(epoch)

        if epoch < self.warmup_epoch + 1:
            self.text_logger.write("NOTICE: Doing warm-up")

        # phase 1;
        # update net_sr, freeze net_cls
        for p in self.net_sr.parameters(): p.requires_grad = True
        for p in self.net_cls.parameters(): p.requires_grad = False
        for iter, (img_hr, img_lr, label) in enumerate(data_loader_train):
            img_hr, img_lr, label = img_hr.to(self.device), img_lr.to(self.device), label.to(self.device)
            current_iter = iter + len(data_loader_train)*(epoch-1)

            img_sr = self.net_sr(img_lr)
            self.optimizer_sr.zero_grad()

            l_total_sr = 0
            if hasattr(self, 'cri_pix'):
                l_pix = self.cri_pix(img_sr, img_hr)
                self.tb_logger.add_scalar('losses/l_pix', l_pix.item(), current_iter)
                l_total_sr += l_pix

            if epoch > self.warmup_epoch:
                if hasattr(self, 'cri_tdp'):
                    img_1 = self.transform_train(img_sr)
                    img_2 = self.transform_train(img_lr)
                    pred_sr = self.net_cls(img_1, img_2)
                    l_tdp = self.cri_tdp(pred_sr, label)
                    self.tb_logger.add_scalar('losses/l_tdp', l_tdp.item(), current_iter)
                    l_total_sr += l_tdp

            self.tb_logger.add_scalar('losses/l_total_sr', l_total_sr.item(), current_iter)
            l_total_sr.backward()
            self.optimizer_sr.step()
            self.schedulers[0].step()
            
        # phase 2;
        # update network_cls, freeze net_sr
        for p in self.net_sr.parameters(): p.requires_grad = False
        for p in self.net_cls.parameters(): p.requires_grad = True
        for iter, (_, img_lr, label) in enumerate(data_loader_train):
            img_lr, label = img_lr.to(self.device), label.to(self.device)
            current_iter = iter + len(data_loader_train)*(epoch-1)

            img_sr = self.net_sr(img_lr).detach()
            self.optimizer_cls.zero_grad()

            l_total_cls = 0
            if hasattr(self, 'cri_ce'):
                img_1 = self.transform_train(img_sr)
                img_2 = self.transform_train(img_lr)
                pred_sr = self.net_cls(img_1, img_2)
                l_ce = self.cri_ce(pred_sr, label)
                self.tb_logger.add_scalar('losses/l_ce', l_ce.item(), current_iter)
                l_total_cls += l_ce
            
            l_total_cls.backward()
            self.optimizer_cls.step()
            self.schedulers[1].step()
        return

    @torch.inference_mode()            
    def evaluate(self, data_loader_test, epoch=0):
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return
            
        self.set_mode(mode='eval')
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test: "

        account = 0
        avg_psnr_offical, avg_ssim_offical = 0.0, 0.0
        num_processed_samples = 0
        for i, (img_hr, img_lr, label) in enumerate(metric_logger.log_every(data_loader_test, self.opt['print_freq'], self.text_logger, header)):
            img_hr, img_lr, label = img_hr.to(self.device), img_lr.to(self.device), label.to(self.device)

            # super resolution
            img_sr = self.net_sr(img_lr)  # torch.Size([batch_size, 3, 256, 256])

            # save
            # from PIL import Image
            # for idx, single_img_sr in enumerate(img_sr):
            #     img = single_img_sr.permute(1, 2, 0) * 255  # torch.Size([3, 256, 256]) -> torch.Size([256, 256, 3])
            #     img = img.clamp(0, 255).to(torch.uint8).cpu().numpy()  # (256, 256, 3), 取值范围(0, 255)
            #     img = Image.fromarray(img)
            #     save_path = os.path.join('/share2/data/wangyi/paper/result/ours', f'{i}_{idx}.png')
            #     img.save(save_path)
            
            # image classification
            img_1 = self.transform_eval(img_sr)
            img_2 = self.transform_eval(img_lr)
            pred_sr = self.net_cls(img_1, img_2)

            # pred = pred_sr.argmax(dim=1)
            # class_names = ['airplane', 'apple', 'ball', 'bear', 'bed', 'bench', 'bird', 'burger', 'butterfly', 'car',
            #                'cat', 'clock', 'cup', 'dog', 'elephant', 'fox', 'frog', 'horse', 'house', 'koala',
            #                'ladybug', 'monkey', 'motorcycle', 'mushroom', 'panda', 'pen', 'phone', 'piano', 'pizza', 'rabbit',
            #                'shark', 'ship', 'shoe', 'snail', 'snake', 'spaghetti', 'swan', 'table', 'tie', 'tiger',
            #                'train', 'turtle']
            # pred = [class_names[idx] for idx in pred]
            # true = [class_names[idx.item()] for idx in label]
            # correct = ['True' if pred[i] == true[i] else 'False' for i in range(len(pred))]
            # with open('/share2/data/wangyi/paper/result/ours/pred.txt', 'a') as f:
            #     for i in range(len(pred)):
            #         f.write(f"{pred[i]} {correct[i]}\n")

            # evaluation on validation batch
            batch_size = img_hr.shape[0]
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr), img_hr)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)                
            if self.opt['test'].get('calculate_lpips', False):
                lpips, valid_batch_size = calculate_lpips_batch(quantize(img_sr), img_hr, self.net_lpips)
                metric_logger.meters["lpips"].update(lpips.item(), n=valid_batch_size)
            acc1_sr, acc5_sr = calculate_accuracy(pred_sr, label, topk=(1, 5))
            metric_logger.meters["acc1_sr"].update(acc1_sr.item(), n=batch_size)
            metric_logger.meters["acc5_sr"].update(acc5_sr.item(), n=batch_size)
            num_processed_samples += batch_size            

            if self.opt['test'].get('calculate_psnr_ssim', False):
                batch_size = img_hr.shape[0]
                for i in range(batch_size):
                    account += 1
                    single_img_hr = (img_hr[i] * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                    single_img_sr = (img_sr[i] * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                    psnr_offical = compare_psnr(single_img_hr, single_img_sr, data_range=255)
                    avg_psnr_offical += psnr_offical
                    ssim_offical = compare_ssim(single_img_hr, single_img_sr, data_range=255, channel_axis=0)
                    avg_ssim_offical += ssim_offical

        # gather the stats from all processes
        num_processed_samples = reduce_across_processes(num_processed_samples)
        if (
            hasattr(data_loader_test.dataset, "__len__")
            and len(data_loader_test.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
        ):
            warnings.warn(
                f"It looks like the dataset has {len(data_loader_test.dataset)} samples, but {num_processed_samples} "
                "samples were used for the validation, which might bias the results. "
                "Try adjusting the batch size and / or the world size. "
                "Setting the world size to 1 is always a safe bet."
            )

        # metirc logger
        metric_logger.synchronize_between_processes()
        metric_summary = f"{header}"
        metric_summary = self.add_metric(metric_summary, 'PSNR', metric_logger.psnr.global_avg, epoch)
        if self.opt['test'].get('calculate_lpips', False):
            metric_summary = self.add_metric(metric_summary, 'LPIPS', metric_logger.lpips.global_avg, epoch)
        metric_summary = self.add_metric(metric_summary, 'ACC_SR@1', metric_logger.acc1_sr.global_avg, epoch)
        metric_summary = self.add_metric(metric_summary, 'ACC_SR@5', metric_logger.acc5_sr.global_avg, epoch)
        self.text_logger.write(metric_summary)

        if self.opt['test'].get('calculate_psnr_ssim', False):
            avg_psnr_offical /= account
            avg_ssim_offical /= account
            print(f'Account = {account}, PSNR = {avg_psnr_offical}, SSIM = {avg_ssim_offical}')
        return

    def save(self, epoch):            
        checkpoint = {"epoch": epoch,
                      "opt": self.opt,
                      "net_sr": self.get_bare_model(self.net_sr).state_dict(),
                      "net_cls": self.get_bare_model(self.net_cls).state_dict(),
                      'schedulers': [],
                      }
        for s in self.schedulers:
            checkpoint['schedulers'].append(s.state_dict())
                
        if epoch % self.opt['train']['save_freq'] == 0:
            save_on_master(self.get_bare_model(self.net_sr).state_dict(), os.path.join(self.exp_dir, 'models', "net_sr_{:03d}.pth".format(epoch)))
            save_on_master(self.get_bare_model(self.net_cls).state_dict(), os.path.join(self.exp_dir, 'models', "net_cls_{:03d}.pth".format(epoch)))
            save_on_master(checkpoint, os.path.join(self.exp_dir, 'checkpoints', "checkpoint_{:03d}.pth".format(epoch)))
            
        save_on_master(self.get_bare_model(self.net_sr).state_dict(), os.path.join(self.exp_dir, 'models', "net_sr_latest.pth"))
        save_on_master(self.get_bare_model(self.net_cls).state_dict(), os.path.join(self.exp_dir, 'models', "net_cls_latest.pth"))
        save_on_master(checkpoint, os.path.join(self.exp_dir, 'checkpoints', "checkpoint_latest.pth"))
        return
