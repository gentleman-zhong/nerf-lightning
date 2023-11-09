import torch
from torch.optim import SGD, Adam
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR
# from .warmup_scheduler import GradualWarmupScheduler
from kornia.losses import ssim as dssim
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T


# 提取模型参数信息
def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


#加载检查点
def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path:
        return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict, strict=False)


# 获得参数
def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else: # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters


# 得到优化器
def get_optimizer(hparams, models):
    eps = 1e-8
    parameters = get_parameters(models)
    if hparams.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=hparams.lr,
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':
        optimizer = Adam(parameters, lr=hparams.lr, eps=eps,
                         weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'radam':
        optimizer = optim.RAdam(parameters, lr=hparams.lr, eps=eps,
                                weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'ranger':
        optimizer = optim.Ranger(parameters, lr=hparams.lr, eps=eps,
                                 weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer


# 调整学习率
def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step,
                                gamma=hparams.decay_gamma)
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)
    elif hparams.lr_scheduler == 'poly':
        scheduler = LambdaLR(optimizer,
                             lambda epoch: (1-epoch/hparams.num_epochs)**hparams.poly_exp)
    else:
        raise ValueError('scheduler not recognized!')

    # if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
    #     scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier,
    #                                        total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)

    return scheduler


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/max(ma-mi, 1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_