import torch
from torchvision.transforms.functional import normalize


def calculate_psnr(img, img2, crop_border=8, img_range=1.0, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (torch.Tensor): Images with range [0, 255].
        img2 (torch.Tensor): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    img = img * 255.0 / img_range
    img2 = img2 * 255.0 / img_range

    if crop_border != 0:
        img = img[:, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, crop_border:-crop_border, crop_border:-crop_border]

    mse = torch.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return (10. * torch.log10(255. * 255. / mse)).item()


def calculate_psnr_batch(img, img2, crop_border=8, img_range=1.0, **kwargs):
    """Computes the PSNR (Peak-Signal-Noise-Ratio) in batch"""
        
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    img = img * 255.0 / img_range
    img2 = img2 * 255.0 / img_range

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    mse = torch.mean((img - img2)**2, dim=(1,2,3))  # batch-wise mse
    valid_mask = (mse != 0.)  # do not count if mse is 0; it causes infinity
    # NOTE : there exist zero mse case where the patch is from 
    # val/n03623198/ILSVRC2012_val_00017853.JPEG (all-zero patch)
    mse = mse[valid_mask]
    
    return (10. * torch.log10(255. * 255. / mse)).mean(), valid_mask.sum()  # batch-wise mean


def calculate_lpips_batch(img, img2, net_lpips, crop_border=8, img_range=1.0, **kwargs):
    """Computes the PSNR (Peak-Signal-Noise-Ratio) in batch"""
        
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    # norm to [-1, 1]
    img = normalize(img, mean, std)
    img2 = normalize(img2, mean, std)

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
    lpips = net_lpips(img, img2).squeeze(1,2,3)  # batch-wise lpips
    valid_mask = (lpips != 0.)  # do not count if mse is 0; it causes infinity
    # NOTE : there exist zero mse case where the patch is from 
    # val/n03623198/ILSVRC2012_val_00017853.JPEG (all-zero patch)
    lpips = lpips[valid_mask]
    
    return lpips.mean(), valid_mask.sum()  # batch-wise mean
