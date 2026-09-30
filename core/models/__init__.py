import torch

from .resnet import Normalization
from .preact_resnet import preact_resnet
from .resnet import resnet
from .wideresnet import wideresnet

from .preact_resnetwithswish import preact_resnetwithswish
from .wideresnetwithswish import wideresnetwithswish
from .ti_wideresnetwithswish import ti_wideresnetwithswish
from .TransformerModels import VisionTransformer, CONFIGS as VIT_CONFIGS
from .vit_wrapper import ViTWrapper
from .vit_model import VIT_MODELS_2, create_vit_2
from .cait_model import CaiT, CAIT_CONFIGS
from .mamba_model import MAMBA_CONFIGS, create_mamba
from .timm_vit_model import TIMM_VIT_CONFIGS, create_timm_vit

from core.data import DATASETS


MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
          'preact-resnet18', 'preact-resnet34', 'preact-resnet50', 'preact-resnet101',
          'wrn-28-10', 'wrn-32-10', 'wrn-34-10', 'wrn-34-20',
          'preact-resnet18-swish', 'preact-resnet34-swish',
          'wrn-28-10-swish', 'wrn-34-20-swish', 'wrn-70-16-swish',
          'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'ViT-H_14', 'R50-ViT-B_16',
          'ViT-B_16-pt', 'ViT-L_16-pt',
          'CaiT-XXS-12', 'CaiT-XXS-24', 'CaiT-S-12', 'CaiT-S-24',
          'mamba_vision_T', 'mamba_vision_S', 'mamba_vision_B', 'mamba_vision_L', 'mamba_vision_L2']

_DATASET_IMG_SIZE = {
    'cifar10': 32, 'cifar10s': 32,
    'cifar100': 32, 'cifar100s': 32,
    'svhn': 32,
    'tiny-imagenet': 64, 'tiny-imagenets': 64,
}


def create_model(name, normalize, info, device):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        normalize (bool): normalize input.
        info (dict): dataset information.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    vit = False

    if name in VIT_MODELS_2 and info.get('vit_design', 0) == 2:
        backbone = create_vit_2(
            name,
            num_classes=info['num_classes'],
            input_size=int(info.get('vit_input_size', 224)),
        )
        vit = True  # Design 2: attack native input; wrapper resizes to 224 internally.

    elif name in VIT_CONFIGS:
        cfg = VIT_CONFIGS[name]
        patch_size = cfg.patches.size[0] if hasattr(cfg.patches, 'size') else 16
        if patch_size <= 8:
            # Small patch: 64 patches at native 32x32 — no upsampling needed
            img_size = _DATASET_IMG_SIZE.get(info['data'], 32)
            backbone = VisionTransformer(
                config=cfg, img_size=img_size, num_classes=info['num_classes'],
                zero_head=False, vis=False,
            )
        else:
            # Large patch (16, 32): upsample to 96x96 for more patches
            backbone = VisionTransformer(
                config=cfg, img_size=96, num_classes=info['num_classes'],
                zero_head=False, vis=False,
            )
            backbone = ViTWrapper(backbone)
        vit = True

    elif name in CAIT_CONFIGS:
        cfg = CAIT_CONFIGS[name]
        img_size = _DATASET_IMG_SIZE.get(info['data'], 32)
        backbone = CaiT(
            image_size=(img_size, img_size),
            patch_size=cfg['patch_size'],
            num_classes=info['num_classes'],
            dim=cfg['dim'],
            depth=cfg['depth'],
            cls_depth=cfg['cls_depth'],
            heads=cfg['heads'],
            mlp_dim=cfg['mlp_dim'],
            dim_head=cfg['dim_head'],
        )
        # vit stays False: CaiT has no internal normalization,
        # so the standard Normalization wrapper is applied below.

    elif name in TIMM_VIT_CONFIGS:
        backbone = create_timm_vit(name, num_classes=info['num_classes'])
        vit = True  # TimmViTWrapper handles upsampling + normalization internally

    elif name in MAMBA_CONFIGS:
        backbone = create_mamba(
            name,
            num_classes=info['num_classes'],
            input_size=int(info.get('mamba_input_size', 224)),
            pretrained=bool(info.get('mamba_pretrained', False)),
        )
        vit = True  # MambaVisionWrapper handles resizing + normalization internally

    elif info['data'] in ['tiny-imagenet', 'tiny-imagenets']:
        if 'wrn' in name and 'swish' in name:
            backbone = ti_wideresnetwithswish(name, num_classes=info['num_classes'], device=device)
        else:
            from .ti_preact_resnet import ti_preact_resnet
            backbone = ti_preact_resnet(name, num_classes=info['num_classes'], device=device)

    elif info['data'] in DATASETS and info['data'] not in ['tiny-imagenet', 'tiny-imagenets']:
        if 'preact-resnet' in name and 'swish' not in name:
            backbone = preact_resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        elif 'preact-resnet' in name and 'swish' in name:
            backbone = preact_resnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'])
        elif 'resnet' in name and 'preact' not in name:
            backbone = resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        elif 'wrn' in name and 'swish' not in name:
            backbone = wideresnet(name, num_classes=info['num_classes'], device=device)
        elif 'wrn' in name and 'swish' in name:
            backbone = wideresnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'], device=device)
        else:
            raise ValueError('Invalid model name {}!'.format(name))

    else:
        raise ValueError('Models for {} not yet supported!'.format(info['data']))

    # VisionTransformer applies its own internal normalization, so skip the wrapper.
    if normalize and not vit:
        model = torch.nn.Sequential(Normalization(info['mean'], info['std']), backbone)
    else:
        model = torch.nn.Sequential(backbone)

    model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model
