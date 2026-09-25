from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.transforms import functional as TF

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def read_rgb(path):
    with Image.open(path) as im:
        if getattr(im, 'n_frames', 1) != 1:
            raise ValueError('Multi-frame images are unsupported; supply individual frames.')
        return ImageOps.exif_transpose(im).convert('RGB')


def grid_boxes(width, height, crop_size=128):
    if min(width, height) < 3 * crop_size:
        raise ValueError(f'Image {width}x{height} is too small; minimum is 384x384 after EXIF orientation.')
    left = width // 2 - (3 * crop_size) // 2
    top = height // 2 - (3 * crop_size) // 2
    return [(left+c*crop_size, top+r*crop_size,
             left+(c+1)*crop_size, top+(r+1)*crop_size)
            for r in range(3) for c in range(3)]


class Features(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        m = self.backbone
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer1(x)
        a = m.layer2(x)
        b = m.layer3(a)
        a = F.avg_pool2d(a, 3, stride=1, padding=1)
        b = F.avg_pool2d(b, 3, stride=1, padding=1)
        b = F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)
        z = torch.cat([F.normalize(a, dim=1), F.normalize(b, dim=1)], dim=1)
        return F.normalize(z, dim=1)


def inputs(image, branch):
    def tensor(im):
        return TF.normalize(TF.to_tensor(im), [.485,.456,.406], [.229,.224,.225])
    if branch == 'whole':
        return tensor(image.resize((256,256), Image.Resampling.BILINEAR)).unsqueeze(0)
    if branch == 'native':
        return torch.stack([tensor(image.crop(box)) for box in grid_boxes(*image.size)])
    raise ValueError(f'Unknown branch {branch}')


@torch.inference_mode()
def nearest(query, bank, query_chunk=256, bank_chunk=2048):
    # Same direct Euclidean calculation and chunks as the notebook.
    distances = []
    for q in query.split(query_chunk):
        best = torch.full((len(q),), float('inf'), device=q.device)
        for start in range(0, len(bank), bank_chunk):
            d = torch.cdist(q, bank[start:start+bank_chunk],
                            compute_mode='donot_use_mm_for_euclid_dist')
            best = torch.minimum(best, d.min(dim=1).values)
        distances.append(best)
    return torch.cat(distances)


def combine_scores(whole, native, settings):
    raw = {'whole': float(whole), 'native': float(native)}
    z = {b: (s-settings['scalers'][b]['median'])/settings['scalers'][b]['iqr']
         for b,s in raw.items()}
    final = max(z.values())
    if not all(math.isfinite(v) for v in [*raw.values(), *z.values(), final]):
        raise ValueError('Nonfinite model score.')
    threshold = float(settings['thresholds']['fusion'])
    return dict(attack_score=final, predicted_attack=int(final > threshold), threshold=threshold,
                score_whole=raw['whole'], score_native=raw['native'],
                z_whole=z['whole'], z_native=z['native'],
                dominant_branch='whole' if z['whole']>z['native'] else
                                'native' if z['native']>z['whole'] else 'tie')


def validate_settings(settings):
    if settings.get('rule') != 'maximum':
        raise ValueError('Only the evaluated maximum fusion rule is supported.')
    for branch in ['whole','native']:
        scaler=settings['scalers'][branch]
        if not math.isfinite(float(scaler['median'])) or not math.isfinite(float(scaler['iqr'])):
            raise ValueError('Nonfinite scaler.')
        if float(scaler['iqr']) <= 1e-8:
            raise ValueError('IQR too small for stable scaling.')
    if not math.isfinite(float(settings['thresholds']['fusion'])):
        raise ValueError('Nonfinite fusion threshold.')


class PADPredictor:
    def __init__(self, checkpoint, device='cpu', threads=4):
        if threads < 1: raise ValueError('threads must be positive')
        torch.set_num_threads(threads)
        if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device not in {'cpu','cuda'}: raise ValueError('device must be cpu, cuda or auto')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable.')
        self.device = torch.device(device)
        torch.backends.cudnn.benchmark = False
        # Load only the user's trusted exported checkpoint, using restricted loading.
        artifact = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
        if artifact.get('format_version') != 1 or artifact.get('kind') != 'whole_native_max_fusion':
            raise ValueError('Expected the exported whole_native_max_fusion checkpoint.')
        self.settings = artifact['settings']
        validate_settings(self.settings)
        self.model = Features()
        self.model.load_state_dict(artifact['backbone_state'], strict=True)
        self.model = self.model.to(self.device).eval()
        self.banks = {}
        if set(artifact['banks']) != {'whole','native'}:
            raise ValueError('Checkpoint must contain whole and native banks.')
        for branch,bank in artifact['banks'].items():
            if (not isinstance(bank,torch.Tensor) or bank.ndim != 2 or bank.shape[1] != 384
                    or len(bank) < 1 or bank.dtype != torch.float32 or not torch.isfinite(bank).all()):
                raise ValueError(f'Invalid {branch} reference bank.')
            self.banks[branch] = bank.to(self.device)

    @torch.inference_mode()
    def predict(self, path):
        image = read_rgb(path)
        grid_boxes(*image.size)  # Fail before running either branch for unsupported dimensions.
        raw = {}
        for branch in ['whole','native']:
            z = self.model(inputs(image,branch).to(self.device))
            expected = (1,384,32,32) if branch=='whole' else (9,384,16,16)
            if tuple(z.shape) != expected: raise ValueError(f'Unexpected feature shape: {tuple(z.shape)}')
            q = z.permute(0,2,3,1).reshape(-1,384)
            distances = nearest(q,self.banks[branch])
            raw[branch] = float(distances.topk(math.ceil(.10*len(distances))).values.mean().item())
        return dict(**combine_scores(raw['whole'],raw['native'],self.settings),
                    width=image.width,height=image.height)
