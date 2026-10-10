import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import hdf5storage as h5s
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.io import savemat
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from head_monitor import HeadMonitor
from tri_MMdataset0509major import (
    RowLimitedShapeGroupedBatchSampler,
    TriModalCSV,
    tri_modal_collate,
)
from downstream_PHY_adapters_opensource import MultiModalPipeline

SEED = 20240912
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@torch.no_grad()
def dump_group_by_shape(loader,
                        tag      : str,
                        out_root : str = '/data/WiFo_MM/test_multi_modal/task_head/time_seq_dataset/1030'):
    out_root = Path(out_root)
    csi_bk, img_bk, lidar_bk = defaultdict(list), defaultdict(list), defaultdict(list)
    img_len_bk    = defaultdict(list)
    img_off_bk    = defaultdict(list)
    lidar_len_bk  = defaultdict(list)
    lidar_off_bk  = defaultdict(list)

    img_shape_cnt   = defaultdict(Counter)
    lidar_shape_cnt = defaultdict(Counter)

    for batch in tqdm(loader, desc=f'Dump-{tag}', ncols=100):
        if batch is None:
            continue

        Nt_list, Kh_list = batch['Nt_list'], batch['K_half']
        csi, img_seq, lidar_seq = batch['csi'], batch['image_seq'], batch['lidar_seq']
        lidar_len   = batch['lid_len']
        lidar_off   = batch['lid_off']
        img_len     = batch['img_len']
        img_off     = batch['img_off']

        for i in range(Nt_list.size(0)):
            key = f"Nt{int(Nt_list[i])}_Kh{int(Kh_list[i])}"

            re = csi[i, 0].cpu().numpy().astype(np.float64)
            im = csi[i, 1].cpu().numpy().astype(np.float64)
            csi_bk[key].append(re + 1j * im)

            if img_seq is not None:
                img_np = img_seq[i].cpu().numpy()
                T, _, H, W = img_np.shape
                img_bk[key].append(img_np)
                img_len_bk[key].append(int(img_len[i]))
                img_off_bk[key].append(int(img_off[i]))
                img_shape_cnt[key][(T, H, W)] += 1

            if lidar_seq is not None:
                lidar_np = lidar_seq[i].cpu().numpy()
                T, P, _  = lidar_np.shape
                lidar_bk[key].append(lidar_np)
                lidar_len_bk[key].append(int(lidar_len[i]))
                lidar_off_bk[key].append(int(lidar_off[i]))
                lidar_shape_cnt[key][(T, P)] += 1

    print('\n====== Size-consistency report ======')
    for key, lst in csi_bk.items():
        N = len(lst)

        if key in img_shape_cnt:
            c = img_shape_cnt[key]
            most_common = c.most_common(1)[0][1]
            mismatch_img = N - most_common
        else:
            mismatch_img = 0

        if key in lidar_shape_cnt:
            c = lidar_shape_cnt[key]
            most_common = c.most_common(1)[0][1]
            mismatch_lidar = N - most_common
        else:
            mismatch_lidar = 0

        print(f'{key:15s} | total {N:4d}  '
              f'img mismatch {mismatch_img:3d}  '
              f'lidar mismatch {mismatch_lidar:3d}')

    out_root.mkdir(parents=True, exist_ok=True)

    def save_mat(path: Path, **arrays):
        savemat(path, arrays, do_compression=True)
        shapes = ', '.join([f'{k}:{v.shape}' for k, v in arrays.items()])
        print(f'✓ {path.name:35s} {shapes}')

    for key, lst in csi_bk.items():
        arr = np.stack(lst, 0).astype(np.complex128)
        save_mat(out_root / f'{tag}_{key}_csi.mat', csi=arr)

    for key, lst in img_bk.items():
        if not lst:
            continue
        Tmax = max(a.shape[0] for a in lst)
        C    = 3
        Hmax = max(a.shape[2] for a in lst)
        Wmax = max(a.shape[3] for a in lst)
        dtype = lst[0].dtype

        N = len(lst)
        img_arr = np.zeros((N, Tmax, C, Hmax, Wmax), dtype=dtype)
        for n, a in enumerate(lst):
            T, _, H, W = a.shape
            img_arr[n, :T, :, :H, :W] = a

        img_len_arr = np.asarray(img_len_bk[key],  dtype=np.int32)
        img_off_arr = np.asarray(img_off_bk[key],  dtype=np.int32)

        save_mat(out_root / f'{tag}_{key}_img.mat',
                 img=img_arr, img_len=img_len_arr, img_off=img_off_arr)

    for key, lst in lidar_bk.items():
        if not lst:
            continue
        Tmax = max(a.shape[0] for a in lst)
        Pmax = max(a.shape[1] for a in lst)
        dtype = lst[0].dtype

        N = len(lst)
        lidar_arr = np.zeros((N, Tmax, Pmax, 4), dtype=dtype)
        for n, a in enumerate(lst):
            T, P, _ = a.shape
            lidar_arr[n, :T, :P] = a

        lidar_len_arr = np.asarray(lidar_len_bk[key], dtype=np.int32)
        lidar_off_arr = np.asarray(lidar_off_bk[key], dtype=np.int32)

        save_mat(out_root / f'{tag}_{key}_lidar.mat',
                 lidar=lidar_arr, lidar_len=lidar_len_arr, lidar_off=lidar_off_arr)

def _stratified_split(
    dataset: "TriModalCSV",
    ratio  : float = 0.8,
    seed   : int   = 42):
    rng = random.Random(seed)

    buckets: Dict[Tuple[int,int], List[int]] = defaultdict(list)
    for idx, (nt, k) in enumerate(zip(dataset._Nts,
                                      [kk // 2 for kk in dataset._Ks])):
        buckets[(nt, k)].append(idx)

    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be between 0 and 1.")

    ft_indices, val_indices = [], []
    for indices in buckets.values():
        rng.shuffle(indices)
        n_ft = max(1, int(len(indices) * ratio))
        if len(indices) > 1:
            n_ft = min(n_ft, len(indices) - 1)
        ft_indices.extend(indices[:n_ft])
        val_indices.extend(indices[n_ft:])

    rng.shuffle(ft_indices)
    rng.shuffle(val_indices)
    return ft_indices, val_indices

def print_trainable(model):
    tot = 0
    print(f'{"name":60s}  {"shape":>20s}  {"#param":>12s}')
    print('-' * 100)
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        shape_str = str(tuple(p.shape))
        cnt       = p.numel()
        print(f'{n:<60s}  {shape_str:>20s}  {cnt:>12,d}')
        tot += cnt
    print('-' * 100)
    print(f'Total trainable parameters: {tot:,d}')

def _make_loader_from_indices(
    dataset     : "TriModalCSV",
    indices     : List[int],
    batch_size  : int,
    shuffle     : bool,
    max_row     : int = 2,
    num_workers : int = 8
):
    from torch.utils.data import Subset, DataLoader
    subset  = Subset(dataset, indices)
    sampler = RowLimitedShapeGroupedBatchSampler(
        shapes=[dataset.shapes[i] for i in indices],
        rows  =[dataset.rows[i]   for i in indices],
        bs=batch_size,
        max_per_row=max_row,
        shuffle=shuffle
    )
    return DataLoader(subset,
                      batch_sampler=sampler,
                      num_workers=num_workers,
                      pin_memory=True,
                      collate_fn=tri_modal_collate)

monitor = HeadMonitor()
monitor.reset()
def power_norm(x: torch.Tensor, eps: float = 1e-12):
    if x.dim() < 3:
        return x

    axes = [i for i in range(1, x.dim()) if x.shape[i] > 1]

    if not axes:
        return x

    power = (x ** 2).mean(dim=axes, keepdim=True)
    scale = torch.sqrt(power + eps)
    return x / scale


def _too_small_batch(b):
    return b is None or len(b['Nt_list']) <= 1


def beam_ce_loss(logits, label):
    criterion = nn.MultiLabelSoftMarginLoss()
    loss = criterion(logits, label)
    return loss

def asym_mse(nh,nt,θ=.3):
    w=torch.where(nh>nt,θ,1-θ)
    return ((w*(nh-nt))**2).mean()
def tv(x): return ((x[:,:,:-1]-x[:,:,1:])**2).mean()+((x[:,:,:,:-1]-x[:,:,:,1:])**2).mean()

class FCDAMP_Loss(nn.Module):
    def __init__(self,α=.4,β=.2): super().__init__(); self.α=α; self.β=β
    def forward(self,h_est,h_gt,n_hat,n_true):
        mse=F.mse_loss(h_est,h_gt)
        nmse=mse/(h_gt.pow(2).mean()+1e-12)
        l2=tv(n_hat)
        return nmse, nmse

def topk_soft_ce_loss(
    logits       : torch.Tensor,
    topk_indices : torch.Tensor,
    Nt_list      : torch.Tensor = None,
    reduction: str = 'mean',
    eps: float = 1e-12
):
    B, C = logits.shape
    device = logits.device
    k = topk_indices.size(1)

    target = torch.zeros(B, C, device=device, dtype=logits.dtype)
    target.scatter_(1, topk_indices, 1.0 / k)

    if Nt_list is not None:
        col = torch.arange(C, device=device)[None, :]
        mask = col >= Nt_list[:, None]
        logits = logits.masked_fill(mask, 0.0)
        target = target.masked_fill(mask, 0.0)

    log_prob = F.log_softmax(logits, dim=1)
    ce = -(target * log_prob).sum(dim=1)

    if reduction == 'mean':
        return ce.mean()
    elif reduction == 'sum':
        return ce.sum()
    else:
        return ce


def export_csi_img_raw(
        loader,
        dataset,
        out_dir: str,
        tag: str):
    os.makedirs(out_dir, exist_ok=True)
    buckets_csi = defaultdict(list)
    buckets_img = defaultdict(list)
    buckets_lidar = defaultdict(list)
    for b in tqdm(loader, desc=f'Export-{tag}', ncols=100):
        if b is None:
            continue
        Nt_l   = b['Nt_list']
        Kh_l   = b['K']
        rows   = b['row']
        csi = b['csi']
        lidar   = b['lidar_pn']
        B = csi.size(0)
        for i in range(csi.size(0)):
            Nt  = int(Nt_l[i]); K = int(Kh_l[i])
            key = f'Nt{Nt}_K{K}'

            real = csi[i, 0].cpu().numpy().astype(np.float64)
            imag = csi[i, 1].cpu().numpy().astype(np.float64)
            buckets_csi[key].append(real + 1j * imag)

            img_path = dataset.df.iloc[int(rows[i])].image
            if img_path != '~' and os.path.isfile(img_path):
                img_rgb = Image.open(img_path).convert('RGB')
                arr = np.asarray(img_rgb, dtype=np.uint8)
                arr = arr.transpose(2,0,1)
                buckets_img[key].append(arr)
            buckets_lidar[key].append(lidar[i].cpu().numpy())

    for key, arr_list in buckets_csi.items():
        csi_arr = np.stack(arr_list, 0)
        save_path = os.path.join(out_dir, f'{tag}_{key}_csi.mat')

    for key, lst in buckets_lidar.items():
        lidar_arr = np.stack(lst, 0)
        h5s.savemat(os.path.join(out_dir, f'{tag}_{key}_lidar.mat'),
                    {'lidar_pn': lidar_arr}, format='7.3', compress=True)
        print(f'✓ {tag}_{key}_lidar.mat  {lidar_arr.shape}')


def validate(pipe, dataset, val_indices, epoch, batch_size: int=64, save_dir: str='/data/WiFo_MM/test_multi_modal/task_head/vis', subset_ratio: float=1.0, log_collector=None, loss_obj=None):
    subset = Subset(dataset, val_indices)
    sampler = RowLimitedShapeGroupedBatchSampler(shapes=[dataset.shapes[i] for i in val_indices], rows=[dataset.rows[i] for i in val_indices], bs=batch_size, max_per_row=2, shuffle=False)
    loader = DataLoader(subset, batch_sampler=sampler, num_workers=8, pin_memory=True, collate_fn=tri_modal_collate)
    pipe.eval()
    tot = n = tot_nmse = 0
    loc_bucket = defaultdict(lambda: {'n': 0, 'sq_err': 0.0, 'abs_err': 0.0, 'mse_loss': 0.0})
    loc_mse_sum = 0.0
    loc_dist_err_sum = 0.0
    loc_sq_dist_sum = 0.0
    loc_samples_cnt = 0
    correct1 = correct3 = correct5 = 0
    max_batches = max(1, int(len(loader) * subset_ratio))
    pbar = tqdm(enumerate(loader), total=max_batches, desc=f'Val epoch {epoch}', leave=False)
    acc_bucket = defaultdict(lambda: {'n': 0, 'c1': 0, 'c3': 0, 'c5': 0})
    ce_bucket = defaultdict(lambda: {'n': 0, 'mse': 0.0, 'pwr': 0.0})
    with torch.no_grad():
        for batch_idx, b in pbar:
            if _too_small_batch(b):
                print('Skip batch: only 1 sample in current batch, continue.')
                continue
            if batch_idx >= max_batches:
                break
            img_seq = b['image_seq'].to(device)
            img_len = b['img_len'].to(device)
            img_off = b['img_off'].to(device)
            lidar_seq = b['lidar_seq'].to(device)
            lid_len = b['lid_len'].to(device)
            lid_off = b['lid_off'].to(device)
            Nt_list = b['Nt_list'].to(device)
            is_ula_indicator = b['is_ula'].to(device)
            if CFG['TASK'] == 'beam' or CFG['TASK'] == 'beam_sm':
                csi = None
                lidar_seq = None
                Kh_list = b['K_half'].to(device)
                K_list = b['K'].to(device)
                logits = pipe(csi=csi, img_seq=img_seq, img_len=img_len, img_off=img_off, lidar_seq=lidar_seq, lid_len=lid_len, lid_off=lid_off, Nt_list=Nt_list, Kh_list=Kh_list, K_list=K_list)
                target = b['beam_label'].to(device)
                loss = topk_soft_ce_loss(logits, target, Nt_list)
                topk_idx = logits.topk(5, dim=1).indices

                def _hit_at_k(k: int) -> torch.Tensor:
                    pred = topk_idx[:, :k].unsqueeze(-1)
                    target = b['beam_label'].to(device).unsqueeze(1)
                    return (pred == target).any(dim=2).any(dim=1)
                hit1 = _hit_at_k(1)
                hit3 = _hit_at_k(3)
                hit5 = _hit_at_k(5)
                key = f'Nt{int(Nt_list[0])}_Kh{int(K_list[0])}'
                bucket = acc_bucket[key]
                bs = Nt_list.size(0)
                bucket['n'] += bs
                bucket['c1'] += hit1.sum().item()
                bucket['c3'] += hit3.sum().item()
                bucket['c5'] += hit5.sum().item()
                correct1 += hit1.sum().item()
                correct3 += hit3.sum().item()
                correct5 += hit5.sum().item()
            elif CFG['TASK'] == 'CE_mm' or CFG['TASK'] == 'CE_sm':
                lidar_arg = None
                img_arg = None
                csi_in = b['csi'].to(device)
                Nt_list = b['Nt_list'].to(device)
                K_list = b['K'].to(device)
                Kh_list = b['K_half'].to(device)
                logits, label = pipe(csi=csi_in, ula_indicator=is_ula_indicator, img_seq=img_seq, img_len=img_len, img_off=img_off, lidar_seq=lidar_seq, lid_len=lid_len, lid_off=lid_off, Nt_list=Nt_list, Kh_list=Kh_list, K_list=K_list, CE_CFG=CE_CFG)
                mse_all = F.mse_loss(logits, label, reduction='none')
                mse_s = mse_all.mean(dim=(1, 2, 3))
                pwr_s = (label ** 2).mean(dim=(1, 2, 3))
                nmse_s = mse_s / (pwr_s + 1e-12)
                loss = mse_s.mean()
                for i in range(Nt_list.size(0)):
                    key = f'Nt{int(Nt_list[i])}_K{int(K_list[i])}'
                    ce_bucket[key]['n'] += 1
                    ce_bucket[key]['mse'] += mse_s[i].item()
                    ce_bucket[key]['pwr'] += pwr_s[i].item()
            elif CFG['TASK'] == 'CE_lamp' or CFG['TASK'] == 'CE_lamp_sm' or CFG['TASK'] == 'CE_lamp_mm':
                Nt_list = b['Nt_list'].to(device)
                K_list = b['K'].to(device)
                h_est, h_gt, n_hat, n_true = pipe(csi=b['csi'].to(device), ula_indicator=is_ula_indicator, img_seq=img_seq, img_len=img_len, img_off=img_off, lidar_seq=lidar_seq, lid_len=lid_len, lid_off=lid_off, Nt_list=Nt_list, Kh_list=b['K_half'].to(device), K_list=b['K'].to(device), CE_CFG=CE_CFG)
                B, _, antenna = h_est.shape
                loss, nmse_batch = loss_obj(h_est, h_gt, n_hat, n_true)
                mse_s = (h_est - h_gt).pow(2).mean(dim=(1, 2))
                pwr_s = h_gt.pow(2).mean(dim=(1, 2))
                nmse_s = mse_s / (pwr_s + 1e-12)
                for i in range(Nt_list.size(0)):
                    key = f"Nt{int(Nt_list[i])}_K{int(b['K'][i])}"
                    ce_bucket[key]['n'] += 1
                    ce_bucket[key]['mse'] += mse_s[i].item()
                    ce_bucket[key]['pwr'] += pwr_s[i].item()
            if CFG['TASK'].startswith('loc_'):
                csi_input = b['csi']
                csi_forloc = power_norm(csi_input[:, :, 0]).squeeze(2)
                logits = pipe(csi=csi_forloc.to(device), img_seq=img_seq, img_len=img_len, img_off=img_off, lidar_seq=lidar_seq, lid_len=lid_len, lid_off=lid_off, Nt_list=Nt_list, K_list=b['K'].to(device))
                target = b['loc_label'].to(device)
                loss = F.mse_loss(logits, target, reduction='mean')
                loc_mean = dataset.loc_mean.to(device)
                loc_std = dataset.loc_std.to(device)
                pred_raw = logits * loc_std + loc_mean
                target_raw = b['raw_loc'].to(device)
                bs = logits.size(0)
                dist_err = torch.norm(pred_raw - target_raw, dim=1).sum().item()
                sq_dist_err = torch.sum((pred_raw - target_raw) ** 2).item()
                key = f"{int(Nt_list[0])}_{int(b['K'][0])}_loc"
                loc_bucket[key]['n'] += bs
                loc_bucket[key]['sq_err'] += sq_dist_err
                loc_bucket[key]['abs_err'] += dist_err
                loc_bucket[key]['mse_loss'] += loss.item() * bs
                loc_mse_sum += loss.item() * bs
                loc_dist_err_sum += dist_err
                loc_sq_dist_sum += sq_dist_err
                loc_samples_cnt += bs
                tot += loss.item() * bs
                n += bs
                monitor.update('val', key, bs, loss.item())
                continue
            bs = Nt_list.size(0)
            tot += loss.item() * bs
            n += bs
            if CFG['TASK'] == 'beam' or CFG['TASK'] == 'beam_sm':
                key = str(Nt_list[0]) + str(K_list[0]) + 'beam'
            elif CFG['TASK'] == 'CE_sm' or CFG['TASK'] == 'CE_mm' or CFG['TASK'] == 'CE_lamp' or (CFG['TASK'] == 'CE_lamp_sm') or (CFG['TASK'] == 'CE_lamp_mm'):
                key = str(Nt_list[0]) + str(K_list[0]) + 'CE'
            monitor.update('val', key, Nt_list.size(0), loss.item())
            pbar.set_postfix(loss=f'{tot / max(1, n):.4f}')
    pbar.close()
    msg = f'Val  loss={tot / n:.4f}'
    if CFG['TASK'] == 'beam' or CFG['TASK'] == 'beam_sm':
        denom = n
        msg += f'  Top-1/3/5 = {correct1 / denom:.3%} / {correct3 / denom:.3%} / {correct5 / denom:.3%}'
        print('\nPer-shape accuracy')
        for key, d in acc_bucket.items():
            cnt = d['n']
            t1 = d['c1'] / cnt
            t3 = d['c3'] / cnt
            t5 = d['c5'] / cnt
            print(f'{key:<12s}  N={cnt:4d}   Top1 {t1:.3%}   Top3 {t3:.3%}   Top5 {t5:.3%}')
    print(CFG['TASK'])
    if CFG['TASK'] in ('CE_sm', 'CE_mm', 'CE_lamp', 'CE_lamp_mm', 'CE_lamp_sm'):
        print('\nPer-shape NMSE (dB)')
        for key, d in ce_bucket.items():
            mse = d['mse'] / d['n']
            pwr = d['pwr'] / d['n']
            nmse_db = 10 * math.log10(mse / (pwr + 1e-12))
            print(f"{key:<12s}  N={d['n']:4d}   NMSE={nmse_db:7.3f} dB")
        tot_nmse = sum((d['mse'] for d in ce_bucket.values()))
        tot_pwr = sum((d['pwr'] for d in ce_bucket.values()))
        msg += f'  NMSE={10 * math.log10(tot_nmse / (tot_pwr + 1e-12)):.3f} dB'
    if CFG['TASK'].startswith('loc_'):
        print('\n' + '=' * 50)
        print(f"{'Dataset Key':<20} | {'N':<6} | {'RMSE (m)':<10} | {'MDE (m)':<10}")
        print('-' * 50)
        for k, d in loc_bucket.items():
            n_samples = d['n']
            rmse = math.sqrt(d['sq_err'] / n_samples)
            mde = d['abs_err'] / n_samples
            print(f'{k:<20} | {n_samples:<6} | {rmse:<10.2f} | {mde:<10.2f}')
        print('=' * 50)
        denom = max(1, loc_samples_cnt)
        mean_dist = loc_dist_err_sum / denom
        phys_rmse = math.sqrt(loc_sq_dist_sum / denom)
        msg += f'  [Global] RMSE = {phys_rmse:.2f}m, MDE = {mean_dist:.2f}m'
    print(msg)
    monitor.report(epoch=epoch)

    def _acc_dict_to_pct(bkt: dict):
        out = {}
        for key, d in bkt.items():
            n = d['n']
            out[key] = dict(N=n, Top1=d['c1'] / n, Top3=d['c3'] / n, Top5=d['c5'] / n)
        return out

    def _ce_dict_to_nmse(bkt: dict):
        out = {}
        for key, d in bkt.items():
            n = d['n']
            mse = d['mse'] / n
            pwr = d['pwr'] / n
            nmse = mse / (pwr + 1e-12)
            out[key] = dict(N=n, MSE=mse, PWR=pwr, NMSE=nmse, NMSEdB=10 * math.log10(nmse + 1e-12))
        return out
    out = dict(epoch=epoch, task=CFG['TASK'], global_msg=msg, per_shape=_acc_dict_to_pct(acc_bucket) if CFG['TASK'] in ('beam', 'beam_sm') else _ce_dict_to_nmse(ce_bucket) if CFG['TASK'] in ('CE_sm', 'CE_mm', 'CE_lamp') else {})
    if log_collector is not None:
        log_collector.append(out)
    return out

def build_optimizer(model, base_lr=1e-4, wd=5e-5):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 1 or 'bias' in n or 'norm' in n:
            no_decay.append(p)
        else:
            decay.append(p)

    optim = torch.optim.AdamW(
        [
            {'params': decay,     'weight_decay': wd},
            {'params': no_decay,  'weight_decay': 0.0},
        ],
        lr=base_lr,
        betas=(0.9, 0.98),
        eps=1e-8
    )
    return optim

def build_scheduler(optim, total_epochs=None, warmup_epochs=0):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda, last_epoch=-1
    )

CFG = dict(
    VAL_CSV = '/data/WiFo_MM/Dataset_4_pretrain/val_data_item.csv',  # Path to the released validation CSV.
    GRID    = (496, 432),
    CKPT_IMG_CSI   = './WiFo_M2_Img.pth',
    CKPT_LIDAR_CSI = './WiFo_M2_LiDAR.pth',
    TASK       = 'loc_sm',  # Select the downstream task to evaluate.
    FT_RATIO =  0.8,  # Fraction of VAL_CSV used for fine-tuning; the rest is used for validation.
    FT_EPOCHS  = 50,
    BATCH_FT   = 32,
    LR         = 1e-4,
    DEVICE     = 'cuda',
)

CE_CFG = dict(
    K_RF      = 4,
    B_bits    = 3,
    T_repeat  = 8,
    P_max     = 1.0,
    SNR_dB    = 15
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu  = torch.cuda.device_count()
print(f'>>> using {n_gpu} GPU(s)')

def _sub(m):
    return m.module if isinstance(m, nn.DataParallel) else m

def build_ft_val_data(
    val_csv : str,
    grid    : Tuple[int,int],
    ratio   : float,
    bs      : int,
    max_row : int = 2,
    seed    : int = 42
):
    dataset = TriModalCSV(val_csv, grid_size=grid, min_power=1e-10)
    ft_indices, val_indices = _stratified_split(dataset, ratio=ratio, seed=seed)
    ft_loader = _make_loader_from_indices(
        dataset, ft_indices, bs, shuffle=True, max_row=max_row
    )
    return ft_loader, dataset, val_indices

def main():
    ft_loader, dataset, val_indices = build_ft_val_data(
        CFG['VAL_CSV'], CFG['GRID'], CFG['FT_RATIO'], CFG['BATCH_FT']
    )

    unique_nt_kh_pairs = dataset.get_unique_nt_kh_pairs()
    print(f'>>> Unique (Nt, Kh) pairs: {unique_nt_kh_pairs}')

    return dataset, unique_nt_kh_pairs, ft_loader, val_indices

def run_one(dataset, unique_nt_kh_pairs, ft_loader, val_indices, seed_exp, run_id=0):
    val_history = []
    task_head_samples = Counter(((nt, kh) for nt, kh in dataset.get_unique_nt_kh_pairs()))
    print('Task Head Sample Distribution:', task_head_samples)
    random.seed(seed_exp)
    np.random.seed(seed_exp)
    torch.manual_seed(seed_exp)
    torch.cuda.manual_seed_all(seed_exp)
    pipe = MultiModalPipeline(encoders_ckpt={'img': CFG['CKPT_IMG_CSI'], 'lidar': CFG['CKPT_LIDAR_CSI']}, unique_nt_kh_pairs=unique_nt_kh_pairs, Grid=CFG['GRID'], task=CFG['TASK'], device=device, seed_ce=seed_exp, CE_CFG=CE_CFG).to(device)
    if n_gpu > 1:
        pipe = nn.DataParallel(pipe)
    encoders = []
    if CFG['TASK'] == 'beam' or CFG['TASK'] == 'beam_sm' or CFG['TASK'] == 'CE_sm' or (CFG['TASK'] == 'CE_mm') or (CFG['TASK'] == 'CE_lamp') or (CFG['TASK'] == 'CE_lamp_sm') or (CFG['TASK'] == 'CE_lamp_mm') or (CFG['TASK'] == 'loc_pure') or (CFG['TASK'] == 'loc_sm') or (CFG['TASK'] == 'loc_mm'):
        _sub(pipe).img_enc.eval()
        _sub(pipe).lidar_enc.eval()
    else:
        for mod in (_sub(pipe).img_enc, _sub(pipe).lidar_enc):
            encoders += list(mod.parameters())
    head_params = _sub(pipe).heads.parameters()
    if CFG['TASK'] == 'beam' or CFG['TASK'] == 'beam_sm':
        enc_lr = 2e-05
        others_lr = 2e-05
        wd = 5e-05
    elif CFG['TASK'] == 'CE_mm' or CFG['TASK'] == 'CE_sm':
        enc_lr = 1e-04
        others_lr = 1e-04
        wd = 5e-05
    elif CFG['TASK'] == 'CE_lamp' or CFG['TASK'] == 'CE_lamp_sm' or CFG['TASK'] == 'CE_lamp_mm':
        enc_lr = 2e-05
        others_lr = 2e-05
        wd = 5e-05
    elif CFG['TASK'] == 'loc_pure' or CFG['TASK'] == 'loc_sm' or CFG['TASK'] == 'loc_mm':
        enc_lr = 2e-05
        others_lr = 2e-05
        wd = 2e-05
    if CFG['TASK'] == 'beam' or CFG['TASK'] == 'CE_mm' or CFG['TASK'] == 'CE_lamp' or (CFG['TASK'] == 'CE_lamp_sm') or (CFG['TASK'] == 'CE_lamp_mm') or (CFG['TASK'] == 'CE_sm') or (CFG['TASK'] == 'beam_sm') or (CFG['TASK'] == 'beam_sm') or (CFG['TASK'] == 'loc_pure') or (CFG['TASK'] == 'loc_sm') or (CFG['TASK'] == 'loc_mm'):
        optim = torch.optim.AdamW([{'params': encoders, 'lr': enc_lr, 'weight_decay': wd}, {'params': list(head_params), 'lr': others_lr, 'weight_decay': wd}], betas=(0.9, 0.98), eps=1e-08)
        sched = build_scheduler(optim, total_epochs=CFG['FT_EPOCHS'])
    model = pipe.module if isinstance(pipe, torch.nn.DataParallel) else pipe
    print_trainable(model)
    lamp_loss = FCDAMP_Loss()
    validate(pipe, dataset=dataset, val_indices=val_indices, epoch=1, batch_size=8, subset_ratio=0.2, loss_obj=lamp_loss)
    for ep in range(1, CFG['FT_EPOCHS'] + 1):
        cur_lr = sched.get_last_lr()[0]
        print(f'>>> Epoch {ep:03d}  lr = {cur_lr:.6e}')
        _sub(pipe).heads.train()
        tot = n = 0
        pbar = tqdm(ft_loader, desc=f"Finetune {ep}/{CFG['FT_EPOCHS']}", ncols=100)
        for b in pbar:
            if _too_small_batch(b):
                print('Skip batch: only 1 sample in current batch, continue.')
                continue
            img_seq = b['image_seq'].to(device)
            img_len = b['img_len'].to(device)
            img_off = b['img_off'].to(device)
            lidar_seq = b['lidar_seq'].to(device)
            lid_len = b['lid_len'].to(device)
            lid_off = b['lid_off'].to(device)
            is_ula_indicator = b['is_ula'].to(device)
            if CFG['TASK'] == 'beam' or CFG['TASK'] == 'beam_sm':
                lidar_seq = None
                csi = None
                Nt_list = b['Nt_list'].to(device)
                Kh_list = b['K_half'].to(device)
                K_list = b['K'].to(device)
                logits = pipe(csi=csi, img_seq=img_seq, img_len=img_len, img_off=img_off, lidar_seq=lidar_seq, lid_len=lid_len, lid_off=lid_off, Nt_list=Nt_list, Kh_list=Kh_list, K_list=K_list)
                loss = topk_soft_ce_loss(logits, b['beam_label'].to(device), Nt_list)
            elif CFG['TASK'] == 'CE_mm' or CFG['TASK'] == 'CE_sm':
                lidar_arg = None
                img_arg = None
                csi_in = b['csi'].to(device)
                Nt_list = b['Nt_list'].to(device)
                K_list = b['K'].to(device)
                Kh_list = b['K_half'].to(device)
                logits, label = pipe(csi=csi_in, ula_indicator=is_ula_indicator, img_seq=img_seq, img_len=img_len, img_off=img_off, lidar_seq=lidar_seq, lid_len=lid_len, lid_off=lid_off, Nt_list=Nt_list, Kh_list=Kh_list, K_list=K_list, CE_CFG=CE_CFG)
                loss = F.mse_loss(logits, label)
            elif CFG['TASK'] == 'CE_lamp' or CFG['TASK'] == 'CE_lamp_sm' or CFG['TASK'] == 'CE_lamp_mm':
                csi_in = b['csi'].to(device)
                Nt_list = b['Nt_list'].to(device)
                K_list = b['K'].to(device)
                Kh_list = b['K_half'].to(device)
                logits, h_gt, n_hat, n_true = pipe(csi=csi_in, ula_indicator=is_ula_indicator, img_seq=img_seq, img_len=img_len, img_off=img_off, lidar_seq=lidar_seq, lid_len=lid_len, lid_off=lid_off, Nt_list=Nt_list, Kh_list=Kh_list, K_list=K_list, CE_CFG=CE_CFG)
                loss, nmse = lamp_loss(logits, h_gt, n_hat, n_true)
            elif CFG['TASK'].startswith('loc_'):
                Nt_list = b['Nt_list'].to(device)
                K_list = b['K'].to(device)
                Kh_list = b['K_half'].to(device)
                csi_in = b['csi'].to(device)
                csi_input = power_norm(csi_in[:, :, 0]).squeeze(2)
                logits = pipe(csi=csi_input, img_seq=img_seq, img_len=img_len, img_off=img_off, lidar_seq=lidar_seq, lid_len=lid_len, lid_off=lid_off, Nt_list=b['Nt_list'].to(device), K_list=b['K'].to(device))
                target = b['loc_label'].to(device)
                loss = F.mse_loss(logits, target)
            else:
                raise ValueError(f"Unknown TASK {CFG['TASK']}")
            optim.zero_grad()
            loss.backward()
            optim.step()
            if CFG['TASK'] == 'beam' or CFG['TASK'] == 'beam_sm':
                key = str(Nt_list[0]) + str(K_list[0]) + 'beam'
            elif CFG['TASK'] == 'CE_sm' or CFG['TASK'] == 'CE_mm' or CFG['TASK'] == 'CE_lamp' or (CFG['TASK'] == 'CE_lamp_sm') or (CFG['TASK'] == 'CE_lamp_mm'):
                key = str(Nt_list[0]) + str(K_list[0]) + 'CE'
            elif CFG['TASK'] == 'loc_pure':
                key = str(Nt_list[0]) + str(Kh_list[0]) + 'loc_pure'
            elif CFG['TASK'] == 'loc_sm':
                key = str(Nt_list[0]) + str(Kh_list[0]) + 'loc_sm'
            elif CFG['TASK'] == 'loc_mm':
                key = str(Nt_list[0]) + str(Kh_list[0]) + 'loc_mm'
            monitor.update('train', key, logits.size(0), loss.item())
            bs = logits.size(0)
            tot += loss.item() * bs
            n += bs
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        sched.step()
        print(f'  ↳ epoch {ep} avg loss {tot / n:.4f}')
        monitor.report(epoch=ep)
        if ep > 5 and ep % 5 == 0:
            validate(pipe, dataset=dataset, val_indices=val_indices, epoch=ep, batch_size=8, subset_ratio=1, log_collector=val_history, loss_obj=lamp_loss)
    log_path = f'/data/WiFo_MM/test_multi_modal/task_head/logs_1104/CE/CE_sm_lidar/logs_val_run{run_id}.txt'
    with open(log_path, 'w') as f:
        for rec in val_history:
            f.write(f"Epoch {rec['epoch']:03d}  {rec['global_msg']}\n")
            ps = rec['per_shape']
            if not ps:
                f.write('\n')
                continue
            if CFG['TASK'] in ('beam', 'beam_sm'):
                for k, d in ps.items():
                    f.write(f"    {k:<15s} N={d['N']:4d} Top1 {d['Top1']:.3%}  Top3 {d['Top3']:.3%}  Top5 {d['Top5']:.3%}\n")
            elif CFG['TASK'] in ('CE_sm', 'CE_mm'):
                for k, d in ps.items():
                    f.write(f"    {k:<15s} N={d['N']:4d} MSE {d['MSE']:.4e}  PWR {d['PWR']:.4e}  NMSE {d['NMSE']:.4e}  NMSEdB {d['NMSEdB']:+7.3f} dB\n")
            f.write('\n')
    print(f'✓ validation log saved to {log_path}')

if __name__ == '__main__':
    dataset, unique_nt_kh_pairs, ft_loader, val_indices = main()
    BASE_SEED = 20251126
    for run in range(1):
        print(f'\n========  {CFG["TASK"]} experiment RUN {run}  ========')
        run_one(
            dataset,
            unique_nt_kh_pairs,
            ft_loader,
            val_indices,
            seed_exp=BASE_SEED + run,
            run_id=run,
        )
