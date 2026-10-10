import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from einops import rearrange

import ResNet_Code as IE_res34
from heads_lamp import FCDAMPHead


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim=256, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ImageWrapper_res34(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.backbone = IE_res34.ImageFeatureNet(out_channels=256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = ProjectionMLP(in_dim=256, out_dim=out_dim)

    def forward(self, img):
        feat = self.backbone(img)
        feat = self.gap(feat).flatten(1)
        return self.proj(feat)


class SeqImageEncoder(nn.Module):
    """Encode variable-length image sequences into per-frame features."""
    def __init__(self, out_dim=512, gru_h=256):
        super().__init__()
        self.img_backbone = ImageWrapper_res34(out_dim=out_dim)
        self.gru = nn.GRU(out_dim, gru_h, batch_first=True)
        self.proj = nn.Linear(gru_h, out_dim)

    def forward(self, imgs, lens):
        B, T_max = imgs.shape[:2]
        imgs = imgs.view(B * T_max, *imgs.shape[2:])
        feat = self.img_backbone(imgs)
        feat = feat.view(B, T_max, -1).contiguous()

        # Zero padded frame features before packing the sequence.
        mask = torch.arange(T_max, device=lens.device)[None] >= lens[:, None]
        feat[mask] = 0.0
        packed = nn.utils.rnn.pack_padded_sequence(
            feat, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=T_max)
        out = F.normalize(self.proj(out), dim=-1)
        return out


class PointNetBackbone(nn.Module):
    def __init__(self, C=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, C, 1),
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, pts):
        x = self.net(pts.transpose(1, 2)).squeeze(-1)
        return x


class SeqLiDAREncoder(nn.Module):
    """Encode LiDAR sequences and predict one feature vector per offset."""
    def __init__(self, C=256, H=256, offs=10):
        super().__init__()
        self.C = C
        self.backbone = PointNetBackbone(C)
        self.gru = nn.GRU(C, H, batch_first=True)
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, C))
             for _ in range(offs)]
        )

    def forward(self, pcs_tensor, lens):
        B, T_max = pcs_tensor.size(0), pcs_tensor.size(1)
        dev = pcs_tensor.device
        feat_batches = []
        # Extract features only for valid frames, then pad for sequence packing.
        for b in range(B):
            L = lens[b]
            pcs_valid = pcs_tensor[b, :L]
            feats = self.backbone(pcs_valid)
            if L < T_max:
                pad = torch.zeros(T_max - L, self.C, device=dev)
                feats = torch.cat([feats, pad], 0)
            feat_batches.append(feats)

        feats = torch.stack(feat_batches, 0)
        lens_cpu = lens.cpu().long()
        packed = pack_padded_sequence(feats, lens_cpu,
                                      batch_first=True, enforce_sorted=False)
        _, hT = self.gru(packed)
        hT = hT.squeeze(0)
        zs = torch.stack([head(hT) for head in self.heads], 1)
        return zs


def eye_like(A: torch.Tensor) -> torch.Tensor:
    n   = A.size(-1)
    eye = torch.eye(n, dtype=A.dtype, device=A.device)

    if A.dim() == 2:
        return eye
    else:
        B = A.size(0)
        return eye.unsqueeze(0).repeat(B, 1, 1).contiguous()


def _dft_matrix(N: int, norm: bool = True):
    k = torch.arange(N).reshape(-1, 1)
    n = torch.arange(N).reshape(1, -1)
    W = torch.exp(-2j * math.pi * k * n / N)
    if norm:
        W = W / math.sqrt(N)
    return W


_UPA_SHAPES = {
    16: (4, 4),
    32: (4, 8),
    64: (8, 8),
    128: (8, 16),
    256: (16, 16),
}


def _make_A(Nt: int, ula: bool):
    if ula:
        return _dft_matrix(Nt)

    if Nt in _UPA_SHAPES:
        Ny, Nx = _UPA_SHAPES[Nt]
    else:
        Ny = int(round(math.sqrt(Nt)))
        while Nt % Ny != 0 and Ny > 1:
            Ny -= 1
        Nx = Nt // Ny
        warnings.warn(f'Nt={Nt} is not in _UPA_SHAPES; '
                      f'using ({Ny}, {Nx}).', RuntimeWarning)

    Fy = _dft_matrix(Ny)
    Fx = _dft_matrix(Nx)
    A2d = torch.kron(Fx, Fy)

    return A2d


def _rand_frfs(k: int, Nt: int, B_bits: int, T: int, device, generator):
    Q = 2 ** B_bits
    phase_set = torch.arange(Q, device=device) * 2 * math.pi / Q
    idx = torch.randint(0, Q, (T, k, Nt), device=device,
                        generator=generator)
    phase = phase_set[idx]
    return torch.exp(1j * phase)


def simulate_ce_batch(
        H_spatial : Tensor,
        Nt_list   : Tensor,
        ula_flags : Tensor,
        CE_CFG,
        generator=None):
    """Simulate channel-estimation observations for a batch with a shared Nt."""
    B, _, Nt_max, _Kh = H_spatial.shape
    k    = CE_CFG['K_RF']
    T    = CE_CFG['T_repeat']
    Bbit = CE_CFG['B_bits']
    Pmax = CE_CFG['P_max']
    snr  = 10 ** (CE_CFG['SNR_dB'] / 10)

    device = H_spatial.device
    dtype  = H_spatial.dtype

    ce_in   = torch.zeros(B, 2, Nt_max, T, device=device, dtype=dtype)
    label   = torch.zeros(B, 2, Nt_max, 1, device=device, dtype=dtype)
    r_buf   = torch.empty(B, T * k,    device=device, dtype=torch.complex64)
    W_buf   = torch.empty(B, T * k, Nt_max, device=device, dtype=torch.complex64)
    n_buf   = torch.empty(B, T * k,       device=device, dtype=torch.cfloat)

    for b in range(B):
        Nt   = int(Nt_list[b])
        ula  = bool(ula_flags[b])

        A     = _make_A(Nt, ula).to(device)
        # Solve on CPU to avoid CUDA solver compatibility issues.
        invA = torch.linalg.solve(A.cpu(), eye_like(A.cpu())).to(A.device)

        Hr    = H_spatial[b, 0, :Nt, 0]
        Hi    = H_spatial[b, 1, :Nt, 0]
        H_cplx = torch.complex(Hr, Hi)
        label[b, 0, :Nt, 0] = Hr
        label[b, 1, :Nt, 0] = Hi
        H_ang = A @ H_cplx

        y_slots, n_slots, M_slots = [], [], []

        for t in range(T):
            F_RF = _rand_frfs(k, Nt, Bbit, 1, device, generator)[0]
            F_RF = F_RF * (math.sqrt(Pmax) /
                           torch.linalg.norm(F_RF))
            M     = F_RF @ invA
            pinvM = torch.linalg.pinv(M)

            sig_pow = torch.linalg.norm(F_RF @ H_cplx) ** 2
            rf_pow  = torch.linalg.norm(F_RF) ** 2
            sigma2  = sig_pow / (snr * rf_pow + 1e-12)

            n_ant = (torch.randn(Nt,  device=device, generator=generator) +
                     1j * torch.randn(Nt, device=device, generator=generator)
                    ) * math.sqrt(sigma2 / 2)
            n_tilde = F_RF @ n_ant

            y = M @ H_ang + n_tilde

            z = pinvM @ y

            ce_in[b, :, :Nt, t] = torch.view_as_real(z).T

            y_slots.append(y)
            n_slots.append(n_tilde)
            M_slots.append(M)

        Y_full = torch.cat(y_slots, dim=0)
        N_full = torch.cat(n_slots, dim=0)
        W_full = torch.cat(M_slots, dim=0)

        r_buf[b] = Y_full
        n_buf[b, :] = N_full

        W_buf[b, :, :Nt] = W_full

    return ce_in, label, r_buf, W_buf[:, :, :Nt], n_buf


class _CondAdapter(nn.Module):
    def __init__(self, M: int):
        super().__init__()
        self.M = M

        self.fc = nn.Sequential(
            nn.Linear(512, 2*M, bias=False),
            nn.SiLU(),
            nn.Linear(2*M, 2*M, bias=False)
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        B = z.size(0)
        out = self.fc(z)

        if y.dim() == 2:
            return out[:, : y.size(1)]

        if y.dim() == 3 and y.size(1) == 2:
            return out.view(B, 2, self.M)

        return out.view(B, 2, self.M).permute(0, 2, 1)


class _RefineNet(nn.Module):
    def __init__(self, in_ch: int = 4, feat: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, feat, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(feat, feat, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(feat, 2, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        return self.net(x)


class FCDAMPHeadSingle(nn.Module):
    """Refine CSI with one conditioning feature: image or LiDAR."""
    def __init__(self, Nt, Rx, M, T):
        super().__init__()
        self.core     = FCDAMPHead(Nt, Rx, M=M, T=T)
        self.adapter  = _CondAdapter(M)
        self.refine   = _RefineNet(in_ch=4)
        self.Nt, self.Kh = Nt, 1

    def _reshape_cond(self, cond, y):
        cond = self.adapter(cond, y)

        if cond.dim() == 2:
            cond = torch.stack((cond, torch.zeros_like(cond)), 1)
        elif cond.size(1) != 2:
            cond = cond.permute(0, 2, 1)

        # Align the measurement-domain condition with the antenna dimension.
        cond = F.interpolate(cond, size=self.Nt, mode='nearest')

        return cond.unsqueeze(-1)

    def forward(self, y, W, n_true=None, z_cond=None):
        h_est, n_hat, n_true = self.core(y=y, W=W, n_true=n_true)
        h_est_2d = h_est.view(y.size(0), 2, self.Nt, self.Kh)

        if z_cond is None:
            return h_est, n_hat, n_true

        cond_feat = self._reshape_cond(z_cond, y)
        fused = torch.cat([h_est_2d, cond_feat], dim=1)
        delta = self.refine(fused)
        h_out = h_est_2d + delta

        return h_out.flatten(2), n_hat, n_true


class FCDAMPHeadDual(nn.Module):
    """Refine CSI with separate image and LiDAR adapters."""
    def __init__(self, Nt, Rx, M, T):
        super().__init__()
        self.core      = FCDAMPHead(Nt, Rx, M=M, T=T)
        self.adapt_img = _CondAdapter(M)
        self.adapt_lid = _CondAdapter(M)
        self.refine    = _RefineNet(in_ch=6)
        self.Nt, self.Kh = Nt, 1

    def _adapt(self, z, adapter, y):
        if z is None:
            return torch.zeros(y.size(0), 2, self.Nt, self.Kh,
                                device=y.device, dtype=y.dtype)

        feat = adapter(z, y)
        if feat.dim() == 2:
            feat = torch.stack((feat, torch.zeros_like(feat)), 1)
        elif feat.size(1) != 2:
            feat = feat.permute(0, 2, 1)

        feat = F.interpolate(feat, size=self.Nt, mode='nearest')
        return feat.unsqueeze(-1)

    def forward(self, y, W, n_true=None,
                z_img=None, z_lid=None):
        h_est, n_hat, n_true = self.core(y=y, W=W, n_true=n_true)
        h_est_2d = h_est.view(y.size(0), 2, self.Nt, self.Kh)

        if z_img is None and z_lid is None:
            return h_est, n_hat, n_true

        img_feat = self._adapt(z_img, self.adapt_img, y)
        lid_feat = self._adapt(z_lid, self.adapt_lid, y)

        fused = torch.cat([h_est_2d, img_feat, lid_feat], dim=1)
        delta = self.refine(fused)
        h_out = h_est_2d + delta

        return h_out.flatten(2), n_hat, n_true


class WiT_Localization_Head(nn.Module):
    """Use cond_dim=0 for CSI only, 512 for one sensor, or 1024 for both."""
    def __init__(self, n_input: int, cond_dim: int = 0, d_model: int = 512, nhead: int = 4, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()

        self.embedding = nn.Linear(n_input * 2, d_model)

        pe = torch.zeros(5000, d_model)
        position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.mlp_head = nn.Sequential(
            nn.Linear(d_model + cond_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2)
        )

    def forward(self, x: torch.Tensor, z_cond: torch.Tensor | None = None):
        # Keep the localization baseline's fixed 25 dB input-noise model.
        snr_db = 25
        noise_power = torch.mean(x ** 2) * 10 ** (-snr_db / 10)
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        x = x + noise

        x = rearrange(x, 'b c n k -> b k (n c)')
        x = self.embedding(x)

        x = x + self.pe[:, :x.size(1), :]

        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)

        if z_cond is not None:
            x = torch.cat([x, z_cond], dim=-1)

        return self.mlp_head(x)


class BeamPredictionHead(nn.Module):
    def __init__(self, in_dim: int, max_Nt: int):
        super().__init__()

        hid = 128

        self.net = nn.Sequential(
            nn.Linear(in_dim, hid, bias=False),
            nn.LayerNorm(hid),
            nn.ReLU(inplace=True),

            nn.Linear(hid, max_Nt)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class CSIFeatureExtractor(nn.Module):
    def __init__(self,
                 Nt: int,
                 T: int,
                 csi_feat_dim: int = 512,
                 hidden_dim: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        in_dim = 2 * Nt * T
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, csi_feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, csi):
        return self.net(csi)


class _OutputMLP(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 Nt: int,
                 hidden=None,
                 dropout: float = 0.1):
        super().__init__()
        if hidden is None:
            hidden = [256, 256, 128]
        out_dim = 2 * Nt
        layers  = []
        last    = feat_dim
        for h in hidden:
            layers += [nn.Linear(last, h),
                       nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.Nt  = Nt

    def forward(self, fused_vec):
        y = self.mlp(fused_vec)
        B = fused_vec.size(0)
        return y.view(B, 2, self.Nt, 1)


class CEBaseHead(nn.Module):
    def __init__(self,
                 Nt, T,
                 cond_dim: int,
                 csi_feat_dim: int = 512,
                 hidden_out=None,
                 dropout: float = 0.3):
        super().__init__()
        self.csi_encoder = CSIFeatureExtractor(
            Nt, T,
            csi_feat_dim=csi_feat_dim,
            hidden_dim=max(1024, csi_feat_dim * 2),
            dropout=dropout)

        self.out_net = _OutputMLP(
            feat_dim=csi_feat_dim + cond_dim,
            Nt=Nt,
            hidden=hidden_out,
            dropout=dropout)


class CEFSingleMLPHead(CEBaseHead):
    """Estimate the channel from CSI and one sensor feature."""
    def __init__(self, Nt, T,
                 cond1_dim: int,
                 csi_feat_dim=512,
                 hidden_out=None, dropout=0.1):
        super().__init__(Nt, T,
                         cond_dim=cond1_dim,
                         csi_feat_dim=csi_feat_dim,
                         hidden_out=hidden_out,
                         dropout=dropout)
        self.c1 = cond1_dim

    def forward(self, ce_in, cond1):
        if cond1.size(1) != self.c1:
            raise ValueError('cond1 dim mismatch')
        csi_feat = self.csi_encoder(ce_in)
        fused    = torch.cat([csi_feat, cond1], dim=1)
        return self.out_net(fused)


class CEFDualMLPHead(CEBaseHead):
    """Estimate the channel from CSI, image, and LiDAR features."""
    def __init__(self, Nt, T,
                 cond1_dim: int, cond2_dim: int,
                 csi_feat_dim=512,
                 hidden_out=None, dropout=0.3):
        super().__init__(Nt, T,
                         cond_dim=cond1_dim + cond2_dim,
                         csi_feat_dim=csi_feat_dim,
                         hidden_out=hidden_out,
                         dropout=dropout)
        self.c1, self.c2 = cond1_dim, cond2_dim

    def forward(self, ce_in, cond1, cond2):
        if cond1.size(1) != self.c1 or cond2.size(1) != self.c2:
            raise ValueError('cond dim mismatch')
        csi_feat = self.csi_encoder(ce_in)
        fused    = torch.cat([csi_feat, cond1, cond2], dim=1)
        return self.out_net(fused)


class MultiModalPipeline(nn.Module):
    """task selects the downstream task; *_sm uses one sensor and *_mm uses both.
    Single-sensor CE and localization branches use z_img by default.
    To use LiDAR instead, change z_img to z_lid at the indicated call sites.
    """

    def __init__(self,
                 unique_nt_kh_pairs: list,
                 encoders_ckpt: dict,
                 Grid,
                 task: str,
                 device,
                 seed_ce: int,
                 d_model: int = 512,
                 CE_CFG=None,
                 freeze_encoders: bool = False):
        super().__init__()
        assert task in ('loc_pure', 'loc_sm', 'loc_mm', 'beam', 'beam_sm', 'CE_sm', 'CE_mm', 'CE_lamp', 'CE_lamp_sm', 'CE_lamp_mm')
        self.task = task
        self.ce_gen = torch.Generator(device).manual_seed(
            torch.randint(0, 2**31 - 1, (), dtype=torch.long).item()
            if seed_ce is None else seed_ce)
        self.img_enc = SeqImageEncoder(out_dim=512)
        print(encoders_ckpt['img'])
        raw_state = torch.load(encoders_ckpt['img'], map_location='cpu', weights_only=True)
        state = raw_state['img']
        # Remove DataParallel prefixes from checkpoint keys.
        stripped_state1 = {k.replace('module.', ''): v for k, v in state.items()}
        result1 = self.img_enc.load_state_dict(stripped_state1, strict=False)
        print('missing 1:', result1.missing_keys)
        print('unexpected 1:', result1.unexpected_keys)
        self.lidar_enc = SeqLiDAREncoder(C=512)
        print(encoders_ckpt['lidar'])
        raw_state = torch.load(encoders_ckpt['lidar'], map_location='cpu', weights_only=True)
        state = raw_state['lidar']
        stripped_state = {k.replace('module.', ''): v for k, v in state.items()}
        result = self.lidar_enc.load_state_dict(stripped_state, strict=False)
        print('missing :', result.missing_keys)
        print('unexpected:', result.unexpected_keys)
        if task == 'loc_pure' or task == 'loc_sm' or task == 'loc_mm':
            self.img_enc.eval()
            for p in self.img_enc.parameters():
                p.requires_grad = False
            self.lidar_enc.eval()
            for p in self.lidar_enc.parameters():
                p.requires_grad = False
        elif task == 'beam' or task == 'beam_sm' or task == 'CE_sm' or (task == 'CE_mm') or (task == 'CE_lamp') or (task == 'CE_lamp_sm') or (task == 'CE_lamp_mm'):
            self.img_enc.eval()
            for p in self.img_enc.parameters():
                p.requires_grad = False
            self.lidar_enc.eval()
            for p in self.lidar_enc.parameters():
                p.requires_grad = False
        self.heads = nn.ModuleDict()
        if 1:
            for nt, kh in unique_nt_kh_pairs:
                k_full = 2 * kh
                if task == 'beam':
                    self.heads[f'{nt}_{k_full}_0_beam'] = BeamPredictionHead(2 * d_model, nt)
                elif task == 'beam_sm':
                    self.heads[f'{nt}_{k_full}_0_beam_sm'] = BeamPredictionHead(d_model, nt)
                elif task == 'CE_sm':
                    for nt, kh in unique_nt_kh_pairs:
                        self.heads[f'{nt}_{kh}_ce_single'] = CEFSingleMLPHead(nt, CE_CFG['T_repeat'], cond1_dim=512)
                elif task == 'CE_mm':
                    for nt, kh in unique_nt_kh_pairs:
                        self.heads[f'{nt}_{kh}_ce_dual'] = CEFDualMLPHead(nt, CE_CFG['T_repeat'], cond1_dim=512, cond2_dim=512)
                elif task == 'CE_lamp':
                    M_y = CE_CFG['K_RF'] * CE_CFG['T_repeat']
                    for nt, kh in unique_nt_kh_pairs:
                        key = f'Nt{nt}_Kh{kh}_celamp'
                        self.heads[key] = FCDAMPHead(nt, 1, M=M_y, T=8)
                elif task == 'CE_lamp_sm':
                    M_y = CE_CFG['K_RF'] * CE_CFG['T_repeat']
                    for nt, kh in unique_nt_kh_pairs:
                        key = f'Nt{nt}_Kh{kh}_celamp_sm'
                        self.heads[key] = FCDAMPHeadSingle(nt, 1, M=M_y, T=8)
                elif task == 'CE_lamp_mm':
                    M_y = CE_CFG['K_RF'] * CE_CFG['T_repeat']
                    for nt, kh in unique_nt_kh_pairs:
                        key = f'Nt{nt}_Kh{kh}_celamp_mm'
                        self.heads[key] = FCDAMPHeadDual(nt, 1, M=M_y, T=8)
                if task == 'loc_pure':
                    for nt, kh in unique_nt_kh_pairs:
                        self.heads[f'{nt}_{2 * kh}_loc'] = WiT_Localization_Head(n_input=nt, cond_dim=0)
                elif task == 'loc_sm':
                    for nt, kh in unique_nt_kh_pairs:
                        self.heads[f'{nt}_{2 * kh}_loc_sm'] = WiT_Localization_Head(n_input=nt, cond_dim=512)
                elif task == 'loc_mm':
                    for nt, kh in unique_nt_kh_pairs:
                        self.heads[f'{nt}_{2 * kh}_loc_mm'] = WiT_Localization_Head(n_input=nt, cond_dim=1024)

    def forward(self,
                csi,
                ula_indicator=None,
                img_seq=None, img_len=None, img_off=None,
                lidar_seq=None, lid_len=None, lid_off=None,
                Nt_list=None, Kh_list=None, K_list=None, CE_CFG=None):
        if img_seq is not None:
            z_img_all = self.img_enc(img_seq, img_len)
            idx = torch.arange(img_seq.size(0), device=img_seq.device)
            z_img = z_img_all[idx, img_off]
        else:
            z_img = None
        if lidar_seq is not None:
            z_lid_all = self.lidar_enc(lidar_seq, lid_len)
            idx = torch.arange(lidar_seq.size(0), device=lidar_seq.device)
            z_lid = z_lid_all[idx, lid_off]
        else:
            z_lid = None
        # Preserve the original ablation setting, which zeros both sensor features.
        ablation_mode = 'zeros'
        if ablation_mode == 'zeros':
            z_img = torch.zeros_like(z_img)
            z_lid = torch.zeros_like(z_lid)
        elif ablation_mode == 'ones':
            z_img = torch.ones_like(z_img)
            z_lid = torch.ones_like(z_lid)
        elif ablation_mode == 'random':
            z_img = torch.randn_like(z_img)
            z_lid = torch.randn_like(z_lid)
        if self.task in ('beam', 'beam_sm'):
            if z_img != None and z_lid != None:
                z = torch.cat([z_img, z_lid], dim=1)
            elif z_img == None:
                z = z_lid
            elif z_lid == None:
                z = z_img
        nt = Nt_list[0].item()
        if self.task == 'beam':
            k = int(K_list[0])
            key = f'{nt}_{k}_0_beam'
            return self.heads[key](z)
        if self.task == 'beam_sm':
            k = int(K_list[0])
            key = f'{nt}_{k}_0_beam_sm'
            return self.heads[key](z)
        if self.task == 'CE_sm':
            ce_in, ce_tgt, _, _, _ = simulate_ce_batch(csi[:, :, 0], Nt_list, ula_indicator.to(csi.device), CE_CFG=CE_CFG, generator=self.ce_gen)
            nt = int(Nt_list[0])
            kh = int(Kh_list[0])
            key = f'{nt}_{kh}_ce_single'
            # Use z_lid instead of z_img for LiDAR conditioning.
            return (self.heads[key](ce_in, z_img), ce_tgt)
        if self.task == 'CE_mm':
            ce_in, ce_tgt, _, _, _ = simulate_ce_batch(csi[:, :, 0], Nt_list, ula_indicator.to(csi.device), CE_CFG=CE_CFG, generator=self.ce_gen)
            nt = int(Nt_list[0])
            kh = int(Kh_list[0])
            key = f'{nt}_{kh}_ce_dual'
            return (self.heads[key](ce_in, z_img, z_lid), ce_tgt)
        if self.task == 'CE_lamp':
            ce_in, h_gt, r, W, n_rx = simulate_ce_batch(csi[:, :, 0], Nt_list, ula_indicator.to(csi.device), CE_CFG=CE_CFG, generator=self.ce_gen)
            nt, kh = (int(Nt_list[0]), int(Kh_list[0]))
            key = f'Nt{nt}_Kh{kh}_celamp'
            h_est, n_hat, n_true = self.heads[key](y=r, W=W, n_true=n_rx)
            return (h_est, h_gt.squeeze(-1), n_hat, n_true.unsqueeze(1))
        if self.task == 'CE_lamp_sm':
            ce_in, h_gt, r, W, n_rx = simulate_ce_batch(csi[:, :, 0], Nt_list, ula_indicator.to(csi.device), CE_CFG=CE_CFG, generator=self.ce_gen)
            nt, kh = (int(Nt_list[0]), int(Kh_list[0]))
            key = f'Nt{nt}_Kh{kh}_celamp_sm'
            # Use z_cond=z_lid for LiDAR conditioning.
            h_est, n_hat, n_true = self.heads[key](y=r, W=W, n_true=n_rx, z_cond=z_img)
            return (h_est, h_gt.squeeze(-1), n_hat, n_true.unsqueeze(1))
        if self.task == 'CE_lamp_mm':
            ce_in, h_gt, r, W, n_rx = simulate_ce_batch(csi[:, :, 0], Nt_list, ula_indicator.to(csi.device), CE_CFG=CE_CFG, generator=self.ce_gen)
            nt, kh = (int(Nt_list[0]), int(Kh_list[0]))
            key = f'Nt{nt}_Kh{kh}_celamp_mm'
            h_est, n_hat, n_true = self.heads[key](y=r, W=W, n_true=n_rx, z_img=z_img, z_lid=z_lid)
            return (h_est, h_gt.squeeze(-1), n_hat, n_true.unsqueeze(1))
        nt = int(Nt_list[0])
        k = int(K_list[0])
        if self.task.startswith('loc_'):
            clean_csi = csi
            if self.task == 'loc_pure':
                key = f'{nt}_{k}_loc'
                return self.heads[key](csi, z_cond=None)
            elif self.task == 'loc_sm':
                key = f'{nt}_{k}_loc_sm'
                # Use z_lid instead of z_img for LiDAR conditioning.
                z_cond = z_img
                return self.heads[key](csi, z_cond=z_cond)
            elif self.task == 'loc_mm':
                key = f'{nt}_{k}_loc_mm'
                zi = torch.zeros(clean_csi.size(0), 512, device=csi.device) if z_img is None else z_img
                zl = torch.zeros(clean_csi.size(0), 512, device=csi.device) if z_lid is None else z_lid
                z_cond = torch.cat([zi, zl], dim=-1)
                return self.heads[key](clean_csi, z_cond=z_cond)
