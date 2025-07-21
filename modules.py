import numpy as np
import torch
import time
import sys
import os
import gc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import torchvision

def estimate_dict_size(dic):
    total_size = 0
    for k, v in dic.items():
        size = sys.getsizeof(k)
        if isinstance(v, np.ndarray):
            size += v.nbytes
        elif hasattr(v, '__sizeof__'):
            size += v.__sizeof__()
        total_size += size
    return total_size

# ----- Pruner Module -----

class Pruner:
    def __init__(self):
        pass

    def prune_points_one_shot(self, prune_mask, sh_mask, gaussians):
        valid_points_mask = ~prune_mask

        gaussians._features_rest = gaussians._features_rest.detach()
        gaussians._features_rest[sh_mask] = 0.0


        gaussians._xyz = gaussians._xyz[valid_points_mask]
        gaussians._features_dc = gaussians._features_dc[valid_points_mask]
        gaussians._features_rest = gaussians._features_rest[valid_points_mask]
        gaussians._opacity = gaussians._opacity[valid_points_mask]
        gaussians._scaling = gaussians._scaling[valid_points_mask]
        gaussians._rotation = gaussians._rotation[valid_points_mask]

    def run(self, prune_rate, sh_rate, import_score, gaussians):
        # Sort importance scores in ascending order
        sorted_scores, sorted_indices = torch.sort(import_score, dim=0)
        num_elements = import_score.size(0)

        # Determine prune mask for the least significant gaussians
        num_prune = int(num_elements * prune_rate)
        prune_indices = sorted_indices[:num_prune]
        prune_mask = torch.zeros(num_elements, dtype=torch.bool)
        prune_mask[prune_indices] = True

        # Determine SH mask for the least significant + SH gaussians
        num_sh = int(num_elements * (prune_rate + sh_rate))
        sh_indices = sorted_indices[:num_sh]
        sh_mask = torch.zeros(num_elements, dtype=torch.bool)
        sh_mask[sh_indices] = True

        # Final SH mask is the set difference between SH and prune masks
        sh_mask = torch.logical_xor(prune_mask, sh_mask)

        # Apply pruning and SH adjustment
        self.prune_points_one_shot(prune_mask, sh_mask, gaussians)


# ----- Quantizer Module -----

class Quantizer:
    def __init__(self, segments: int = 1000, bits_config: dict = None):
        self.segments = segments
        self.bits = bits_config or {
            "bits_f_dc": 8,
            "bits_f_rest": 4,
            "bits_opacity": 4,
            "bits_scaling": 8,
            "bits_rotation": 8
        }

    # -------------------- Bit-Packing Utilities -------------------- #

    def pack_4bit_to_8bit(self, a, b):
        return (a.astype(np.uint8) << 4) | b.astype(np.uint8)

    def unpack_8bit_to_4bit(self, packed_value):
        return (packed_value >> 4) & 0x0F, packed_value & 0x0F

    def pack_2bit_to_8bit(self, a, b, c, d):
        return (
            (a.astype(np.uint8) << 6) |
            (b.astype(np.uint8) << 4) |
            (c.astype(np.uint8) << 2) |
            d.astype(np.uint8)
        )

    def unpack_8bit_to_2bit(self, packed_value):
        return (
            (packed_value >> 6) & 0x03,
            (packed_value >> 4) & 0x03,
            (packed_value >> 2) & 0x03,
            packed_value & 0x03
        )

    # -------------------- Int8 Quantization -------------------- #

    def asymmetric_quantize_int8_per_segment_parallel(self, r, num_segments):
        N, K = r.shape
        num_segments = min(max(num_segments, 1), N)
        indices = np.linspace(0, N, num_segments + 1, dtype=int)

        segment_bounds = [
            (indices[i], indices[i + 1])
            for i in range(num_segments)
            if indices[i] < indices[i + 1]
        ]

        scales = np.zeros((len(segment_bounds), K))
        zero_points = np.zeros((len(segment_bounds), K))
        quantized_data = np.zeros_like(r, dtype=np.int8)

        def process_segment(i):
            start, end = segment_bounds[i]
            seg = r[start:end, :]

            alpha = seg.min(axis=0)
            beta = seg.max(axis=0)
            scale = (beta - alpha) / 255.0
            scale[scale == 0] = 1.0
            zero_point = -(alpha / scale + 128)

            q = np.round(seg / scale + zero_point).astype(np.int8)
            q = np.clip(q, -128, 127)
            q[:, scale == 0] = -128

            return i, q, scale, zero_point

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_segment, i) for i in range(len(segment_bounds))]
            for future in futures:
                i, q_seg, scale, zp = future.result()
                start, end = segment_bounds[i]
                quantized_data[start:end, :] = q_seg
                scales[i, :] = scale
                zero_points[i, :] = zp

        return quantized_data, scales, zero_points

    def asymmetric_dequantize_int8_per_segment_parallel(self, quantized_data, scales, zero_points, num_segments, shape):
        N, K = shape
        num_segments = min(num_segments, N)
        indices = np.linspace(0, N, num_segments + 1, dtype=int)

        segment_bounds = [
            (indices[i], indices[i + 1])
            for i in range(num_segments)
            if indices[i] < indices[i + 1]
        ]

        reconstructed = np.zeros(shape, dtype=np.float32)

        def process_segment(i):
            start, end = segment_bounds[i]
            q = quantized_data[start:end, :]
            return i, (q.astype(np.float32) - zero_points[i][np.newaxis, :]) * scales[i][np.newaxis, :]

        with ThreadPoolExecutor() as executor:
            for future in executor.map(process_segment, range(len(segment_bounds))):
                i, seg = future
                start, end = segment_bounds[i]
                reconstructed[start:end, :] = seg

        return reconstructed

    # -------------------- Int4 Quantization -------------------- #

    def quantize_segment(self, segment_r, alpha_q=0, beta_q=15):
        alpha = segment_r.min(axis=0)
        beta = segment_r.max(axis=0)
        scale = (beta - alpha) / (beta_q - alpha_q)
        scale[scale == 0] = 1.0
        zero_point = alpha_q - alpha / scale

        quantized = np.round(segment_r / scale + zero_point).astype(np.uint8)
        quantized = np.clip(quantized, alpha_q, beta_q)

        quantized = quantized.reshape(segment_r.shape[0] // 2, 2, segment_r.shape[1])
        packed = self.pack_4bit_to_8bit(quantized[:, 0, :], quantized[:, 1, :])
        return packed, scale, zero_point

    def asymmetric_quantize_int4_segment_parallel(self, r, num_segments):
        N, K = r.shape
        num_segments = min(num_segments, N)
        segment_size = int(np.ceil(N / num_segments))
        segment_size += segment_size % 2

        total_padded = num_segments * segment_size
        padding = total_padded - N
        r_padded = np.vstack([r, np.zeros((padding, K), dtype=r.dtype)]) if padding > 0 else r

        r_segments = r_padded.reshape(num_segments, segment_size, K)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.quantize_segment, r_segments))

        packed_segments, scales, zero_points = zip(*results)
        return (
            np.vstack(packed_segments),
            np.vstack(scales),
            np.vstack(zero_points),
            segment_size,
            N
        )

    def dequantize_segment(self, packed_segment, scale, zero_point, segment_size, K):
        a, b = self.unpack_8bit_to_4bit(packed_segment)
        quantized = np.empty((segment_size, K), dtype=np.float32)
        quantized[::2], quantized[1::2] = a, b

        scale = scale[np.newaxis, :]
        zero_point = zero_point[np.newaxis, :]
        return (quantized - zero_point) * scale

    def asymmetric_dequantize_int4_segment_parallel(self, packed_matrix, scales, zero_points, segment_size, N_original):
        K = packed_matrix.shape[1]
        segment_size += segment_size % 2
        num_segments = len(scales)
        per_segment = segment_size // 2

        with ThreadPoolExecutor() as executor:
            results = [
                executor.submit(
                    self.dequantize_segment,
                    packed_matrix[i * per_segment: (i + 1) * per_segment, :],
                    scales[i],
                    zero_points[i],
                    segment_size,
                    K
                )
                for i in range(num_segments)
            ]
            dequantized = [f.result() for f in results]

        return np.vstack(dequantized)[:N_original, :]

    # -------------------- Int2 Quantization -------------------- #

    def quantize_segment_int2(self, segment_r, alpha_q=0, beta_q=3):
        alpha = segment_r.min(axis=0)
        beta = segment_r.max(axis=0)
        scale = (beta - alpha) / (beta_q - alpha_q)
        scale[scale == 0] = 1.0
        zero_point = alpha_q - alpha / scale

        quantized = np.round(segment_r / scale + zero_point).astype(np.uint8)
        quantized = np.clip(quantized, alpha_q, beta_q)

        quantized = quantized.reshape(segment_r.shape[0] // 4, 4, segment_r.shape[1])
        packed = self.pack_2bit_to_8bit(quantized[:, 0, :], quantized[:, 1, :], quantized[:, 2, :], quantized[:, 3, :])
        return packed, scale, zero_point

    def asymmetric_quantize_int2_segment_parallel(self, r, num_segments):
        N, K = r.shape
        num_segments = min(num_segments, N)
        segment_size = int(np.ceil(N / num_segments))
        if segment_size % 4 != 0:
            segment_size += 4 - (segment_size % 4)

        total_padded = num_segments * segment_size
        padding = total_padded - N
        r_padded = np.vstack([r, np.zeros((padding, K), dtype=r.dtype)]) if padding > 0 else r

        r_segments = r_padded.reshape(num_segments, segment_size, K)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.quantize_segment_int2, r_segments))

        packed_segments, scales, zero_points = zip(*results)
        return (
            np.vstack(packed_segments),
            np.vstack(scales),
            np.vstack(zero_points),
            segment_size,
            N
        )

    def dequantize_segment_int2(self, packed_segment, scale, zero_point, segment_size, K):
        a, b, c, d = self.unpack_8bit_to_2bit(packed_segment)
        quantized = np.empty((segment_size, K), dtype=np.float32)
        quantized[::4], quantized[1::4], quantized[2::4], quantized[3::4] = a, b, c, d

        scale = scale[np.newaxis, :]
        zero_point = zero_point[np.newaxis, :]
        return (quantized - zero_point) * scale

    def asymmetric_dequantize_int2_segment_parallel(self, packed_matrix, scales, zero_points, segment_size, N_original):
        K = packed_matrix.shape[1]
        if segment_size % 4 != 0:
            segment_size += 4 - (segment_size % 4)

        per_segment = segment_size // 4
        num_segments = len(scales)

        with ThreadPoolExecutor() as executor:
            results = [
                executor.submit(
                    self.dequantize_segment_int2,
                    packed_matrix[i * per_segment: (i + 1) * per_segment, :],
                    scales[i],
                    zero_points[i],
                    segment_size,
                    K
                )
                for i in range(num_segments)
            ]
            dequantized = [f.result() for f in results]

        return np.vstack(dequantized)[:N_original, :]

    def quantize(self, gaussians, path, num_segments=1000,
                bits_f_dc=8, bits_f_rest=4,
                bits_opacity=4, bits_scaling=8, bits_rotation=8):
        # Extract features and move to CPU
        xyz = gaussians._xyz.detach().cpu().numpy()
        f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = gaussians._opacity.detach().cpu().numpy()
        scale = gaussians._scaling.detach().cpu().numpy()
        rotation = gaussians.rotation_activation(gaussians._rotation).detach().cpu().numpy()
        N = xyz.shape[0]

        zero_mask = np.all(f_rest == 0, axis=1)
        nonzero_idx = np.where(~zero_mask)[0]
        zero_idx = np.where(zero_mask)[0]
        sorted_idx = np.concatenate([nonzero_idx, zero_idx])
        boundary_index = len(nonzero_idx)

        xyz = xyz[sorted_idx]
        f_dc = f_dc[sorted_idx]
        f_rest = f_rest[sorted_idx][:boundary_index]
        opacities = opacities[sorted_idx]
        scale = scale[sorted_idx]
        rotation = rotation[sorted_idx]

        npz_data = {
            'bits_f_rest': bits_f_rest,
            'sh_all_zero': f_rest.shape[0] == 0,
            'boundary_index': boundary_index
        }

        if not npz_data['sh_all_zero']:
            if bits_f_rest == 2:
                q, S, Z, seg_size, Nq = self.asymmetric_quantize_int2_segment_parallel(f_rest, num_segments)
                npz_data.update({'M_rest': q.astype(np.int8), 'S_rest': S, 'Z_rest': Z, 'segment_size_f_rest': seg_size, 'N_rest': Nq})
            elif bits_f_rest == 4:
                q, S, Z, seg_size, Nq = self.asymmetric_quantize_int4_segment_parallel(f_rest, num_segments)
                npz_data.update({'M_rest': q.astype(np.int8), 'S_rest': S, 'Z_rest': Z, 'segment_size_f_rest': seg_size, 'N_rest': Nq})
            elif bits_f_rest == 8:
                q, S, Z = self.asymmetric_quantize_int8_per_segment_parallel(f_rest, num_segments)
                npz_data.update({'M_rest': q.astype(np.int8), 'S_rest': S, 'Z_rest': Z})
            else:
                npz_data['f_rest'] = f_rest.astype(np.float16)

        bits_map = {'f_dc': bits_f_dc, 'opacities': bits_opacity, 'scale': bits_scaling, 'rotation': bits_rotation}
        var_map = {'f_dc': f_dc, 'opacities': opacities, 'scale': scale, 'rotation': rotation}
        bit_groups = {2: [], 4: [], 8: [], 16: []}

        for k, b in bits_map.items():
            if b in bit_groups:
                bit_groups[b].append(k)

        if bit_groups[2]:
            int2_attributes = np.concatenate([var_map[k] for k in bit_groups[2]], axis=1)
            q, S, Z, seg_size, Nq = self.asymmetric_quantize_int2_segment_parallel(int2_attributes, num_segments)
            npz_data.update({'M_int2': q.astype(np.int8), 'S_int2': S, 'Z_int2': Z, 'segment_size_int2': seg_size, 'N_int2': Nq})
        if bit_groups[4]:
            int4_attributes = np.concatenate([var_map[k] for k in bit_groups[4]], axis=1)
            q, S, Z, seg_size, Nq = self.asymmetric_quantize_int4_segment_parallel(int4_attributes, num_segments)
            npz_data.update({'M_int4': q.astype(np.int8), 'S_int4': S, 'Z_int4': Z, 'segment_size_int4': seg_size, 'N_int4': Nq})
        if bit_groups[8]:
            int8_attributes = np.concatenate([var_map[k] for k in bit_groups[8]], axis=1)
            q, S, Z = self.asymmetric_quantize_int8_per_segment_parallel(int8_attributes, num_segments)
            npz_data.update({'M_int8': q.astype(np.int8), 'S_int8': S, 'Z_int8': Z})
        if bit_groups[16]:
            fp16_attributes = np.concatenate([var_map[k] for k in bit_groups[16]], axis=1)
            npz_data['M_fp16'] = fp16_attributes.astype(np.float16)

        npz_data.update({
            'N': N,
            'num_segments': num_segments,
            'xyz': xyz.astype(np.float16),
            'int2_names': bit_groups[2],
            'int4_names': bit_groups[4],
            'int8_names': bit_groups[8],
            'fp16_names': bit_groups[16],
            'npz_path': os.path.join(path, 'quantization_parameters_pipeline.npz')
        })
        return npz_data

    def dequantize(self, gaussians, npz_data):
        N = npz_data['N']
        num_segments = npz_data['num_segments']
        features_extra = np.zeros((N, 45), dtype=np.float32)

        if not npz_data['sh_all_zero']:
            if npz_data['bits_f_rest'] == 2:
                deq = self.asymmetric_dequantize_int2_segment_parallel(
                    npz_data['M_rest'], npz_data['S_rest'], npz_data['Z_rest'],
                    npz_data['segment_size_f_rest'], npz_data['N_rest']
                )
            elif npz_data['bits_f_rest'] == 4:
                deq = self.asymmetric_dequantize_int4_segment_parallel(
                    npz_data['M_rest'], npz_data['S_rest'], npz_data['Z_rest'],
                    npz_data['segment_size_f_rest'], npz_data['N_rest']
                )
            elif npz_data['bits_f_rest'] == 8:
                deq = self.asymmetric_dequantize_int8_per_segment_parallel(
                    npz_data['M_rest'], npz_data['S_rest'], npz_data['Z_rest'],
                    num_segments, npz_data['M_rest'].shape
                )
            else:
                deq = npz_data['f_rest'].astype(np.float32)

            features_extra[:npz_data['boundary_index'], :] = deq

        features_extra = features_extra.reshape((N, 3, (gaussians.max_sh_degree + 1) ** 2 - 1))
        length_map = {'f_dc': 3, 'opacities': 1, 'scale': 3, 'rotation': 4}
        new_var = {}

        for bits, key in [(2, 'int2'), (4, 'int4'), (8, 'int8'), (16, 'fp16')]:
            names = npz_data.get(f'{key}_names', [])
            if not names:
                continue

            if bits in [2, 4, 8]:
                if bits == 2:
                    deq = self.asymmetric_dequantize_int2_segment_parallel(
                        npz_data['M_int2'], npz_data['S_int2'], npz_data['Z_int2'],
                        npz_data['segment_size_int2'], npz_data['N_int2']
                    )
                elif bits == 4:
                    deq = self.asymmetric_dequantize_int4_segment_parallel(
                        npz_data['M_int4'], npz_data['S_int4'], npz_data['Z_int4'],
                        npz_data['segment_size_int4'], npz_data['N_int4']
                    )
                else:
                    deq = self.asymmetric_dequantize_int8_per_segment_parallel(
                        npz_data['M_int8'], npz_data['S_int8'], npz_data['Z_int8'],
                        num_segments, npz_data['M_int8'].shape
                    )

                start = 0
                for name in names:
                    width = length_map[name]
                    new_var[name] = deq[:, start:start + width]
                    start += width
            elif bits == 16:
                start = 0
                fp16 = npz_data['M_fp16'].astype(np.float32)
                for name in names:
                    width = length_map[name]
                    new_var[name] = fp16[:, start:start + width]
                    start += width

        features_dc = np.zeros((N, 3, 1), dtype=np.float32)
        features_dc[:, :, 0] = new_var['f_dc']

        gaussians._xyz = torch.nn.Parameter(torch.tensor(npz_data['xyz'].astype(np.float32), device="cuda").requires_grad_(True))
        gaussians._features_dc = torch.nn.Parameter(torch.tensor(features_dc, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        gaussians._features_rest = torch.nn.Parameter(torch.tensor(features_extra, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        gaussians._opacity = torch.nn.Parameter(torch.tensor(new_var['opacities'], device="cuda").requires_grad_(True))
        gaussians._scaling = torch.nn.Parameter(torch.tensor(new_var['scale'], device="cuda").requires_grad_(True))
        gaussians._rotation = torch.nn.Parameter(torch.tensor(new_var['rotation'], device="cuda").requires_grad_(True))
        gaussians.active_sh_degree = gaussians.max_sh_degree


from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
import shutil
import copy

@dataclass(frozen=True)
class SearchConfig:
    pruning_rate: float
    sh_rate: float

class Searcher:
    def __init__(
        self,
        args,
        dataset,
        gaussians,
        pipe,
        scene,
        imp_score,
        search_space=None,
        target_psnr_drop=1.0,
        save_render=False
    ):
        self.args = args
        self.dataset = dataset
        self.gaussians = gaussians
        self.pipe = pipe
        self.scene = scene
        self.imp_score = imp_score
        self.target_psnr_drop = target_psnr_drop
        self.save_render = save_render

        self.Pruner = Pruner()
        self.Quantizer = Quantizer()

        self.search_space = self._init_search_space(search_space)

    def _init_search_space(self, search_space):
        if search_space == 'default':
            return [
                SearchConfig(p, s) for p, s in [
                    (0.2, 0.3), (0.1, 0.5), (0.2, 0.4), (0.3, 0.3),
                    (0.1, 0.6), (0.2, 0.5), (0.3, 0.4), (0.4, 0.3),
                    (0.2, 0.6), (0.3, 0.5), (0.4, 0.4), (0.3, 0.6),
                    (0.4, 0.5), (0.5, 0.4), (0.6, 0.3), (0.5, 0.5),
                    (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)
                ]
            ]
        return [SearchConfig(*cfg) if not isinstance(cfg, SearchConfig) else cfg for cfg in search_space]

    def render_set_pipeline(self, views, gaussians, pipeline, background):
        psnrs, ssims, lpipss = [], [], []

        if self.save_render:
            render_path = os.path.join(self.tgt_path, "test/renders")
            gt_path = os.path.join(self.tgt_path, "test/gt")
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(gt_path, exist_ok=True)

        for idx, view in enumerate(views):
            rendering = render(view, gaussians, pipeline, background)["render"]
            gt = view.original_image[0:3, :, :]
            psnrs.append(psnr(rendering, gt).mean().item())
            # ssims.append(ssim(rendering, gt).mean().item())
            # lpipss.append(lpips(rendering, gt, net_type="vgg").mean().item())

            if self.save_render:
                torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
                torchvision.utils.save_image(gt, os.path.join(gt_path, f"{idx:05d}.png"))

        # print ("psnr: ", np.mean(np.array(psnrs)))
        # print ("ssim: ", np.mean(np.array(ssims)))
        # print ("lpips: ", np.mean(np.array(lpipss)))

        return psnrs

    def render_sets_pipeline(self, gaussians : GaussianModel, skip_train : bool, skip_test : bool, load_iteration=None):
        with torch.no_grad():
            bg_color = torch.tensor([1, 1, 1] if self.dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
            final_scores = []
            if not skip_train:
                final_scores += self.render_set_pipeline(self.scene.getTrainCameras(), gaussians, self.pipe, bg_color)
            if not skip_test:
                final_scores += self.render_set_pipeline(self.scene.getTestCameras(), gaussians, self.pipe, bg_color)
            return np.mean(final_scores)
    
    def spin_once(self, pruning_rate, sh_rate, save_render=False):
        self.save_render = save_render
        configure = f"prune_{pruning_rate:.1f}_sh_{sh_rate:.1f}"
        out_path = os.path.join(self.tgt_path, "point_cloud", configure)
        # os.makedirs(out_path, exist_ok=True)

        # Deep copy original gaussians
        start_time = time.time()
        gaussians = copy.deepcopy(self.gaussians)
        print(f"  Deep copy time:         {time.time() - start_time:.4f} s")

        start_time = time.time()
        self.Pruner.run(pruning_rate, sh_rate, torch.from_numpy(self.imp_score), gaussians)
        print(f"  Prune time:             {time.time() - start_time:.2f} s")

        # Render after pruning
        # start_time = time.time()
        # current_psnr = self.render_sets_pipeline(
        #     gaussians, skip_train=True, skip_test=False
        # )
        # print(f"  Render time: {time.time() - start_time:.2f}s")
        # print(f"  PSNR after pruning: {current_psnr:.4f}")

        # Quantize
        start_time = time.time()
        dic = self.Quantizer.quantize(
            gaussians, out_path, self.args.segments,
            bits_f_dc=8, bits_f_rest=4,
            bits_opacity=4, bits_scaling=8, bits_rotation=8
        )
        print(f"  Quantization time:      {time.time() - start_time:.2f} s")

        # Dequantize
        start_time = time.time()
        self.Quantizer.dequantize(gaussians, dic)
        print(f"  Dequantization time:    {time.time() - start_time:.2f} s")

        # Estimate size
        filesize_est = estimate_dict_size(dic)
        print(f"  Estimated size:         {filesize_est / (1024 * 1024):.2f} MB")

        # Evaluate quality
        start_time = time.time()
        current_psnr = self.render_sets_pipeline(
            gaussians, skip_train=True, skip_test=False
        )
        print(f"  Final render time:      {time.time() - start_time:.2f} s")
        print(f"  Final PSNR:             {current_psnr:.4f} dB")
        print(f"  ΔPSNR from input:       {self.baseline_psnr - current_psnr:.4f} dB")
        return current_psnr, filesize_est, dic

    def run_search(self):
        src_path = self.args.model_path
        scene_name = os.path.basename(os.path.normpath(src_path))
        self.tgt_path = os.path.join(self.args.output_path, scene_name)
        os.makedirs(self.tgt_path, exist_ok=True)

        # np.savez(os.path.join(self.tgt_path, "imp_score"), self.imp_score)
        # Save baseline point cloud
        # os.makedirs(os.path.join(self.tgt_path, 'point_cloud/baseline'), exist_ok=True)
        # shutil.copy(
        #     os.path.join(src_path, 'point_cloud/iteration_30000/point_cloud.ply'),
        #     os.path.join(self.tgt_path, 'point_cloud/baseline/point_cloud.ply')
        # )

        # Compute baseline PSNR
        print(f"\n[Baseline] ...")
        start_time = time.time()
        self.baseline_psnr = self.render_sets_pipeline(
            self.gaussians, skip_train=True, skip_test=False, load_iteration=30000
        )
        print(f"  Baseline render time:   {time.time() - start_time:.2f} s")
        print(f"  Baseline PSNR:          {self.baseline_psnr:.4f} dB")

        start_index = len(self.search_space) // 2  # Middle index
        direction = 1
        index = start_index
        # found_target = False
        closest_dic = None

        print(f"\nStarting search at middle index {start_index} with "
            f"(pruning_rate, sh_rate) = "
            f"({self.search_space[start_index].pruning_rate:.1f}, {self.search_space[start_index].sh_rate:.1f})")

        while 0 <= index < len(self.search_space):
            cfg = self.search_space[index]
            print(f"\n[Search] pruning_rate: {cfg.pruning_rate:.1f}, sh_rate: {cfg.sh_rate:.1f}")

            current_psnr, filesize_est, dic = self.spin_once(cfg.pruning_rate, cfg.sh_rate)

            if index == start_index:
                direction = 1 if current_psnr >= (self.baseline_psnr - self.target_psnr_drop) else -1

            if current_psnr >= self.baseline_psnr - self.target_psnr_drop:
                closest_dic = dic
                closest_dic.update({
                    "pruning_rate": cfg.pruning_rate,
                    "sh_rate": cfg.sh_rate,
                    "psnr": current_psnr,
                    "filesize": filesize_est
                })
                if direction == -1:
                    break
            else:
                if direction == 1:
                    break

            index += direction

        print("\nSearch complete.")
        return closest_dic if closest_dic else None