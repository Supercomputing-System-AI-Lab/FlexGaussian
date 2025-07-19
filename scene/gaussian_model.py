#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from icecream import ic
# from vectree.utils import load_vqgaussian, write_ply_data

from utils.quantization_utils import asymmetric_quantize_int4_segment_parallel, asymmetric_quantize_int8_per_segment_parallel, asymmetric_dequantize_int8_per_segment_parallel, asymmetric_dequantize_int4_segment_parallel, asymmetric_dequantize_int2_segment_parallel, asymmetric_quantize_int2_segment_parallel 
import gc
import time


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)  # empty or frezze
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)
        
    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    # def prune_points_one_shot(self, prune_mask, sh_mask):
    #     valid_points_mask = ~prune_mask

    #     self._features_rest = self._features_rest.detach()
    #     self._features_rest[sh_mask] = 0.0


    #     self._xyz = self._xyz[valid_points_mask]
    #     self._features_dc = self._features_dc[valid_points_mask]
    #     self._features_rest = self._features_rest[valid_points_mask]
    #     self._opacity = self._opacity[valid_points_mask]
    #     self._scaling = self._scaling[valid_points_mask]
    #     self._rotation = self._rotation[valid_points_mask]

    def prune_opacity(self, percent):
        sorted_tensor, _ = torch.sort(self.get_opacity, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (self.get_opacity <= value_nth_percentile).squeeze()

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_gaussians(self, percent, import_score: list):
        ic(import_score.shape)
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)

    # def prune_gaussians_one_shot(self, prune_rate, sh_rate, import_score: list):
    #     # Sort importance scores in ascending order
    #     sorted_scores, sorted_indices = torch.sort(import_score, dim=0)
    #     num_elements = import_score.size(0)

    #     # Determine prune mask for the least significant gaussians
    #     num_prune = int(num_elements * prune_rate)
    #     prune_indices = sorted_indices[:num_prune]
    #     prune_mask = torch.zeros(num_elements, dtype=torch.bool)
    #     prune_mask[prune_indices] = True

    #     # Determine SH mask for the least significant + SH gaussians
    #     num_sh = int(num_elements * (prune_rate + sh_rate))
    #     sh_indices = sorted_indices[:num_sh]
    #     sh_mask = torch.zeros(num_elements, dtype=torch.bool)
    #     sh_mask[sh_indices] = True

    #     # Final SH mask is the set difference between SH and prune masks
    #     sh_mask = torch.logical_xor(prune_mask, sh_mask)

    #     # Apply pruning and SH adjustment
    #     self.prune_points_one_shot(prune_mask, sh_mask)

    def quantization(self, path, num_segments=1000,
                    bits_f_dc=8, bits_f_rest=4,
                    bits_opacity=4, bits_scaling=8, bits_rotation=8):
        start_time = time.time()

        # === Extract features and move to CPU ===
        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self.rotation_activation(self._rotation).detach().cpu().numpy()
        N = xyz.shape[0]

        # === Reorder based on SH activity ===
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

        # === Quantize f_rest ===
        if not npz_data['sh_all_zero']:
            if bits_f_rest == 2:
                q, S, Z, seg_size, Nq = asymmetric_quantize_int2_segment_parallel(f_rest, num_segments)
                npz_data.update({
                    'M_rest': q.astype(np.int8), 'S_rest': S, 'Z_rest': Z,
                    'segment_size_f_rest': seg_size, 'N_rest': Nq
                })
            elif bits_f_rest == 4:
                q, S, Z, seg_size, Nq = asymmetric_quantize_int4_segment_parallel(f_rest, num_segments)
                npz_data.update({
                    'M_rest': q.astype(np.int8), 'S_rest': S, 'Z_rest': Z,
                    'segment_size_f_rest': seg_size, 'N_rest': Nq
                })
            elif bits_f_rest == 8:
                q, S, Z = asymmetric_quantize_int8_per_segment_parallel(f_rest, num_segments)
                npz_data.update({
                    'M_rest': q.astype(np.int8), 'S_rest': S, 'Z_rest': Z
                })
            else:
                npz_data['f_rest'] = f_rest.astype(np.float16)

        del f_rest

        # === Organize feature types by bit depth ===
        bits_map = {
            'f_dc': bits_f_dc,
            'opacities': bits_opacity,
            'scale': bits_scaling,
            'rotation': bits_rotation
        }

        var_map = {
            'f_dc': f_dc,
            'opacities': opacities,
            'scale': scale,
            'rotation': rotation
        }

        bit_groups = {2: [], 4: [], 8: [], 16: []}
        for k, b in bits_map.items():
            if b in bit_groups:
                bit_groups[b].append(k)
            else:
                print(f"[Warning] Unsupported bit depth: {b} for {k}")

        int2_attributes = int4_attributes = int8_attributes = fp16_attributes = None
        if bit_groups[2]:
            int2_attributes = np.concatenate([var_map[k] for k in bit_groups[2]], axis=1)
        if bit_groups[4]:
            int4_attributes = np.concatenate([var_map[k] for k in bit_groups[4]], axis=1)
        if bit_groups[8]:
            int8_attributes = np.concatenate([var_map[k] for k in bit_groups[8]], axis=1)
        if bit_groups[16]:
            fp16_attributes = np.concatenate([var_map[k] for k in bit_groups[16]], axis=1)

        if int2_attributes is not None:
            q, S, Z, seg_size, Nq = asymmetric_quantize_int2_segment_parallel(int2_attributes, num_segments)
            npz_data.update({
                'M_int2': q.astype(np.int8), 'S_int2': S, 'Z_int2': Z,
                'segment_size_int2': seg_size, 'N_int2': Nq
            })

        if int4_attributes is not None:
            q, S, Z, seg_size, Nq = asymmetric_quantize_int4_segment_parallel(int4_attributes, num_segments)
            npz_data.update({
                'M_int4': q.astype(np.int8), 'S_int4': S, 'Z_int4': Z,
                'segment_size_int4': seg_size, 'N_int4': Nq
            })

        if int8_attributes is not None:
            q, S, Z = asymmetric_quantize_int8_per_segment_parallel(int8_attributes, num_segments)
            npz_data.update({
                'M_int8': q.astype(np.int8), 'S_int8': S, 'Z_int8': Z
            })

        if fp16_attributes is not None:
            npz_data['M_fp16'] = fp16_attributes.astype(np.float16)

        # === Metadata and cleanup ===
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

        del f_dc, opacities, scale, rotation, xyz
        gc.collect()

        print(f"Quantization: {time.time() - start_time:.2f} seconds")

        # === Dequantization phase ===
        start_time = time.time()
        features_extra = np.zeros((N, 45), dtype=np.float32)

        if not npz_data['sh_all_zero']:
            if bits_f_rest == 2:
                deq = asymmetric_dequantize_int2_segment_parallel(
                    npz_data['M_rest'], npz_data['S_rest'], npz_data['Z_rest'],
                    npz_data['segment_size_f_rest'], npz_data['N_rest']
                )
            elif bits_f_rest == 4:
                deq = asymmetric_dequantize_int4_segment_parallel(
                    npz_data['M_rest'], npz_data['S_rest'], npz_data['Z_rest'],
                    npz_data['segment_size_f_rest'], npz_data['N_rest']
                )
            elif bits_f_rest == 8:
                deq = asymmetric_dequantize_int8_per_segment_parallel(
                    npz_data['M_rest'], npz_data['S_rest'], npz_data['Z_rest'],
                    num_segments, npz_data['M_rest'].shape
                )
            elif bits_f_rest == 16:
                deq = npz_data['f_rest'].astype(np.float32)

            features_extra[:npz_data['boundary_index'], :] = deq

        features_extra = features_extra.reshape((N, 3, (self.max_sh_degree + 1) ** 2 - 1))

        # === Dequantize other attributes ===
        length_map = {'f_dc': 3, 'opacities': 1, 'scale': 3, 'rotation': 4}
        new_var = {}

        for bits, key in [(2, 'int2'), (4, 'int4'), (8, 'int8'), (16, 'fp16')]:
            names = npz_data.get(f'{key}_names', [])
            if not names:
                continue

            if bits in [2, 4, 8]:
                if bits == 2:
                    deq = asymmetric_dequantize_int2_segment_parallel(
                        npz_data['M_int2'], npz_data['S_int2'], npz_data['Z_int2'],
                        npz_data['segment_size_int2'], npz_data['N_int2']
                    )
                elif bits == 4:
                    deq = asymmetric_dequantize_int4_segment_parallel(
                        npz_data['M_int4'], npz_data['S_int4'], npz_data['Z_int4'],
                        npz_data['segment_size_int4'], npz_data['N_int4']
                    )
                else:
                    deq = asymmetric_dequantize_int8_per_segment_parallel(
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

        # === Convert dequantized features back to tensors ===
        features_dc = np.zeros((N, 3, 1), dtype=np.float32)
        features_dc[:, :, 0] = new_var['f_dc']

        self._xyz = nn.Parameter(torch.tensor(npz_data['xyz'].astype(np.float32), device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(new_var['opacities'], device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(new_var['scale'], device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(new_var['rotation'], device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

        print(f"Dequantization: {time.time() - start_time:.2f} seconds")

        return npz_data

    def load_quantization(self, path, prune_ratio, sh_ratio):
        npz_data = np.load(os.path.join(path, 'point_cloud', f'prune_{prune_ratio}_sh_{sh_ratio}', 'quantization_parameters_pipeline.npz'))
        N = npz_data['N']
        features_extra = np.zeros((N, 45))
        sh_all_zero = npz_data['sh_all_zero']
        num_segments = npz_data['num_segments']
        if not sh_all_zero:
            if npz_data['bits_f_rest'] == 4:
                boundary_index = npz_data['boundary_index']
                quantized_int4_f_rest = npz_data['M_rest']
                Z_int4_rest = npz_data['Z_rest']
                S_int4_rest = npz_data['S_rest']
                segment_size_f_rest = npz_data['segment_size_f_rest']
                N_rest = npz_data['N_rest']
                dequantized_features = asymmetric_dequantize_int4_segment_parallel(quantized_int4_f_rest, S_int4_rest, Z_int4_rest, segment_size_f_rest, N_rest)
                features_extra[:boundary_index,:] = dequantized_features
            elif npz_data['bits_f_rest'] == 8:
                quantized_rest_matrix_int8 = npz_data['M_rest']
                Z_rest_int8 = npz_data['Z_rest']
                S_rest_int8 = npz_data['S_rest']
                
                dequantized_features = asymmetric_dequantize_int8_per_segment_parallel(quantized_rest_matrix_int8, S_rest_int8, Z_rest_int8, num_segments, quantized_rest_matrix_int8.shape)
                features_extra[:boundary_index,:] = dequantized_features
            elif npz_data['bits_f_rest'] == 16:
                features_extra[:boundary_index,:] = npz_data['f_rest'].astype(np.float32)

            elif npz_data['bits_f_rest'] == 2: 
                # added bit 2
                boundary_index = npz_data['boundary_index']
                quantized_int4_f_rest = npz_data['M_rest']
                Z_int4_rest = npz_data['Z_rest']
                S_int4_rest = npz_data['S_rest']
                segment_size_f_rest = npz_data['segment_size_f_rest']
                N_rest = npz_data['N_rest']
                dequantized_features = asymmetric_dequantize_int4_segment_parallel(quantized_int4_f_rest, S_int4_rest, Z_int4_rest, segment_size_f_rest, N_rest)
                features_extra[:boundary_index,:] = dequantized_features


        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
    

        length_dic = {"f_dc" : 3, "opacities" : 1, "scale" : 3, "rotation" : 4}
        new_var_dic = {}
        bit_4_list = npz_data['int4_names']
        bit_8_list = npz_data['int8_names']
        bit_16_list = npz_data['fp16_names']

        if len(bit_4_list) != 0:
            Z_int4 = npz_data['Z_int4']
            S_int4 = npz_data['S_int4']
            quantized_int4 = npz_data['M_int4']
            segment_size_int4 = npz_data['segment_size_int4']
            N_int4 = npz_data['N_int4']
            unpacked_int4 = asymmetric_dequantize_int4_segment_parallel(quantized_int4, S_int4, Z_int4, segment_size_int4, N_int4)
            int_4_start = 0
            for name in bit_4_list:
                new_var_dic[name] = unpacked_int4[:, int_4_start:int_4_start + length_dic[name]]
                int_4_start += length_dic[name]

        if len(bit_8_list) != 0:
            
            Z_int8 = npz_data['Z_int8']
            S_int8 = npz_data['S_int8']
            quantized_int8 = npz_data['M_int8']

            unpacked_int8 = asymmetric_dequantize_int8_per_segment_parallel(quantized_int8, S_int8, Z_int8, num_segments, quantized_int8.shape)
            int_8_start = 0
            for name in bit_8_list:
                new_var_dic[name] = unpacked_int8[:, int_8_start:int_8_start + length_dic[name]]
                int_8_start += length_dic[name]

        if len(bit_16_list) != 0:
            fp_16_start = 0
            for name in bit_16_list:
                fp16_attributes = npz_data['M_fp16']
                new_var_dic[name] = fp16_attributes[:, fp_16_start:fp_16_start + length_dic[name]]
                fp_16_start += length_dic[name]
        
        

        features_dc = np.zeros((N, 3, 1))
        features_dc[:, 0, 0] = new_var_dic['f_dc'][:, 0]
        features_dc[:, 1, 0] = new_var_dic['f_dc'][:, 1]
        features_dc[:, 2, 0] = new_var_dic['f_dc'][:, 2]

        xyz = npz_data['xyz'].astype(np.float32)

        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(new_var_dic['opacities'], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(new_var_dic['scale'], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(new_var_dic['rotation'], dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
