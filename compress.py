#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 00:05:28 2024

@author: bytian
"""

import os
import sys
import torch
import time
import numpy as np
import shutil
import copy
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from prune import prune_list, calculate_v_imp_score
from modules import Pruner, Quantizer, Searcher


def timed_function(func):
    """Decorator to time a function, can be toggled on/off via self.enable_timing."""
    def wrapper(self, *args, **kwargs):
        if self.enable_timing:
            torch.cuda.synchronize()
            start_time = time.time()
        result = func(self, *args, **kwargs)
        if self.enable_timing:
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed_time = f"\033[2;32m{end_time - start_time:.4f} seconds\033[0m"  # Darker green
            print(f"\033[2;32m[TIMING]\033[0m {func.__name__}: {elapsed_time}")  # Darker green for entire line
        return result
    return wrapper

class Pipeline:
    def __init__(self, args, model_params, pipeline_params):
        self.args = args
        self.dataset = model_params.extract(args)
        self.pipe = pipeline_params.extract(args)
        self.gaussians = None
        self.scene = None
        self.enable_timing = False  # Set to False to disable timing
        self.imp_score = None
        self.filesize_input = 0
        self.filesize_output = 0
        self.closest_dic = None
        self.baseline_psnr = 0

    @timed_function
    def load_model_cameras(self):
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = Scene(self.dataset, self.gaussians)
        self.filesize_input = os.path.getsize(
            os.path.join(self.args.model_path, 'point_cloud/iteration_30000/point_cloud.ply'))

    @timed_function
    def compute_importance_scores(self):
        if hasattr(self.args, "imp_score_path") and self.args.imp_score_path and os.path.exists(self.args.imp_score_path):
            print(f"[INFO] Loading importance scores from: {self.args.imp_score_path}")
            data = np.load(self.args.imp_score_path)
            # Automatically get the first (and only) array
            if len(data.files) != 1:
                raise ValueError(f"Expected one array in {self.args.imp_score_path}, but found: {data.files}")
            self.imp_score = data[data.files[0]]

            print (self.imp_score.shape)

        else:
            print("[INFO] Computing importance scores...")
            v_pow = 0.1
            bg_color = torch.tensor(
                [1, 1, 1] if self.dataset.white_background else [0, 0, 0],
                dtype=torch.float32,
                device="cuda"
            )
            _, imp_list = prune_list(self.gaussians, self.scene, self.pipe, bg_color)
            v_list = calculate_v_imp_score(self.gaussians, imp_list, v_pow)
            self.imp_score = v_list.cpu().detach().numpy()

    @timed_function
    def store_model(self):
        if not self.closest_dic:
            print("[INFO] No valid configuration found within PSNR threshold.")
            return

        pruning_rate = self.closest_dic['pruning_rate']
        sh_rate = self.closest_dic['sh_rate']
        best_psnr = self.closest_dic['psnr']
        best_size = self.closest_dic['filesize'] / (1024 ** 2)  # in MB
        base_psnr = self.baseline_psnr
        input_mb = self.filesize_input / (1024 ** 2)
        compression_ratio = self.filesize_input / self.closest_dic['filesize']

        print()
        print("=" * 60)
        print(f"Best combination found: pruning={pruning_rate}, sh={sh_rate}")
        print()
        print(f"[INFO] Best config:           pruning_rate = {pruning_rate:.1f}, sh_rate = {sh_rate:.1f}")
        print(f"[INFO] Base PSNR:             {base_psnr:.4f} dB")
        print(f"[INFO] Best PSNR:             {best_psnr:.4f} dB")
        print(f"[INFO] PSNR drop:             {base_psnr - best_psnr:.4f} dB")
        print(f"[INFO] Input file size:       {input_mb:.2f} MB")
        print(f"[INFO] Output file size:      {best_size:.2f} MB")
        print(f"[INFO] Compression ratio:     {compression_ratio:.2f}x")
        print("=" * 60)

        # Save compressed model
        npz_path = self.closest_dic['npz_path']
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path, **self.closest_dic)

        if self.args.save_render:
            print(f" Saving render results ...")
            self.optimizer.spin_once(self.closest_dic['pruning_rate'], self.closest_dic['sh_rate'], save_render=True)

    @timed_function
    def run(self):
        t0 = time.time()

        self.load_model_cameras()

        t1 = time.time()

        self.compute_importance_scores()

        t2 = time.time()

        self.optimizer = Searcher(
            args=args,
            dataset=self.dataset,
            gaussians=self.gaussians,
            pipe=self.pipe,
            scene=self.scene,
            imp_score=self.imp_score,
            search_space='default',
            target_psnr_drop=args.quality_target_diff
        )
        self.closest_dic = self.optimizer.run_search()
        self.baseline_psnr = self.optimizer.baseline_psnr

        t3 = time.time()

        self.store_model()

        t4 = time.time()

        print("\n" + "=" * 60)
        print("[STATS] Pipeline Time Breakdown:")
        print(f"  Load time:    {t1 - t0:>6.2f} s")
        print(f"  Score time:   {t2 - t1:>6.2f} s")
        print(f"  Search time:  {t3 - t2:>6.2f} s")
        print(f"  Store time:   {t4 - t3:>6.2f} s")
        print(f"  Total time:   {t4 - t0:>6.2f} s")
        print("=" * 60)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--segments", default=1000, type=int)
    parser.add_argument("--save_render", default=False, type=bool)
    parser.add_argument("--quality_target_diff", default=1.0, type=float)
    parser.add_argument("--imp_score_path", type=str)

    args = get_combined_args(parser)
    args.data_device = "cuda"
    print("[ARGS] Device:", args.data_device)
    print("[ARGS] Input Model Path:", args.model_path)
    print("[ARGS] Save Render:", args.save_render)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Run pipeline
    Pipeline(args, model_params, pipeline_params).run()
