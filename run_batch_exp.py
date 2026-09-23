#!/usr/bin/env python3

import os
import sys
import gc
import yaml
import torch
import argparse
import copy
import sys
import tempfile
import yaml
from pathlib import Path

from os.path import join, isdir, abspath, dirname, basename, splitext
from datetime import datetime
from shutil import copy as shut_copy

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler
from src.pipeline import StableSyncMVDPipeline
from src.configs import parse_config


def load_opt_from_config(config_path):
    config_path = Path(config_path).resolve()

    # Get the original parser's defaults using an empty config.
    # Provide dummy values for the required mesh and prompt arguments.
    if not hasattr(load_opt_from_config, "_defaults"):
        with tempfile.TemporaryDirectory() as tmp:
            empty_config = Path(tmp) / "empty.yaml"
            empty_config.write_text("")

            old_argv = sys.argv[:]

            try:
                sys.argv = [
                    sys.argv[0],
                    "--config", str(empty_config),
                    "--mesh", "dummy.obj",
                    "--prompt", "dummy",
                ]

                load_opt_from_config._defaults = parse_config()

            finally:
                sys.argv = old_argv

    opt = copy.deepcopy(load_opt_from_config._defaults)

    # Read YAML without converting its contents into CLI arguments.
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping: {config_path}")

    for key, value in config.items():
        if not hasattr(opt, key):
            raise ValueError(f"Unknown config option: {key}")
        setattr(opt, key, value)

    opt.config = str(config_path)

    return opt


def build_syncmvd(cond_type):
    """
    Load ControlNet + SD pipeline once for a given cond_type.
    """
    if cond_type == "normal":
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_normalbae",
            variant="fp16",
            torch_dtype=torch.float16,
        )
    elif cond_type == "depth":
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth",
            variant="fp16",
            torch_dtype=torch.float16,
        )
    else:
        raise ValueError(f"Unsupported cond_type: {cond_type}")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "benjamin-paine/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )

    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Optional: move to GPU explicitly if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    syncmvd = StableSyncMVDPipeline(**pipe.components)
    return syncmvd


def run_one_config(opt, syncmvd):
    """
    Run a single job using an already-loaded syncmvd pipeline.
    This is basically your original script body.
    """
    if opt.mesh_config_relative:
        mesh_path = join(dirname(opt.config), opt.mesh)
    else:
        mesh_path = abspath(opt.mesh)

    if opt.output:
        output_root = abspath(opt.output)
    else:
        output_root = dirname(opt.config)

    output_name_components = []
    if getattr(opt, "prefix", None) and opt.prefix != "":
        output_name_components.append(opt.prefix)

    if getattr(opt, "use_mesh_name", False):
        mesh_name = splitext(basename(mesh_path))[0].replace(" ", "_")
        output_name_components.append(mesh_name)

    if getattr(opt, "timeformat", None) and opt.timeformat != "":
        output_name_components.append(datetime.now().strftime(opt.timeformat))

    output_name = "_".join(output_name_components)

    # fallback if prefix/timeformat empty
    if output_name == "":
        output_name = "output"

    output_dir = join(output_root, output_name)

    if not isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    else:
        print(f"[SKIP] Results already exist in {output_dir}")
        return

    print(f"Saving to {output_dir}")
    shut_copy(opt.config, join(output_dir, "config.yaml"))

    logging_config = {
        "output_dir": output_dir,
        "log_interval": opt.log_interval,
        "view_fast_preview": opt.view_fast_preview,
        "tex_fast_preview": opt.tex_fast_preview,
    }

    print(f"Running mesh: {mesh_path}")
    print(f"Prompt: {opt.prompt}")
    print(f"keep_mesh_uv: {opt.keep_mesh_uv}  -> mesh_autouv: {not opt.keep_mesh_uv}")

    result_tex_rgb, textured_views, v = syncmvd(
        prompt=opt.prompt,
        height=opt.latent_view_size * 8,
        width=opt.latent_view_size * 8,
        num_inference_steps=opt.steps,
        guidance_scale=opt.guidance_scale,
        negative_prompt=opt.negative_prompt,

        generator=torch.manual_seed(opt.seed),
        max_batch_size=48,
        controlnet_guess_mode=opt.guess_mode,
        controlnet_conditioning_scale=opt.conditioning_scale,
        controlnet_conditioning_end_scale=opt.conditioning_scale_end,
        control_guidance_start=opt.control_guidance_start,
        control_guidance_end=opt.control_guidance_end,
        guidance_rescale=opt.guidance_rescale,
        use_directional_prompt=True,

        mesh_path=mesh_path,
        mesh_transform={"scale": opt.mesh_scale},
        mesh_autouv=not opt.keep_mesh_uv,

        camera_azims=opt.camera_azims,
        top_cameras=not opt.no_top_cameras,
        texture_size=opt.latent_tex_size,
        render_rgb_size=opt.rgb_view_size,
        texture_rgb_size=opt.rgb_tex_size,
        multiview_diffusion_end=opt.mvd_end,
        exp_start=opt.mvd_exp_start,
        exp_end=opt.mvd_exp_end,
        ref_attention_end=opt.ref_attention_end,
        shuffle_background_change=opt.shuffle_bg_change,
        shuffle_background_end=opt.shuffle_bg_end,

        logging_config=logging_config,
        cond_type=opt.cond_type,
        flat_cond=opt.flat_cond,
    )

    # optional cleanup between jobs
    del result_tex_rgb, textured_views, v
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[DONE] {opt.config}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-root",
        type=str,
        required=True,
        help="Root folder containing per-scan config YAMLs",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="config_flat_cond.yaml",
        help="Config filename pattern to search for",
    )
    args = parser.parse_args()

    config_root = abspath(args.config_root)

    config_paths = []
    for root, dirs, files in os.walk(config_root):
        for fname in files:
            if fname == args.pattern:
                config_paths.append(join(root, fname))

    config_paths = sorted(config_paths)

    if len(config_paths) == 0:
        print(f"No configs found under {config_root} with name {args.pattern}")
        return

    print(f"Found {len(config_paths)} configs")

    # Cache pipelines by cond_type so if later you mix "normal" and "depth",
    # each one is only loaded once.
    pipeline_cache = {}

    for config_path in config_paths:
        try:
            opt = load_opt_from_config(config_path)

            # Important for registered scans converted from PLY -> OBJ
            # if UVs are not present, set keep_mesh_uv = False in config
            cond_type = opt.cond_type

            if cond_type not in pipeline_cache:
                print(f"Loading pipeline for cond_type={cond_type}")
                pipeline_cache[cond_type] = build_syncmvd(cond_type)

            syncmvd = pipeline_cache[cond_type]
            run_one_config(opt, syncmvd)

        except Exception as e:
            print(f"[FAILED] {config_path}")
            print(e)

    print("All jobs processed.")


if __name__ == "__main__":
    main()