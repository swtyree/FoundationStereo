# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import png

import tqdm
from omegaconf import OmegaConf

project_path = Path(__file__).resolve().parent.parent
sys.path.append(project_path.as_posix())
# slower imports for PyTorch and Open3D are made after argument parsing

# ignore warnings about `xformers` package
warnings.filterwarnings("ignore", module="dinov2")


if __name__ == "__main__":
    # parse arguments
    # TODO organize these
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--left_dir",
        type=Path,
        required=True,
        help="directory of left image files",
    )
    parser.add_argument(
        "--right_dir",
        type=Path,
        required=True,
        help="directory of right image files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="the directory to save results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="enable verbose logging (-vv for debug messages)",
    )

    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        required=True,
        help="camera intrinsics [fx, fy, cx, cy]",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        required=True,
        help="stereo baseline, typically in meters",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="scale factor for converting depth to uint16",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="downsize the image by scale, must be <=1",
    )
    parser.add_argument(
        "--upsample_depth_image",
        action="store_true",
        help="upsample depth map to original image size before saving",
    )
    parser.add_argument(
        "--hiera",
        action="store_true",
        help="hierarchical inference (only needed for high-resolution images (>1K))",
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="skip every Nth image",
    )
    parser.add_argument(
        "--save_pc",
        action="store_true",
        help="save point cloud output",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="save visualization",
    )

    parser.add_argument(
        "--ckpt_file",
        type=Path,
        default=project_path / "pretrained_models/23-51-11/model_best_bp2.pth",
        help="pretrained model path",
    )
    parser.add_argument(
        "--z_far",
        type=float,
        default=10.0,
        help="max depth to clip in point cloud",
    )
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of flow-field updates during forward pass",
    )
    parser.add_argument(
        "--keep_invisible",
        action="store_true",
        help="keep non-overlapping observations between left and right images from point cloud",
    )
    args = parser.parse_args()

    # perform slower imports
    import torch
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder
    from Utils import (
        depth2xyzmap,
        set_logging_format,
        set_seed,
        toOpen3dCloud,
        vis_disparity,
    )

    if args.save_pc:
        import open3d as o3d

    # set logging format
    set_logging_format(
        logging.DEBUG
        if args.verbose > 1
        else logging.INFO if args.verbose else logging.WARNING
    )
    logging.info(f"Arguments:\n{json.dumps(args.__dict__, indent=4, default=str)}")
    np.set_printoptions(suppress=True)  # suppress scientific notation

    # check args
    assert args.scale <= 1 and args.scale > 0, "scale must be <=1 and >0"

    # get image filenames
    assert args.left_dir.is_dir(), f"Left directory {args.left_dir} does not exist"
    assert args.right_dir.is_dir(), f"Right directory {args.right_dir} does not exist"
    left_fns = [f for f in args.left_dir.iterdir() if f.suffix in [".png", ".jpg"]]
    right_fns = [f for f in args.right_dir.iterdir() if f.suffix in [".png", ".jpg"]]
    assert len(left_fns) == len(right_fns), "Number of left and right images must match"
    if len(left_fns) == 0:
        logging.warning(f"No images found in {args.left_dir} and {args.right_dir}")
        sys.exit(1)
    logging.debug(f"Found {len(left_fns)} image pairs")
    left_fns = sorted(left_fns)[:: args.skip]
    right_fns = sorted(right_fns)[:: args.skip]
    logging.debug(f"Using every {args.skip}th image pair")
    logging.info(f"Processing {len(left_fns)} image pairs")

    # create output directory
    if not args.output_dir.exists():
        logging.info(f"Creating output directory {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        logging.info(f"Using existing output directory {args.output_dir}")

    # get intrinsics
    fx, fy, cx, cy = args.intrinsics
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    K[:2] *= args.scale
    logging.info(f"Camera intrinsics:\n{K}")

    # load model
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    logging.debug(f"Loading model config from {args.ckpt_file.parent / 'cfg.yaml'}")
    cfg = OmegaConf.load(args.ckpt_file.parent / "cfg.yaml")
    model = FoundationStereo(cfg)
    logging.info(f"Loading model weights from {args.ckpt_file}")
    ckpt = torch.load(args.ckpt_file, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()

    # initialize depth writer
    image_size = imageio.imread(left_fns[0]).shape[:2]
    if args.scale < 1 and not args.upsample_depth_image:
        image_size = tuple(int(s * args.scale) for s in image_size)
    depth_writer = png.Writer(
        width=image_size[1],
        height=image_size[0],
        greyscale=True,
        bitdepth=16,
    )  # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite()

    # process image pairs
    for left_fn, right_fn in zip(tqdm.tqdm(left_fns), right_fns):
        # load images
        logging.info(f"Loading {left_fn} and {right_fn}")
        im_left = imageio.imread(left_fn, pilmode="RGB")  # ignore alpha channel
        im_right = imageio.imread(right_fn, pilmode="RGB")
        assert (
            im_left.shape == im_right.shape
        ), f"Image shape mismatch: {left_fn} ({im_left.shape}) and {right_fn} ({im_right.shape})"
        logging.debug(f"Image shape is {im_left.shape}")

        # resize images
        if args.scale < 1:
            logging.debug("Resizing images")
            im_left = cv2.resize(im_left, fx=args.scale, fy=args.scale, dsize=None)
            im_right = cv2.resize(im_right, fx=args.scale, fy=args.scale, dsize=None)
            logging.debug(f"Resized images to shape {im_left.shape}")
        H, W = im_left.shape[:2]

        # create image input tensors
        logging.debug("Creating image input tensors")
        im_left_t = torch.as_tensor(im_left).cuda().float()[None].permute(0, 3, 1, 2)
        im_right_t = torch.as_tensor(im_right).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(im_left_t.shape, divis_by=32, force_square=False)
        im_left_t, im_right_t = padder.pad(im_left_t, im_right_t)

        # run inference
        with torch.amp.autocast("cuda", enabled=True):
            if not args.hiera:
                logging.debug("Running non-hierarchical inference")
                disp = model.forward(
                    im_left_t, im_right_t, iters=args.valid_iters, test_mode=True
                )
            else:
                logging.debug("Running hierarchical inference")
                disp = model.run_hierachical(
                    im_left_t,
                    im_right_t,
                    iters=args.valid_iters,
                    test_mode=True,
                    small_ratio=0.5,
                )

        # process output
        logging.debug("Unpadding output")
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)

        # remove invisible points
        if not args.keep_invisible:
            logging.debug("Removing invisible points")
            yy, xx = np.meshgrid(
                np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij"
            )
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf

        # compute depth map
        logging.debug("Computing depth map")
        depth_map_fn = args.output_dir / left_fn.with_suffix(".png").name
        depth_map = K[0, 0] * args.baseline / disp

        # save depth map
        depth_image = np.round(depth_map * args.depth_scale).astype(np.uint16)
        if args.scale < 1.0 and args.upsample_depth_image:
            logging.debug("Upsampling depth map")
            depth_image = cv2.resize(
                depth_image,
                dsize=None,
                fx=1 / args.scale,
                fy=1 / args.scale,
                interpolation=cv2.INTER_NEAREST,
            )
        with open(depth_map_fn, "wb") as fp:
            depth_writer.write(fp, depth_image)
        logging.info(
            f"Saved depth map to {depth_map_fn} with shape {depth_image.shape}"
        )

        # save visualization
        if args.save_vis:
            logging.debug("Creating visualization")
            vis = vis_disparity(disp)
            vis = np.concatenate([im_left, vis], axis=1)
            vis_fn = (
                args.output_dir / left_fn.with_suffix(".jpg").name
            )  # use JPG to avoid name conflicts
            imageio.imwrite(vis_fn, vis)
            logging.info(f"Visualization saved to {vis_fn}")

        # save point cloud
        if args.save_pc:
            logging.debug("Computing point cloud")
            xyz_map = depth2xyzmap(depth_map, K)
            point_cloud = toOpen3dCloud(xyz_map.reshape(-1, 3), im_left.reshape(-1, 3))
            keep_mask = (np.asarray(point_cloud.points)[:, 2] > 0) & (
                np.asarray(point_cloud.points)[:, 2] <= args.z_far
            )
            keep_ids = np.arange(len(np.asarray(point_cloud.points)))[keep_mask]
            point_cloud = point_cloud.select_by_index(keep_ids)
            point_cloud_fn = args.output_dir / left_fn.with_suffix(".ply").name
            o3d.io.write_point_cloud(point_cloud_fn, point_cloud)
            logging.info(
                f"Point cloud saved to {point_cloud_fn} with {len(keep_ids)} points"
            )
