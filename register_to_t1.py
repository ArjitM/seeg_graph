
"""
Assumes t1 image has already been preprocessed, skull-stripped, etc via image_preproc_register.py
"""

import argparse
import os
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "4"  # change as needed; export prior to ants import
# For cluster runs, uncomment below
n_threads = os.environ.get("SLURM_CPUS_PER_TASK", "1")
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = n_threads

import ants
import json
import subprocess
from pathlib import Path
from image_preproc_register import preprocess_t1


def image_to_t1(img_path, t1_anat_path, output_dir, prefix='img'):

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath('registrations').mkdir(parents=False, exist_ok=True)
    output_dir.joinpath('transforms').mkdir(parents=False, exist_ok=True)

    t1 = ants.image_read(str(t1_anat_path))
    img = ants.image_read(str(img_path))
    img_to_t1 = ants.registration(
        fixed=t1,
        moving=img,
        type_of_transform="Rigid",
        aff_metric="mattes",
        outprefix=str(output_dir.joinpath("transforms").joinpath(f"{prefix}_to_t1_")),
        # verbose=True,
    )
    ants.image_write(img_to_t1["warpedmovout"],
                     str(output_dir.joinpath("registrations").joinpath(f"{prefix}_T1_space.nii.gz")))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--preop_MRI_path", type=str, default=None)
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--registration_template", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--synthstrip_exe_path", type=str, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--preprocess", action='store_true')

    args = parser.parse_args()

    # # Local testing only
    # import caffeine
    # caffeine.on(display=False)

    output_dir = Path(args.output_dir).resolve()
    if output_dir.joinpath("anat").joinpath("desc-brain_mask.nii.gz").exists():
        t1_brain = output_dir.joinpath("anat").joinpath("desc-brain_mask.nii.gz")

    else:
        if args.preprocess:
            _, t1_brain, _ = preprocess_t1(args.preop_MRI_path, args.output_dir, args.synthstrip_exe_path)
        else:
            raise ValueError("Skull-stripped MRI not found")

    if args.prefix:
        image_to_t1(args.img_path, t1_brain, args.output_dir, prefix=args.prefix)
    else:
        image_to_t1(args.img_path, t1_brain, args.output_dir)

    # # Local testing only
    # caffeine.off()
