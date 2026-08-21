import argparse
import os
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "4"  # change as needed; export prior to ants import
# # For cluster runs, uncomment below
# n_threads = os.environ.get("SLURM_CPUS_PER_TASK", "1")
# os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = n_threads

import ants
import json
import subprocess
from pathlib import Path


def preprocess_t1(t1_path, output_dir, synthstrip_exe):
    """
    N4 bias correction with ANTs; skull-stripping using freesurfer Synthstrip (Hoopes et al, NeuroImage 2022).
    SYNTHSTRIP_EXE is a wrapper executable that utilizes SynthStrip container without full FreeSurfer install.
    """
    t1_path = Path(t1_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath('anat').mkdir(parents=False, exist_ok=True)

    n4_path = output_dir.joinpath("anat").joinpath("desc-N4_T1w.nii.gz")
    stripped_path = output_dir.joinpath("anat").joinpath("desc-brain_T1w.nii.gz")
    mask_path = output_dir.joinpath("anat").joinpath("desc-brain_mask.nii.gz")

    # 1. N4 bias correction with ANTsPy

    t1 = ants.image_read(str(t1_path))
    t1_n4 = ants.n4_bias_field_correction(t1)
    ants.image_write(t1_n4, str(n4_path))

    # 2. SynthStrip

    command = [
        str(synthstrip_exe),
        "-i", str(n4_path),
        "-o", str(stripped_path),
        "-m", str(mask_path),
    ]

    subprocess.run(
        command,
        check=True
    )

    # 3. Verify output exists
    if not stripped_path.exists():
        raise RuntimeError(
            f"SynthStrip did not create {stripped_path}"
        )

    if not mask_path.exists():
        raise RuntimeError(
            f"SynthStrip did not create {mask_path}"
        )

    return n4_path, stripped_path, mask_path


def register_t1_to_mni(stripped_t1_path, mni_template_path, output_dir):

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath('registrations').mkdir(parents=False, exist_ok=True)
    output_dir.joinpath('transforms').mkdir(parents=False, exist_ok=True)

    fixed = ants.image_read(str(mni_template_path))
    moving = ants.image_read(str(stripped_t1_path))

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        # type_of_transform="SyNRA",
        type_of_transform="antsRegistrationSyN[s]",
        outprefix=str(output_dir.joinpath("transforms").joinpath("t1_to_MNI_")),
        verbose=True,
    )

    metadata = {
        "moving_space": "subject_T1w",
        "fixed_space": "MNI152NLin2009bAsym",
        "template": "mni_icbm152_t1_nlin_asym_09b",
        "registration": "SyNRA",
        "software": "ANTsPyX",
        "forward_transforms": [
            "t1ToMNI_1Warp.nii.gz",
            "t1ToMNI_0GenericAffine.mat",
        ],
        "inverse_transforms": [
            "t1ToMNI_0GenericAffine.mat",
            "t1ToMNI_1InverseWarp.nii.gz",
        ],
    }

    with open("t1ToMNI_transform.json", "w") as f:
        json.dump(metadata, f, indent=2)

    ants.image_write(reg["warpedmovout"],
                     str(output_dir.joinpath("registrations").joinpath("t1_to_MNI_Warped.nii.gz")))

    ants.image_write(reg["warpedfixout"],
                     str(output_dir.joinpath("registrations").joinpath("t1_to_MNI_InverseWarped.nii.gz")))


def ct_to_t1(ct_path, t1_anat_path, output_dir):

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath('registrations').mkdir(parents=False, exist_ok=True)
    output_dir.joinpath('transforms').mkdir(parents=False, exist_ok=True)

    t1 = ants.image_read(str(t1_anat_path))
    ct = ants.image_read(str(ct_path))
    ct_to_t1 = ants.registration(
        fixed=t1,
        moving=ct,
        type_of_transform="Rigid",
        aff_metric="mattes",
        outprefix=str(output_dir.joinpath("transforms").joinpath("ct_to_t1_")),
        # verbose=True,
    )
    ants.image_write(ct_to_t1["warpedmovout"],
                     str(output_dir.joinpath("registrations").joinpath("postop_CT_T1_space.nii.gz")))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--preop_MRI_path", type=str, default=None)
    parser.add_argument("--postop_CT_path", type=str, default=None)
    parser.add_argument("--registration_template", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--synthstrip_exe_path", type=str, default=None)

    args = parser.parse_args()

    # Local testing only
    import caffeine
    caffeine.on(display=False)

    n4, stripped, mask = preprocess_t1(args.preop_MRI_path, args.output_dir, args.synthstrip_exe_path)
    register_t1_to_mni(stripped, args.registration_template, args.output_dir)
    ct_to_t1(args.postop_CT_path, args.preop_MRI_path, args.output_dir)

    # Local testing only
    caffeine.off()

'''
python3 image_preproc_register.py \
--preop_MRI_path /Users/arjit/Documents/_Lab/thalamic_stim/Colorado_thalamic_SEEG_data/Imaging/CUS001_22_08_01/anat_t1.nii \
--postop_CT_path /Users/arjit/Documents/_Lab/thalamic_stim/Colorado_thalamic_SEEG_data/Imaging/CUS001_22_08_01/postop_ct.nii \
--registration_template /Users/arjit/Documents/_Lab/thalamic_stim/mni_icbm152_t1_nlin_asym_09b_stripped.nii \
--output_dir /Users/arjit/Documents/_Lab/thalamic_stim/Colorado_thalamic_SEEG_data/derivatives/CUS001_22_08_01 \
--synthstrip_exe_path /Users/arjit/Documents/_Lab/thalamic_stim/synthstrip-docker

'''