import re
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

map_int_to_anat = dict()

def _initialize_map(lut_path, two_labels_col):
    global map_int_to_anat
    anat_col = 2 if two_labels_col else 1
    lut_rows = []
    with open(lut_path, 'r') as f:
        lut_rows = [ll.strip() for ll in f.readlines() if (ll.strip() and
                                                           not ll.startswith("#"))]
    map_int_to_anat = {int(ll.split()[0]): ll.split()[anat_col] for ll in lut_rows}


def get_voi_overlap(voi_path, seg_path):
    voi_img = nib.load(voi_path)
    seg_img = nib.load(seg_path)

    if voi_img.shape != seg_img.shape:
        # Put categorical segmentation onto the VOI grid.
        # order=0 = nearest-neighbor, so label integers are preserved.
        seg_img = resample_from_to(
            seg_img,
            voi_img,
            order=0
        )

    if not np.allclose(voi_img.affine, seg_img.affine, atol=1e-3):
        raise ValueError("VOI and segmentation affines differ")

    voi = np.asarray(voi_img.dataobj) > 0
    seg = np.asarray(seg_img.dataobj).astype(int)

    labels = seg[voi]

    counts = Counter(labels)
    total = len(labels)

    return [
        {
            "label": map_int_to_anat.get(int(label_id)),
            "n_voxels": count,
            "fraction": count / total
        }
        for label_id, count in counts.most_common(n=5) if label_id != 0
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--VOIs_path", type=str, default="")
    parser.add_argument("--LUT_file", type=str, default="")
    parser.add_argument("--two_labels_col", action="store_true")
    parser.add_argument("--segmentation_path", type=str, default="")
    parser.add_argument("--output_prefix", type=str, default="")
    parser.add_argument("--output_directory", type=str, default=None)
    args = parser.parse_args()

    _initialize_map(args.LUT_file, args.two_labels_col)

    vpath = Path(args.VOIs_path)
    out_path = Path(args.output_directory) if args.output_directory else vpath.parent

    contact_overlaps_w_regions = dict()
    for mask in vpath.iterdir():

        m = re.match("contact_(\\d+)_mask.nii.gz", mask.name)
        if not m:
            continue
        num = int(m.group(1))

        overlaps = get_voi_overlap(mask, args.segmentation_path)

        for (ii, oo) in enumerate(overlaps):
            contact_overlaps_w_regions[(num, ii)] =  oo

    rows = {}

    for (contact, region), oo in contact_overlaps_w_regions.items():
        for metric in ["label", "n_voxels", "fraction"]:
            rows.setdefault((contact, metric),{})[f"Region {region + 1}"] = oo[metric]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["Contact", "Metric"])
    df = df.sort_index()

    df.to_csv(out_path.joinpath(f"{args.output_prefix}_contact_labels.csv"))


