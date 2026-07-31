"""
Code to derive structural connectivity measures from sEEG contact coordinates and normative HCP connectome.
Patient-specific DTI derived connectivity to be implemented.

args: INPUT_FILE

Key assumptions:


Requirements:
Intended use with python3.12
Likely compatible with python3.10+
dipy
nibabel
"""

import itertools
import json
import re
import shlex
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import argparse
from nibabel.affines import apply_affine
from dipy.tracking.utils import connectivity_matrix, length, subsegment
from helper_methods import getBipolarChannels, check_streamline_bounds, pad_inferior_z

# Edit variables below or pass as command line args
CONTACTS_TSV = None
TEMPLATE_NII = None
TRACTOGRAM_TRK = None
OUTPUT_DIR = None

RADIUS_MM = 3.0
MAX_SEGMENT_MM = 0.75


def safe_filename(name: str) -> str:
    """Convert a contact name into a safe filename."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
    return value.strip("_") or "contact"


def sphere_voxels(
        center_mm: np.ndarray,
        radius_mm: float,
        shape: tuple[int, int, int],
        affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Find voxels within sphere ROI
    :param center_mm:
    :param radius_mm:
    :param shape:
    :param affine:
    :return:
    voxels
        Integer array of shape (n_voxels, 3).
    distance_squared
        Squared world-space distance of each voxel center from the ROI center.
    '''

    inverse_affine = np.linalg.inv(affine)

    # Transform a world-space bounding cube into voxel space.
    offsets = np.array(
        list(itertools.product((-radius_mm, radius_mm), repeat=3)),
        dtype=float,
    )
    voxel_corners = apply_affine(
        inverse_affine,
        center_mm[None, :] + offsets,
    )

    lower = np.floor(voxel_corners.min(axis=0)).astype(int) - 1
    upper = np.ceil(voxel_corners.max(axis=0)).astype(int) + 1

    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, np.asarray(shape) - 1)

    if np.any(lower > upper):
        return np.empty((0, 3), dtype=int), np.empty(0)

    i, j, k = np.meshgrid(
        np.arange(lower[0], upper[0] + 1),
        np.arange(lower[1], upper[1] + 1),
        np.arange(lower[2], upper[2] + 1),
        indexing="ij",
    )

    voxels = np.column_stack(
        [i.ravel(), j.ravel(), k.ravel()]
    ).astype(int)

    world_coordinates = apply_affine(affine, voxels)
    distance_squared = np.sum(
        (world_coordinates - center_mm) ** 2,
        axis=1,
    )

    keep = distance_squared <= radius_mm ** 2
    return voxels[keep], distance_squared[keep]


def create_contact_rois(
        contacts: pd.DataFrame,
        reference_img: nib.spatialimages.SpatialImage,
        output_dir: Path,
        radius_mm: float,
        bipolar_convert=False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Create individual binary masks and one nonoverlapping label image.

    Overlapping voxels are assigned to the nearest contact center.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    shape = reference_img.shape[:3]
    affine = reference_img.affine
    header = reference_img.header.copy()

    label_volume = np.zeros(shape, dtype=np.int16)
    nearest_distance_squared = np.full(shape, np.inf, dtype=np.float32)

    if not bipolar_convert:
        nodes = contacts.copy().reset_index(drop=True)
        nodes["label_id"] = np.arange(1, len(nodes) + 1)
    else:
        bp_nodes = getBipolarChannels(contacts['name'])
        cc = contacts.set_index('name', inplace=False)
        bp_df = cc.reindex(bp_nodes.get('anode')).reset_index().join(cc.reindex(bp_nodes.get('cathode')).reset_index(),
                                                                     how='inner', lsuffix='_anode', rsuffix='_cathode')
        bp_df['name'] = bp_nodes.get('ch_name')
        bp_df['x'] = bp_df[['x_anode', 'x_cathode']].mean(axis=1)
        bp_df['y'] = bp_df[['y_anode', 'y_cathode']].mean(axis=1)
        bp_df['z'] = bp_df[['z_anode', 'z_cathode']].mean(axis=1)
        nodes = bp_df[['name', 'x', 'y', 'z']]
        nodes["label_id"] = np.arange(1, len(nodes) + 1)

    for row_index, row in nodes.iterrows():
        label_id = int(row["label_id"])
        center = row[["x", "y", "z"]].to_numpy(dtype=float)

        voxels, distances_squared = sphere_voxels(center_mm=center, radius_mm=radius_mm, shape=shape, affine=affine)

        if len(voxels) == 0:
            raise ValueError(f"No valid voxels for contact {row['name']}.")

        i, j, k = voxels.T

        # Save the individual binary ROI.
        binary_mask = np.zeros(shape, dtype=np.uint8)
        binary_mask[i, j, k] = 1

        mask_header = header.copy()
        mask_header.set_data_dtype(np.uint8)

        mask_path = output_dir.joinpath(f"label-{label_id:03d}_{safe_filename(row['name'])}_mask.nii.gz")

        nib.save(nib.Nifti1Image(binary_mask, affine, mask_header), mask_path)

        # For the combined atlas, assign overlap voxels to the closest center.
        previous_distance = nearest_distance_squared[i, j, k]
        replace = distances_squared < previous_distance

        label_volume[
            i[replace],
            j[replace],
            k[replace],
        ] = label_id

        nearest_distance_squared[
            i[replace],
            j[replace],
            k[replace],
        ] = distances_squared[replace]

        nodes.loc[row_index, "roi_mask"] = mask_path.name
        nodes.loc[row_index, "binary_roi_voxels"] = len(voxels)

    atlas_header = header.copy()
    atlas_header.set_data_dtype(np.int16)

    atlas_path = output_dir.joinpath(f"contacts_radius-{radius_mm:g}mm_dseg.nii.gz")

    nib.save(
        nib.Nifti1Image( label_volume, affine, atlas_header),
        atlas_path,
    )

    nodes["atlas_roi_voxels"] = [
        int(np.count_nonzero(label_volume == label_id))
        for label_id in nodes["label_id"]
    ]

    if (nodes["atlas_roi_voxels"] == 0).any():
        missing = nodes.loc[
            nodes["atlas_roi_voxels"] == 0,
            "name",
        ].tolist()
        raise RuntimeError(
            f"Contacts disappeared because of ROI overlap: {missing}"
        )

    nodes.to_csv(output_dir.joinpath("contact_nodes.tsv"), sep="\t", index=False)

    return label_volume, nodes


def compute_connectivity(
        tractogram_path: Path,
        label_volume: np.ndarray,
        affine: np.ndarray,
        nodes: pd.DataFrame,
        output_dir: Path,
        max_segment_mm: float,
):
    """
    Calculate pass-through connectivity.

    A streamline contributes once to every unordered pair of contact ROIs
    that it intersects.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tract_file = nib.streamlines.load(str(tractogram_path), lazy_load=True)  # keep in mind lazy-load
    # tract_file.tractogram.streamlines is a generator, do not attempt to re-use

    # Record original streamline lengths.
    streamline_lengths_mm = np.fromiter(
        length(tract_file.tractogram.streamlines),  # this is dipy.tracking.utils.length, which returns an iterable map
        dtype=np.float64,
    )

    # #  Debugging
    # check_streamline_bounds(
    #     subsegment(
    #         tract_file.tractogram.streamlines,
    #         max_segment_length=max_segment_mm,
    #     ),
    #     affine,
    #     label_volume.shape,
    # ) Needed padding of 16 voxels

    # Add intermediate points so streamline segments do not skip small ROIs.
    dense_streamlines_gen = subsegment(tract_file.tractogram.streamlines, max_segment_length=max_segment_mm)

    padded_labels, padded_affine = pad_inferior_z(
        label_volume,
        affine,
        pad_voxels=16,
    )

    counts2D, edge_map = connectivity_matrix(dense_streamlines_gen,
                                             affine=padded_affine,
                                             label_volume=padded_labels,
                                             inclusive=True,
                                             symmetric=True,
                                             return_mapping=True)

    # QC Check
    print(edge_map.get((0,0), None))
    print(edge_map.get((10,10), None))


    mean_length = np.zeros(counts2D.shape)
    median_length = np.zeros(counts2D.shape)
    sum_inv_length = np.zeros(counts2D.shape)


    for (i, j), indices in edge_map.items():
        slens = streamline_lengths_mm[indices]

        ml = np.mean(slens)
        mean_length[i][j] = ml
        mean_length[j][i] = ml

        mn = np.median(slens)
        median_length[i][j] = mn
        median_length[j][i] = mn

        si = np.reciprocal(slens).sum()
        sum_inv_length[i][j] = si
        sum_inv_length[j][i] = si



    centers = nodes[["x", "y", "z"]].to_numpy(dtype=float)

    euclidean_distance = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :],
        axis=2,
    )

    matrices = {
        "streamline_count": counts2D[1:, 1:],
        "mean_streamline_length_mm": mean_length[1:, 1:],
        "median_streamline_length_mm": median_length[1:, 1:],
        "sum_inverse_streamline_length": sum_inv_length[1:, 1:],
        "euclidean_contact_distance_mm": euclidean_distance,
    }

    names = nodes["name"].astype(str).tolist()

    for matrix_name, matrix in matrices.items():
        pd.DataFrame(matrix, index=names, columns=names).to_csv(output_dir.joinpath(f"{matrix_name}.csv"))

    np.savez_compressed(
        output_dir.joinpath("connectivity_matrices.npz"),
        names=np.asarray(names),
        centers_mm=centers,
        **matrices,
    )

    n_streamlines = len(streamline_lengths_mm)
    connects = np.zeros(n_streamlines, dtype=bool)
    for (i, j), indices in edge_map.items():
        if i > 0 and j > 0:
            connects[indices] = True
    n_streamlines_intersecting_2plus_rois = int(connects.sum())

    metadata = {
        "tractogram": str(tractogram_path),
        "total_streamlines": len(streamline_lengths_mm),
        "streamlines_intersecting_two_or_more_rois": n_streamlines_intersecting_2plus_rois,
        "roi_radius_mm": RADIUS_MM,
        "maximum_resampled_segment_mm": max_segment_mm,
        "connectivity_definition": (
            "Pass-through streamline intersection with "
            "nonoverlapping voxelized contact ROIs."
        ),
    }

    output_dir.joinpath("connectivity_metadata.json").write_text(json.dumps(metadata, indent=2))


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    contacts = pd.read_csv(CONTACTS_TSV, sep="\t")

    required_columns = {"name", "x", "y", "z"}
    missing = required_columns - set(contacts.columns)

    if missing:
        raise ValueError(f"Missing contact columns: {sorted(missing)}")

    contacts[["x", "y", "z"]] = contacts[["x", "y", "z"]].apply(pd.to_numeric, errors="raise")

    template_img = nib.load(TEMPLATE_NII)

    label_volume, nodes = create_contact_rois(
        contacts=contacts,
        reference_img=template_img,
        output_dir=OUTPUT_DIR.joinpath("rois"),
        radius_mm=RADIUS_MM,
        bipolar_convert=True,
    )

    compute_connectivity(
        tractogram_path=TRACTOGRAM_TRK,
        label_volume=label_volume,
        affine=template_img.affine,
        nodes=nodes,
        output_dir=OUTPUT_DIR.joinpath("matrices"),
        max_segment_mm=MAX_SEGMENT_MM,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--contacts_coordinates", type=str, default='')
    parser.add_argument("--template_image", type=str, default='')
    parser.add_argument("--tractogram", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='')

    args = parser.parse_args()

    if CONTACTS_TSV is None:
        CONTACTS_TSV = args.contacts_coordinates
    if TEMPLATE_NII is None:
        TEMPLATE_NII = args.template_image
    if TRACTOGRAM_TRK is None:
        TRACTOGRAM_TRK = args.tractogram
    if OUTPUT_DIR is None:
        OUTPUT_DIR = args.output_dir

    if not (CONTACTS_TSV and TEMPLATE_NII and TRACTOGRAM_TRK and OUTPUT_DIR):
        raise ValueError("Missing arguments! Need input file locations for contact coordinates, template image, "
                         "normative tractogram, and output directory ")

    CONTACTS_TSV = Path(" ".join(shlex.split(CONTACTS_TSV)))
    TEMPLATE_NII = Path(" ".join(shlex.split(TEMPLATE_NII)))
    TRACTOGRAM_TRK = Path(" ".join(shlex.split(TRACTOGRAM_TRK)))
    OUTPUT_DIR = Path(" ".join(shlex.split(OUTPUT_DIR)))

    run()


