import numpy as np
import re
from nibabel.affines import apply_affine


def getBipolarChannels(channels: list[str], allow_missing=True, exclude_pairs=None):

    # can handle space or no space prior to contact name, but expects LEAD<CONTACT_NUM> format
    c_id = [(c, c.split(' ')[-1]) for c in channels]

    ch_names, contacts = zip(*c_id)
    ch_name_by_contact = dict({*zip(contacts, ch_names)})
    pattern = re.compile(r"([A-Za-z]+)(\d+)$")
    contacts_by_lead = {}
    exclude_contacts_by_lead = {}

    if exclude_pairs is not None:
        pair_pat = re.compile(r"([A-Za-z]+)(\d+)-(\d+)")
        for ep in exclude_pairs:
            m = pair_pat.match(ep)
            ll, c1, c2 = m.groups()
            if abs(c2 - c1) == 1:
                exclude_contacts_by_lead[ll] = exclude_contacts_by_lead.get(ll, list())
                exclude_contacts_by_lead[ll].append(int(c1))

    for c in contacts:
        m = pattern.match(c)
        if m:
            c_id = m.group(1)
            contacts_by_lead[c_id] = contacts_by_lead.get(c_id, list())

            cntct  = int(m.group(2))
            if cntct not in exclude_contacts_by_lead.get(c_id, list()):
                contacts_by_lead[c_id].append(cntct)

    for lead in list(contacts_by_lead.keys()):  # keep list() expression to avoid write while iterating
        contacts_by_lead[lead] = sorted(contacts_by_lead.get(lead))

    bp_args = {
        'anode': list(),
        'cathode': list(),
        'ch_name': list(),
    }

    for lead, contacts in contacts_by_lead.items():
        '''
        contacts are contiguous iff np.unique(np.diff(contacts)).shape[0] == 1

        example case 
        >>> non_contig = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15]  # 6, 10 missing
        then
        >>> cl = np.where(np.diff(non_contig) == 1)[0]
        >>> cl
        array([ 0,  1,  2,  3,  5,  6,  8,  9, 10, 11])
        >>> [non_contig[c] for c in cl]
        [1, 2, 3, 4, 7, 8, 11, 12, 13, 14]'

        Note that 4-5 is OK; 5-6 and 6-7 do not exist. Likewise for 8-9 OK. 9-10, 10-11 do not exist. 
        This is the desired behavior for missing contacts 6 and 10 when ALLOW_MISSING is set to TRUE.
        '''

        if not allow_missing and np.unique(np.diff(contacts)).shape[0] != 1:
            raise ValueError("missing contacts")

        contig_locations = np.where(np.diff(contacts) == 1)[0]
        to_use = [contacts[ii] for ii in contig_locations]

        bp_args['anode'] += [ch_name_by_contact.get(f'{lead}{c}') for c in to_use]
        bp_args['cathode'] += [ch_name_by_contact.get(f'{lead}{c + 1}') for c in to_use]
        bp_args['ch_name'] += [f'{lead}_{c}-{c + 1}' for c in to_use]

    return bp_args


def check_streamline_bounds(
    streamlines,
    affine: np.ndarray,
    shape: tuple[int, int, int],
) -> None:
    inverse_affine = np.linalg.inv(affine)
    shape_array = np.asarray(shape)

    global_min = np.full(3, np.inf)
    global_max = np.full(3, -np.inf)

    n_negative = 0
    n_above = 0
    n_total = 0

    for streamline in streamlines:
        streamline = np.asarray(streamline)

        voxel_coords = apply_affine(
            inverse_affine,
            streamline,
        )

        global_min = np.minimum(
            global_min,
            voxel_coords.min(axis=0),
        )
        global_max = np.maximum(
            global_max,
            voxel_coords.max(axis=0),
        )

        n_negative += int(np.any(voxel_coords < -0.5))
        n_above += int(
            np.any(voxel_coords >= shape_array - 0.5)
        )
        n_total += 1

    print("Label shape:", shape)
    print("Minimum mapped voxel:", global_min)
    print("Maximum mapped voxel:", global_max)
    print("Negative-bound streamlines:", n_negative)
    print("Upper-bound streamlines:", n_above)
    print("Total streamlines:", n_total)


def pad_inferior_z(
    label_volume: np.ndarray,
    affine: np.ndarray,
    pad_voxels: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    padded_volume = np.pad(
        label_volume,
        pad_width=(
            (0, 0),           # x
            (0, 0),           # y
            (pad_voxels, 0),  # z: pad before only
        ),
        mode="constant",
        constant_values=0,
    )

    # New voxel [0, 0, pad_voxels] must map to the same
    # world position as old voxel [0, 0, 0].
    voxel_shift = np.eye(4)
    voxel_shift[2, 3] = -pad_voxels

    padded_affine = affine @ voxel_shift

    return padded_volume, padded_affine