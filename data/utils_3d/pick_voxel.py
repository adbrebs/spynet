__author__ = 'adeb'

import numpy as np

from spynet.utils.utilities import distrib_balls_in_bins


def create_pick_voxel(config_ini):
    """
    Factory function to create the objects responsible for picking the voxels
    """
    where_vx = config_ini.pick_vx["where"]
    how_vx = config_ini.pick_vx["how"]
    if where_vx == "anywhere":
        select_region = SelectWholeBrain()
    elif where_vx == "plane":
        axis = config_ini.pick_vx["axis"]
        plane = config_ini.pick_vx["plane"]
        select_region = SelectPlane(axis, plane)
    else:
        print "error in pick_voxel"
        return

    n_patch_per_voxel = config_ini.general["n_patch_per_voxel"]

    if how_vx == "all":
        extract_voxel = ExtractVoxelAll(n_patch_per_voxel)
    else:
        if how_vx == "random":
            extract_voxel = ExtractVoxelRandomly(n_patch_per_voxel)
        elif how_vx == "balanced":
            extract_voxel = ExtractVoxelBalanced(n_patch_per_voxel)
        else:
            print "error in pick_voxel"
            return

    return PickVoxel(select_region, extract_voxel)


class PickVoxel():
    """
    Manage the selection and extraction of voxels in an mri image
    """
    def __init__(self, select_region, extract_voxel, batch_size=10000):
        self.select_region = select_region
        self.extract_voxel = extract_voxel
        self.batch_size = batch_size

    def pick(self, n_vx, label, verbose=False):
        # Select the region in which voxels are going to be extracted
        idx_region = self.select_region.select(label)
        region = label.ravel()[idx_region]
        # Once the region is selected, extract the voxels
        if n_vx is None:
            n_vx = len(idx_region)
        return self.extract_voxel.extract(n_vx, idx_region, region, label.shape, self.batch_size, verbose)


class SelectRegion():
    """
    Select a specific spatial region of the mri image in which voxels will later be extracted
    """
    def __init__(self):
        pass

    def select(self, label):
        raise NotImplementedError


class SelectWholeBrain(SelectRegion):
    """
    Select the whole labelled brain
    """
    def __init__(self):
        SelectRegion.__init__(self)

    def select(self, label):
        return label.ravel().nonzero()[0]


class SelectPlane(SelectRegion):
    """
    Select a specific orthogonal plane defined by an axis (the plane is orthogonal to this axis) and a specific axis
    coordinate.
    """
    def __init__(self, axis, axis_coordinate):
        SelectRegion.__init__(self)
        self.axis = axis
        self.axis_coordinate = axis_coordinate

    def select(self, label):
        plan = np.zeros(label.shape, dtype=float)
        slice_axis = [slice(None)] * 3
        slice_axis[self.axis] = self.axis_coordinate
        plan[slice_axis] = label[slice_axis]
        return plan.ravel().nonzero()[0]


class ExtractVoxel():
    """
    This class extract voxels from a given region of the mri image
    """
    def __init__(self, n_repeat):
        self.n_repeat = n_repeat

    def extract(self, n_vx, idx_region, region, shape, batch_size, verbose=False):
        n_batches, last_batch_size = divmod(n_vx, batch_size)

        def extract_inner(vx_id, batch_size_inner):
            vx_idx = self.extract_batch_virtual(vx_id, batch_size_inner, idx_region, region)
            if self.n_repeat > 1:
                vx_idx = np.repeat(vx_idx, self.n_repeat)
            return np.asarray(np.unravel_index(vx_idx, shape), dtype=int).T

        for b in xrange(n_batches):
            vx_id = b*batch_size
            if verbose:
                print "        voxels [{} - {}] / {}".format(vx_id, vx_id + batch_size, n_vx)
            yield extract_inner(vx_id, batch_size)

        vx_id = n_batches*batch_size
        if verbose:
            print "        voxels [{} - {}] / {}".format(vx_id, vx_id + last_batch_size, n_vx)
        yield extract_inner(vx_id, last_batch_size)

    def extract_batch_virtual(self, vx_id, batch_size, idx_region, region):
        raise NotImplementedError


class ExtractVoxelRandomly(ExtractVoxel):
    """
    Uniform spatial distribution of the patches
    """
    def __init__(self, n_repeat):
        ExtractVoxel.__init__(self, n_repeat)

    def extract_batch_virtual(self, vx_id, batch_size, idx_region, region):
        r = np.random.randint(idx_region.size, size=batch_size)
        return idx_region[r]


class ExtractVoxelBalanced(ExtractVoxel):
    """
    Same number of voxels per class
    """
    def __init__(self, n_repeat):
        ExtractVoxel.__init__(self, n_repeat)

    def extract_batch_virtual(self, vx_id, batch_size, idx_region, region):
        vx_idx = np.zeros((batch_size,), dtype=int)

        # Compute the number of voxels for each region
        classes_present = np.unique(region)
        n_classes_present = len(classes_present)
        voxels_per_region = distrib_balls_in_bins(batch_size, n_classes_present)

        vx_counter = 0
        for id_k, k in enumerate(classes_present):
            if voxels_per_region[id_k] == 0:
                continue
            sub_region = np.where(region == k)[0]
            r = np.random.randint(len(sub_region), size=voxels_per_region[id_k])
            vx_counter_next = vx_counter + voxels_per_region[id_k]
            vx_idx[vx_counter:vx_counter_next] = idx_region[sub_region[r]]
            vx_counter = vx_counter_next

        return vx_idx


class ExtractVoxelAll(ExtractVoxel):
    """
    Extract all the possible voxels from the mri region
    """
    def __init__(self, n_repeat):
        ExtractVoxel.__init__(self, n_repeat)

    def extract_batch_virtual(self, vx_id, batch_size, idx_region, region):
        return idx_region[vx_id:vx_id+batch_size]