__author__ = 'adeb'

import numpy as np


def create_pick_features(config):
    """
    Factory function to create the objects responsible for picking the patches
    """
    ls_pick_features = []

    for pick_features_dictionary in config.pick_features:
        ls_pick_features.extend(create_pick_features_from_dict(pick_features_dictionary))

    pick_features = PickComposed(ls_pick_features)

    return pick_features


def create_pick_features_from_dict(pick_feature_dictionary):
    """
    Factory function to create the objects responsible for picking the patches
    """
    ls_pick_features = []
    how = pick_feature_dictionary["how"]

    ## Patch-based features
    if how == "3D":
        patch_width = pick_feature_dictionary["patch_width"]
        scale = pick_feature_dictionary["scale"]
        ls_pick_features.append(PickPatch3D(patch_width, scale))
    elif how == "2Dortho":
        ls_axis = pick_feature_dictionary["axis"]
        patch_width = pick_feature_dictionary["patch_width"]
        scale = pick_feature_dictionary["scale"]
        for axis in ls_axis:
            ls_pick_features.append(PickPatch2D(patch_width, axis, scale))
    # elif how_patch == "2DorthoRotated":
    #     axis = config_ini.pick_patch["axis"]
    #     max_degree_rotation = config_ini.pick_patch["max_degree_rotation"]
    #     pick_patch = PickPatchSlightlyRotated(patch_width, axis, max_degree_rotation)
    # elif how == "grid_patches":
    #     patch_width = pick_feature_dictionary["patch_width"]
    #     ls_pick_features.append(PickLocalGridOfPatches(patch_width))

    ## Geometric features
    elif how == "centroid":
        n_features = pick_feature_dictionary["n_features"]
        ls_pick_features.append(PickCentroidDistances(n_features))
    elif how == "xyz":
        ls_pick_features.append(PickXYZ())

    else:
        print "pick_features not specified"
        return

    return ls_pick_features


class PickFeatures():
    """
    Manage the selection and extraction of patches in an mri image from their central voxels
    """
    def __init__(self, n_features, required_pad=0, n_types_of_features=1):
        self.n_features = n_features
        self.required_pad = required_pad
        self.n_types_of_features = n_types_of_features

    def pick(self, vx, mri, label, region_centroids=None):
        """
        Returns a list of tuples of the form (t0, ...), t0 being the extracted features. Additional information can be
        added in each tuple.
        """
        raise NotImplementedError

    def has_instance_of(self, class_object):
        return isinstance(self, class_object)


class PickXYZ(PickFeatures):
    def __init__(self):
        PickFeatures.__init__(self, 3)

    def pick(self, vx, mri, label, region_centroids=None):
        return vx, None


class PickCentroidDistances(PickFeatures):
    def __init__(self, n_features):
        PickFeatures.__init__(self, n_features)

    def pick(self, vx, mri, label, region_centroids=None):
        n_points = vx.shape[0]
        distances = np.zeros((n_points, self.n_features))
        for i in xrange(n_points):
            distances[i] = region_centroids.compute_scaled_distances(vx[i])

        return distances, None


class PickPatch(PickFeatures):
    def __init__(self, n_in, patch_width, scale=1):
        PickFeatures.__init__(self, n_in)
        self.patch_width = patch_width
        self.scale = scale
        self.required_pad = 1 + patch_width * scale / 2

    def pick(self, vx, mri, label, region_centroids=None):
        n_vx = vx.shape[0]
        idx_patch = np.zeros((n_vx, self.n_features), dtype=int)
        patch = np.zeros((n_vx, self.n_features), dtype=np.float32)
        for i in xrange(n_vx):
            patch_temp = self.extract_patch(mri, vx[i], self.scale * self.patch_width)
            patch_temp = self.rebin(patch_temp, (self.patch_width,) * len(patch_temp.shape))
            patch[i, :] = patch_temp.ravel()
        return patch, [idx_patch]

    def extract_patch(self, mri, vx, patch_width):
        raise NotImplementedError

    def rebin(self, patch, new_shape):
        """
        Convert patch into a new patch of shape new_shape by averaging the pixels.
        """
        ls_sh = []
        for sh_old, sh_new in zip(patch.shape, new_shape):
            ls_sh.extend([sh_new, sh_old//sh_new])

        patch_temp = patch.reshape(tuple(ls_sh))
        for i in xrange(len(new_shape)):
            patch_temp = patch_temp.mean(i+1)
        return patch_temp


class PickPatch2D(PickPatch):
    """
    Pick a 2D patch centered on the voxels. The final patch has a width of patch_width but captures a window of
    patch_width * scale width which is averaged.
    """
    def __init__(self, patch_width, orthogonal_axis, scale=1):
        PickPatch.__init__(self, patch_width**2, patch_width, scale)
        self.orthogonal_axis = orthogonal_axis
        self.parallel_axis = range(3)
        del self.parallel_axis[self.orthogonal_axis]

    def extract_patch(self, mri, single_vx, patch_width):
        s = [slice(None)]*3
        s[self.orthogonal_axis] = single_vx[self.orthogonal_axis]
        mri_slice = mri[s]

        vx_slice = single_vx[self.parallel_axis]

        radius = patch_width / 2

        return mri_slice[vx_slice[0] - radius:vx_slice[0] + radius + 1,
               vx_slice[1] - radius:vx_slice[1] + radius + 1]


class PickPatch3D(PickPatch):
    def __init__(self, patch_width, scale=1):
        PickPatch.__init__(self, patch_width**3, patch_width, scale)

    def extract_patch(self, mri, single_vx, patch_width):
        dims = mri.shape
        radius = self.patch_width / 2

        def crop(j, voxel):
            v = np.arange(voxel[j] - radius, voxel[j] + radius + 1)
            v[v < 0] = 0
            v[v >= dims[j]] = dims[j]-1
            return v

        v_axis = []
        for ax in range(3):
            v_axis.append(crop(ax, single_vx))

        x, y, z = np.meshgrid(v_axis[0], v_axis[1], v_axis[2])
        # idx_patch = np.ravel_multi_index((x.ravel(), y.ravel(), z.ravel()), dims)
        patch = mri[x, y, z]

        return patch


class PickComposed(PickFeatures):
    """
    PickFeatures subclass composed of a list of PickFeatures objects. Method pick returns an array of features. This
    array is the concatenation of the features arrays picked by each PickFeatures object.
    Attributes:
        ls_pick_features (list of PickFeatures objects): list containing the PickFeatures
        ls_slices_different_features (list of slices): slices corresponding to each set of homogeneous features
    """
    def __init__(self, ls_pick_patch):
        self.ls_pick_features = ls_pick_patch
        self.ls_slices_different_features = []
        n_features = 0
        required_pad = 0
        c = 0
        for pick_patch in ls_pick_patch:
            self.ls_slices_different_features.append(slice(c, c+pick_patch.n_features))
            c += pick_patch.n_features
            n_features += pick_patch.n_features
            if required_pad < pick_patch.required_pad:
                required_pad = pick_patch.required_pad
        PickFeatures.__init__(self, n_features, required_pad, len(ls_pick_patch))

    def pick(self, vx, mri, label, region_centroids=None):
        n_vx = vx.shape[0]
        patch = np.zeros((n_vx, self.n_features), dtype=np.float32)
        ls_extra_info = []
        for slice_features, pick_patch in zip(self.ls_slices_different_features, self.ls_pick_features):
            res = pick_patch.pick(vx, mri, label, region_centroids)
            patch[:, slice_features] = res[0]
            if res[1] is not None:
                ls_extra_info.append(None)
            else:
                ls_extra_info.append(res[1:])

        return patch, ls_extra_info

    def has_instance_of(self, class_object):
        for pick_patch in self.ls_pick_features:
            if isinstance(pick_patch, class_object):
                return True
        return False

    def __iter__(self):
        return self.ls_pick_features.__iter__()

    def next(self):
        return self.ls_pick_features.next()