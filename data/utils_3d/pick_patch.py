__author__ = 'adeb'

import numpy as np

from scipy.ndimage.interpolation import rotate


def create_pick_features(config_ini):
    """
    Factory function to create the objects responsible for picking the patches
    """
    how_patch = config_ini.pick_patch["how"]
    patch_width = config_ini.pick_patch["patch_width"]
    if how_patch == "3D":
        pick_patch = PickPatch3DSimple(patch_width)
    elif how_patch == "2Dortho":
        axis = config_ini.pick_patch["axis"]
        pick_patch = PickPatch2Dorthogonal(patch_width, axis)
    elif how_patch == "2DorthoRotated":
        axis = config_ini.pick_patch["axis"]
        max_degree_rotation = config_ini.pick_patch["max_degree_rotation"]
        pick_patch = PickPatchSlightlyRotated(patch_width, axis, max_degree_rotation)
    elif how_patch == "ultimate":
        pick_patch = PickUltimate(patch_width)
    else:
        print "error in pick_patch"
        return

    return pick_patch


class PickFeatures():
    """
    Manage the selection and extraction of patches in an mri image from their central voxels
    """
    def __init__(self, n_features):
        self.n_features = n_features

    def pick(self, vx, mri, label):
        raise NotImplementedError


class PickXYZ(PickFeatures):
    def __init__(self):
        PickFeatures.__init__(self, 3)

    def pick(self, vx, mri, label):
        idx_patch = 0
        patch = vx
        return patch, idx_patch


class PickPatch2D(PickFeatures):
    def __init__(self, patch_width):
        PickFeatures.__init__(self, patch_width**2)
        self.patch_width = patch_width

    def pick(self, vx, mri, label):
        n_vx = vx.shape[0]
        idx_patch = np.zeros((n_vx, self.n_features), dtype=int)
        patch = np.zeros((n_vx, self.n_features), dtype=np.float32)
        self.pick_virtual2d(patch, idx_patch, vx, mri, label)
        return patch, idx_patch

    def pick_virtual2d(self, patch, idx_patch, vx, mri, label):
        raise NotImplementedError

    def find_limits_along_axis(self, axis, vx, patch_width, dims_mri):
        radius = patch_width / 2
        s1_inf = vx[:, axis] - radius
        s1_sup = vx[:, axis] + radius + 1
        s2_inf = np.zeros(s1_inf.shape, dtype=s1_inf.dtype)
        s2_sup = patch_width * np.ones(s1_inf.shape, dtype=s1_inf.dtype)

        # Too low
        s1_too_low = s1_inf < 0
        s2_inf[s1_too_low] = -s1_inf[s1_too_low]
        s1_inf[s1_too_low] = 0

        # Too high
        s1_too_high = s1_sup > dims_mri[axis]
        s2_sup[s1_too_high] = patch_width - (s1_sup[s1_too_high] - dims_mri[axis])
        s1_sup[s1_too_high] = dims_mri[axis]

        assert (s1_sup - s1_inf == s2_sup - s2_inf).all()

        return s1_inf, s1_sup, s2_inf, s2_sup


class PickPatch2Dorthogonal(PickPatch2D):
    """
    Pick a 2D patch centered on the voxels. No rotation
    """
    def __init__(self, patch_width, orthogonal_axis):
        PickPatch2D.__init__(self, patch_width)
        self.orhtogonal_axis = orthogonal_axis

    def pick_virtual2d(self, patch, idx_patch, vx, mri, label):
        dims = mri.shape

        parallel_axis = range(3)
        del parallel_axis[self.orhtogonal_axis]
        s1_inf_0, s1_sup_0, s2_inf_0, s2_sup_0 = self.find_limits_along_axis(parallel_axis[0], vx,
                                                                             self.patch_width, dims)
        s1_inf_1, s1_sup_1, s2_inf_1, s2_sup_1 = self.find_limits_along_axis(parallel_axis[1], vx,
                                                                             self.patch_width, dims)

        for i in xrange(idx_patch.shape[0]):

            s1 = [0]*3
            s1[self.orhtogonal_axis] = vx[i, self.orhtogonal_axis]
            s1[parallel_axis[0]] = slice(s1_inf_0[i], s1_sup_0[i])
            s1[parallel_axis[1]] = slice(s1_inf_1[i], s1_sup_1[i])

            s2 = [0]*2
            s2[0] = slice(s2_inf_0[i], s2_sup_0[i])
            s2[1] = slice(s2_inf_1[i], s2_sup_1[i])

            patch_temp = np.zeros((self.patch_width,)*2)
            patch_temp[s2[0], s2[1]] = mri[s1[0], s1[1], s1[2]]
            patch[i, :] = patch_temp.ravel()


class PickPatch2DorthogonalAveraged(PickPatch2D):
    """
    Pick a 2D patch centered on the voxels. No rotation
    """
    def __init__(self, patch_width, orthogonal_axis, scale):
        PickPatch2D.__init__(self, patch_width)
        self.orhtogonal_axis = orthogonal_axis
        self.scale = scale

    def pick_virtual2d(self, patch, idx_patch, vx, mri, label):
        dims = mri.shape
        patch_width_global = self.patch_width * self.scale

        parallel_axis = range(3)
        del parallel_axis[self.orhtogonal_axis]
        s1_inf_0, s1_sup_0, s2_inf_0, s2_sup_0 = self.find_limits_along_axis(parallel_axis[0], vx,
                                                                             patch_width_global, dims)
        s1_inf_1, s1_sup_1, s2_inf_1, s2_sup_1 = self.find_limits_along_axis(parallel_axis[1], vx,
                                                                             patch_width_global, dims)

        for i in xrange(idx_patch.shape[0]):

            s1 = [0]*3
            s1[self.orhtogonal_axis] = vx[i, self.orhtogonal_axis]
            s1[parallel_axis[0]] = slice(s1_inf_0[i], s1_sup_0[i])
            s1[parallel_axis[1]] = slice(s1_inf_1[i], s1_sup_1[i])

            s2 = [0]*2
            s2[0] = slice(s2_inf_0[i], s2_sup_0[i])
            s2[1] = slice(s2_inf_1[i], s2_sup_1[i])

            patch_temp = np.zeros((self.patch_width* self.scale,)*2)
            patch_temp[s2[0], s2[1]] = mri[s1[0], s1[1], s1[2]]
            patch_temp = self.rebin(patch_temp, (self.patch_width, self.patch_width))
            patch[i, :] = patch_temp.ravel()

    def rebin(self, patch, new_shape):
        sh = new_shape[0], patch.shape[0]//new_shape[0], new_shape[1], patch.shape[1]//new_shape[1]
        return patch.reshape(sh).mean(-1).mean(1)


class PickPatchSlightlyRotated(PickPatch2D):
    """
    Pick a 2D patch centered on the voxels. Brains are slightly rotated.
    """
    def __init__(self, patch_width, orhtogonal_axis, max_degree_rotation):
        PickPatch2D.__init__(self, patch_width)
        self.orthogonal_axis = orhtogonal_axis
        self.max_degree_rotation = max_degree_rotation

    def pick_virtual2d(self, patch, idx_patch, vx, mri, label):
        dims = mri.shape

        parallel_axis = range(3)
        del parallel_axis[self.orthogonal_axis]
        s1_inf_0, s1_sup_0, s2_inf_0, s2_sup_0 = self.find_limits_along_axis(parallel_axis[0], vx,
                                                                             self.patch_width, dims)
        s1_inf_1, s1_sup_1, s2_inf_1, s2_sup_1 = self.find_limits_along_axis(parallel_axis[1], vx,
                                                                             self.patch_width, dims)
        s1_inf_2, s1_sup_2, s2_inf_2, s2_sup_2 = self.find_limits_along_axis(parallel_axis[2], vx,
                                                                             self.patch_width, dims)

        for i in xrange(idx_patch.shape[0]):

            cube = np.zeros((self.patch_width,)*3)

            s1 = [0]*3
            s1[0] = slice(s1_inf_0[i], s1_sup_0[i])
            s1[1] = slice(s1_inf_1[i], s1_sup_1[i])
            s1[2] = slice(s1_inf_2[i], s1_sup_2[i])

            s2 = [0]*3
            s2[0] = slice(s2_inf_0[i], s2_sup_0[i])
            s2[1] = slice(s2_inf_1[i], s2_sup_1[i])
            s2[2] = slice(s2_inf_2[i], s2_sup_2[i])

            cube[s2[0], s2[1]] = cube[s1[0], s1[1], s1[2]]
            cube = rotate(cube, np.random.uniform(-self.max_degree_rotation, -self.max_degree_rotation), axes=(0, 1))
            cube = rotate(cube, np.random.uniform(-self.max_degree_rotation, -self.max_degree_rotation), axes=(1, 2))
            cube = rotate(cube, np.random.uniform(-self.max_degree_rotation, -self.max_degree_rotation), axes=(2, 0))

            radius = self.patch_width / 2
            central_vx_cube = np.array(cube.shape)/2
            li = central_vx_cube - radius
            ls = central_vx_cube + radius + 1
            li[self.orthogonal_axis] = central_vx_cube[self.orthogonal_axis]
            ls[self.orthogonal_axis] = central_vx_cube[self.orthogonal_axis]+1

            patch[i] = cube[li[0]:ls[0], li[1]:ls[1], li[2]:ls[2]].ravel()


class PickPatch3D(PickFeatures):
    def __init__(self, patch_width):
        PickFeatures.__init__(self, patch_width**3)
        self.patch_width = patch_width

    def pick(self, vx, mri, label):
        n_vx = vx.shape[0]
        idx_patch = np.zeros((n_vx, self.n_features), dtype=int)
        patch = np.zeros((n_vx, self.n_features), dtype=np.float32)
        self.pick_virtual3d(patch, idx_patch, vx, mri, label)
        return patch, idx_patch

    def pick_virtual3d(self, patch, idx_patch, vx, mri, label):
        raise NotImplementedError


class PickPatch3DSimple(PickPatch3D):
    """
    Pick 3D patches centered on the voxels. No rotation
    """
    def __init__(self, patch_width):
        PickPatch3D.__init__(self, patch_width)

    def pick_virtual3d(self, patch, idx_patch, vx, mri, label):
        dims = mri.shape
        radius = self.patch_width / 2

        def crop(j, voxel):
            v = np.arange(voxel[j] - radius, voxel[j] + radius + 1)
            v[v < 0] = 0
            v[v >= dims[j]] = dims[j]-1
            return v

        for i in xrange(idx_patch.shape[0]):
            vx_cur = vx[i]
            v_axis = []
            for ax in range(3):
                v_axis.append(crop(ax, vx_cur))

            x, y, z = np.meshgrid(v_axis[0], v_axis[1], v_axis[2])
            idx_patch[i] = np.ravel_multi_index((x.ravel(), y.ravel(), z.ravel()), dims)
            patch[i] = mri[x, y, z].ravel()