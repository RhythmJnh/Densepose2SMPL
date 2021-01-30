import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pickle
import os.path as osp
import sys
import cv2
from utils import transformations
from utils import geometry

cur_dir = osp.dirname(__file__)

pose_sample_index = torch.tensor([0, 2, 18, 35, 4, 20, 36, 5, 21, 37, 7, 23, 88, 38, 63, 90, 39, 64, 41, 66, 43, 68, 42, 67, 40, 65, 3, 19], dtype=torch.int64)
pose_arm_sample_index = torch.tensor([40, 65, 42, 67], dtype=torch.int64)
pose_leg_sample_index = torch.tensor([19, 3], dtype=torch.int64)
COCO25_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
focal_length = 5000 * 28. / 234.7

select_366vidx = torch.tensor([1,7,26,31,32,33,35,46,53,56,58,60,69,85,86,91,94,96,99,100,176,180,216,227,231,464,465,512,513,517,526,549,583,614,776,850,854,855,858,895,915,919,939,999,1323,1341,1358,1370,1392,1417,1498,1505,1517,1533,1535,1544,1546,1633,1694,1720,1746,1816,1826,1856,1878,1941,1948,1957,1963,2027,2161,2167,2177,2237,2415,2483,2488,2495,2549,2604,2661,2689,2704,2710,2721,2766,2774,2777,2778,2847,2850,2878,2881,2897,2920,2944,2959,2961,2986,2989,3016,3020,3043,3046,3063,3073,3082,3090,3114,3122,3131,3142,3275,3369,3409,3414,3422,3440,3446,3454,3463,3483,3491,3501,3532,3540,3589,3646,3648,3656,3686,3704,3734,3753,3762,3791,3813,3872,3875,3888,3991,4024,4051,4103,4109,4113,4128,4161,4185,4230,4234,4424,4450,4465,4477,4484,4542,4564,4625,4627,4652,4665,4717,4732,4747,4759,4762,4775,4839,5026,5032,5046,5053,5062,5232,5302,5321,5345,5382,5414,5494,5570,5605,5615,5658,5662,6315,6344,6345,6346,6361,6386,6465,6491,6493,6529,6538,6569,6686,6692,6718,6726,6737,6741,6821,6974,6975,7022,7023,7027,7036,7059,7093,7124,7286,7360,7364,7365,7368,7405,7429,7449,7509,7833,7851,7868,7880,7927,8008,8015,8027,8043,8045,8054,8056,8143,8204,8234,8256,8326,8336,8366,8388,8451,8458,8467,8473,8537,8671,8677,8687,8747,8925,8993,9001,9005,9059,9114,9171,9199,9214,9220,9231,9276,9284,9287,9288,9357,9360,9391,9407,9430,9454,9469,9471,9496,9499,9526,9530,9553,9556,9573,9583,9592,9600,9623,9631,9649,9774,9864,9904,9909,9917,9935,9941,9949,9958,9978,9986,9996,10025,10033,10080,10137,10139,10147,10176,10221,10240,10249,10267,10278,10300,10359,10362,10375,10475,10508,10535,10579,10593,10597,10612,10645,10669,10713,10717,10905,10930,10945,10956,10963,11040,11099,11123,11135,11187,11202,11217,11220,11229,11245,11308,11495,11501,11511,11522,11531,11767,11842,12020,12055,12065,12106,12110,12703,12732,12733,12734,12749,12774,12848,12875,12909,12918,12949], dtype=torch.int64)

def flip_kp3d(kp):
    # """Flip keypoints."""
    if len(kp) == 25:
        flipped_parts = COCO25_FLIP_PERM
    else:
        assert False
    kp = kp[flipped_parts]
    kp[:, 0] = -kp[:, 0]
    return kp

def j3d_processing(coco_kpt25_3d, r, center, f):
    # in-plane rotation
    rot_mat = np.eye(3)
    if not r == 0:
        rot_rad = -r * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
    # coco_kpt25_3d = np.einsum('ij,kj->ki', rot_mat, coco_kpt25_3d)
    coco_kpt25_3d = coco_kpt25_3d - center
    coco_kpt25_3d = (rot_mat @ coco_kpt25_3d.transpose()).transpose()
    coco_kpt25_3d = coco_kpt25_3d + center
    if f:
        coco_kpt25_3d = flip_kp3d(coco_kpt25_3d)
    coco_kpt25_3d = coco_kpt25_3d.astype(np.float32)
    return coco_kpt25_3d

def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(R @ per_rdg)
    aa = (resrot.T)[0]
    return aa

DAZ_JOOINTS_ORIG_PERM = [0, 2, 18, 35, 4, 20, 36, 5, 21, 37, 7, 23, 88, 38, 63, 90, 39, 64, 41, 66, 43, 68]
DAZ_POSE_ORIG_PERM = []
for i in DAZ_JOOINTS_ORIG_PERM:
    DAZ_POSE_ORIG_PERM.append(3 * i)
    DAZ_POSE_ORIG_PERM.append(3 * i + 1)
    DAZ_POSE_ORIG_PERM.append(3 * i + 2)
DAZ_JOOINTS_FLIP_PERM = [0, 18, 2, 35, 20, 4, 36, 21, 5, 37, 23, 7, 88, 63, 38, 90, 64, 39, 66, 41, 68, 43]
DAZ_POSE_FLIP_PERM = []
for i in DAZ_JOOINTS_FLIP_PERM:
    DAZ_POSE_FLIP_PERM.append(3 * i)
    DAZ_POSE_FLIP_PERM.append(3 * i + 1)
    DAZ_POSE_FLIP_PERM.append(3 * i + 2)

def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """

    orig_parts = DAZ_POSE_ORIG_PERM
    flipped_parts = DAZ_POSE_FLIP_PERM

    # we also negate the second and the third dimension of the axis-angle
    pose[orig_parts] = pose[flipped_parts]
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose

def pose_processing(pose, r, f):
    """Process SMPL theta parameters  and apply all augmentation transforms."""
    # rotation or the pose parameters
    pose[:3] = rot_aa(pose[:3], r)
    # flip the pose parameters
    if f:
        pose = flip_pose(pose)
    pose = pose.astype('float32')
    return pose

def pose170to28(pose170):
    bs = pose170.size(0)
    tmp_pose = pose170.reshape(bs, 170, -1)
    pose28 = torch.zeros((bs, pose_sample_index.size(0), tmp_pose.size(2)), dtype=torch.float32, device=pose170.device)
    for ix, pindex in enumerate(pose_sample_index):
        pose28[:, ix] = tmp_pose[:, pindex]
    pose28 = pose28.reshape(bs, -1)
    return pose28.contiguous()

def pose28to170(pose28):
    bs = pose28.size(0)
    tmp_pose = pose28.reshape(bs, pose_sample_index.size(0), -1)
    pose170 = torch.zeros((bs, 170, tmp_pose.size(2)), dtype=torch.float32, device=pose28.device)
    for ix, pindex in enumerate(pose_sample_index):
        pose170[:, pindex] = tmp_pose[:, ix]
    pose170 = pose170.reshape(bs, -1)
    return pose170.contiguous()

def poserotmat_to_pose(pose6d):
    bs = pose6d.size(0)
    pred_rotmat_hom = torch.cat(
        [
            pose6d.view(-1, 3, 3),
            torch.tensor([0, 0, 1], dtype=torch.float32, device=pose6d.device).view(1, 3, 1).expand(
                bs * pose_sample_index.size(0), -1, -1)
        ],
        dim=-1)
    pose = geometry.quaternion_to_angle_axis(
        geometry.rotation_matrix_to_quaternion(pred_rotmat_hom)
    ).reshape(bs, -1)

    return pose.contiguous()


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(torch.matmul(transforms.clone(), joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    # rel_transforms = transforms.clone()

    return posed_joints, rel_transforms

def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

class DazModel(nn.Module):
    def __init__(self, model_dir=osp.join(cur_dir, '../weights'), batch_size=1, gender='female'):
        super(DazModel, self).__init__()

        if gender == 'female':
            self.model_path = osp.join(model_dir, "basic_v18.pkl")
        else:
            self.model_path = osp.join(model_dir, "daz_male_v3.pkl")

        if os.path.exists(self.model_path):
            pkl_info = pickle.load(open(self.model_path, 'rb'))

            self.v_template_np = pkl_info['v_template'].astype(np.float32)
            v_template = torch.from_numpy(self.v_template_np)
            self.register_buffer('v_template', v_template)

            shapedirs = torch.from_numpy(pkl_info['shape_dirs'].astype(np.float32).transpose((1, 2, 0)))
            self.register_buffer('shapedirs', shapedirs)

            base_joint = torch.from_numpy(pkl_info['joints'].astype(np.float32))
            self.register_buffer('base_joint', base_joint)

            jointdirs = torch.from_numpy(pkl_info['joints_dirs'].astype(np.float32).transpose((1, 2, 0)))
            self.register_buffer('jointdirs', jointdirs)

            self.faces = pkl_info['f'].astype(np.int64)

            parent = torch.from_numpy(np.asarray(pkl_info['kintree_table'][0], dtype=np.int64))
            self.register_buffer('parent', parent)

            lbs_weights = torch.from_numpy(pkl_info['weights'].astype(np.float32).transpose((1, 0)))
            self.register_buffer('lbs_weights', lbs_weights)

            if gender == 'female':
                J_regressor_coco25 = pickle.load(open(osp.join(model_dir, 'J_regressor_coco25_for_daz.pkl'), 'rb'))
            else:
                J_regressor_coco25 = pickle.load(open(osp.join(model_dir,
                                                                   'J_regressor_coco25_for_daz_male.pkl'), 'rb'))

            self.register_buffer('J_regressor_coco25', torch.tensor(J_regressor_coco25, dtype=torch.float32))

        else:
            print(self.__class__.__name__, "Fatal Error!")
            print("Daz model donot exist", self.model_path)
            exit(-1)

        betas = torch.zeros((batch_size, self.shapedirs.shape[2]), dtype=torch.float32)
        self.register_buffer('betas', betas)

        global_orient = torch.zeros((batch_size, 1 * 3), dtype=torch.float32)
        self.register_buffer('global_orient', global_orient)
        body_pose = torch.zeros((batch_size, 169 * 3), dtype=torch.float32)
        self.register_buffer('body_pose', body_pose)

        transl = torch.zeros((batch_size, 3), dtype=torch.float32)
        self.register_buffer('transl', transl)

        self.batch_size = batch_size

    def forward(self, betas=None, global_orient=None, body_pose=None, transl=None, pose2rot=True, template=None, dance=False):
        global_orient = global_orient if global_orient is not None else self.global_orient
        body_pose = body_pose if body_pose is not None else self.body_pose
        transl = transl if transl is not None else self.transl

        global_orient = global_orient.view(self.batch_size, -1, 3)
        body_pose = body_pose.view(self.batch_size, -1, 3)


        new_betas = self.betas.clone()
        if not betas is None:
            betas = betas.reshape(new_betas.size(0), -1)
            if betas.size(1) > new_betas.size(1):
                print("####### fatal error: bad betas:", betas.shape)
                exit(-1)
            else:
                new_betas[:,:betas.size(1)] = betas[:,:]

        if new_betas.size(1) >= 27 and dance:
            new_betas[:, 26] = 1
            new_betas[:, 27] = 1
            new_betas[:, 28] = 1

            l_euler = -transformations.euler_from_matrix(cv2.Rodrigues(body_pose[0][39-1].cpu().numpy())[0], 'sxyz')[2] * 180. / np.pi
            new_betas[:, 24] = 0 if l_euler < 10 else 1 if l_euler > 45 else (l_euler - 10) / 35  # left shoulder

            r_euler = transformations.euler_from_matrix(cv2.Rodrigues(body_pose[0][64-1].cpu().numpy())[0], 'sxyz')[2] * 180. / np.pi
            new_betas[:, 25] = 0 if r_euler < 10 else 1 if r_euler > 45 else (r_euler - 10) / 35  # right shoulder

        if new_betas.size(1) >= 32:
            new_betas[:, 31] = 1
        new_betas[:, 7] = - new_betas[:, 3]

        if template is None:
            v_template = self.v_template
        else:
            v_template_np = self.v_template_np.copy()
            v_template_np[:, :] = template[:, :]
            v_template = torch.from_numpy(v_template_np).to(self.v_template.device)

        v_shaped = v_template + blend_shapes(new_betas, self.shapedirs)
        v_joint = self.base_joint + blend_shapes(new_betas, self.jointdirs)

        full_pose = torch.cat([global_orient, body_pose], dim=1)
        device = new_betas.device
        if pose2rot:
            rot_mats = batch_rodrigues(full_pose.view(-1, 3), dtype=torch.float32).view([self.batch_size, -1, 3, 3])
        else:
            rot_mats = full_pose.view(self.batch_size, -1, 3, 3)

        v_posed = v_shaped
        J_transformed, A = batch_rigid_transform(rot_mats, v_joint, self.parent, dtype=torch.float32)

        W = self.lbs_weights.unsqueeze(dim=0).expand([self.batch_size, -1, -1])
        num_joints = self.lbs_weights.shape[1]
        T = torch.matmul(W, A.view(self.batch_size, num_joints, 16)).view(self.batch_size, -1, 4, 4)

        homogen_coord = torch.ones([self.batch_size, v_posed.shape[1], 1],
                                   dtype=torch.float32, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]

        J_transformed /= 110
        verts /= 110

        J_transformed += transl.unsqueeze(dim=1)
        verts += transl.unsqueeze(dim=1)

        coco25_joints = vertices2joints(self.J_regressor_coco25, verts)

        return verts, J_transformed, coco25_joints

class DazClothModel(DazModel):
    def __init__(self, cloth_info=None, model_dir=None, batch_size=1, gender='female'):

        super(DazClothModel, self).__init__(model_dir, gender=gender)

        if cloth_info is not None:
            self.v_template = torch.from_numpy(cloth_info['v_template'].astype(np.float32))
            if cloth_info['shape_dirs'].shape[0] == 24:
                shape_dirs_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
                tmp_shape_dirs = np.concatenate([cloth_info['shape_dirs'][idx][np.newaxis] for idx in shape_dirs_idx], axis=0)
                self.shapedirs = torch.from_numpy(tmp_shape_dirs.astype(np.float32).transpose((1, 2, 0)))
            else:
                # hack for earlier version of cloth
                diff = 24 - cloth_info['shape_dirs'].shape[0]
                tmp_shape_dirs = np.array(cloth_info['shape_dirs'])
                padding = np.zeros((diff, tmp_shape_dirs.shape[1], tmp_shape_dirs.shape[2]), dtype=np.float)
                tmp_shape_dirs = np.concatenate([tmp_shape_dirs, padding], axis=0)
                self.shapedirs = torch.from_numpy(tmp_shape_dirs.astype(np.float32).transpose((1, 2, 0)))

            self.faces = cloth_info['f'].astype(np.int64)
            self.lbs_weights = torch.from_numpy(cloth_info['weights'].astype(np.float32).transpose((1, 0)))
            self.uv = cloth_info['uv']
        else:
            print('Daz cloth model path not exists')
            sys.exit(0)

    def forward_zrn(self, betas=None, pose=None, pose2rot=True, template=None):
        betas = betas if betas is not None else self.betas
        pose = pose if pose is not None else self.pose

        new_betas = self.betas.clone()
        if not betas is None:
            betas = betas.reshape(new_betas.size(0), -1)
            if betas.size(1) > new_betas.size(1):
                print("####### fatal error: bad betas:", betas.shape)
                exit(-1)
            else:
                new_betas[:,:betas.size(1)] = betas[:,:]

        if template is None:
            v_template = self.v_template
        else:
            v_template_np = self.v_template_np.copy()
            v_template_np[self.head_idxes, :] = template[self.head_idxes, :]
            v_template = torch.from_numpy(v_template_np).to(self.v_template.device)

        v_shaped = v_template + blend_shapes(betas, self.shapedirs)
        v_joint = self.base_joint + blend_shapes(betas, self.jointdirs)

        device = betas.device
        if pose2rot:
            rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=torch.float32).view([self.batch_size, -1, 3, 3])
        else:
            rot_mats = pose.view(self.batch_size, -1, 3, 3)

        v_posed = v_shaped
        J_transformed, A = batch_rigid_transform(rot_mats, v_joint, self.parent, dtype=torch.float32)

        W = self.lbs_weights.unsqueeze(dim=0).expand([self.batch_size, -1, -1])
        num_joints = self.lbs_weights.shape[1]
        T = torch.matmul(W, A.view(self.batch_size, num_joints, 16)).view(self.batch_size, -1, 4, 4)

        homogen_coord = torch.ones([self.batch_size, v_posed.shape[1], 1],
                                   dtype=torch.float32, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0] / 110.0

        return verts, J_transformed




