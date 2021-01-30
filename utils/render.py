import numpy as np
import math
import torch.nn as nn
import torch
import torch.nn.functional as F

# from pytorch3d.structures import Meshes, Textures
# from pytorch3d.renderer import (
#     OpenGLPerspectiveCameras,TexturedSoftPhongShader,
#     look_at_view_transform, look_at_rotation,SoftPhongShader,
#     RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
#     SoftSilhouetteShader, HardPhongShader, PointLights, DirectionalLights
# )

from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, TexturesVertex,
    SoftPhongShader, RasterizationSettings,
    MeshRenderer, MeshRasterizer,
    BlendParams, DirectionalLights, SoftSilhouetteShader,
)

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

class PerspectiveCamera(nn.Module):

    FOCAL_LENGTH = 5000

    def __init__(self, rotation=None, translation=None,
                 focal_length_x=None, focal_length_y=None,
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(PerspectiveCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_x is None else
                focal_length_x,
                dtype=dtype)

        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_y is None else
                focal_length_y,
                dtype=dtype)

        self.register_buffer('focal_length_x', focal_length_x)
        self.register_buffer('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)

        if rotation is None:
            rotation = torch.eye(
                3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)

        rotation = nn.Parameter(rotation, requires_grad=True)
        self.register_parameter('rotation', rotation)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)

        translation = nn.Parameter(translation,
                                   requires_grad=True)
        self.register_parameter('translation', translation)

    def forward(self, points, transl=None):
        device = points.device

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y
        '''
            rotation: 3 * 3
            translation: 1 * 3
            camera_transform: 1 * 4 * 4
        '''
        if transl is None:
            transl = self.translation
        camera_transform = transform_mat(self.rotation,
                                         transl.unsqueeze(dim=-1))
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)

        # Convert the points to homogeneous coordinates
        '''
            points_h: 1 * 118 * 4
            projected_points: 1 * 118 * 4
            img_points: 1 * 118 * 2
        '''
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + self.center.unsqueeze(dim=1)
        return img_points

class Pytorch3DCamera(nn.Module):
    def __init__(self, image_size=512, focal=1., device='cpu'):
        super(Pytorch3DCamera, self).__init__()
        self.device = device
        R = torch.from_numpy(np.array([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]], dtype=np.float32)).to(
            device)
        T = torch.from_numpy(np.array([[0, 0, 0]], dtype=np.float32)).to(
            device)

        self.cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T,
                                           fov=2 * math.atan(image_size / 2 / (focal)) / np.pi * 180, aspect_ratio=1)
        # cameras = OpenGLRealPerspectiveCameras(device=device, R=R, T=T, w=512, h=512, x0=-256, y0=-256, focal_length=5000 * fscale, znear=0.0001, zfar=1000)
        self.blend_params = BlendParams(sigma=1e-5, gamma=1e-3)

        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0,  # np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=1,
            bin_size=None,  # this setting controls whether naive or coarse-to-fine rasterization is used
            max_faces_per_bin=None  # this setting is for coarse rasterization
        )

        self.raster_settings_sil = RasterizationSettings(
            image_size=image_size,
            blur_radius=0,  # np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=1,
            bin_size=50,  # this setting controls whether naive or coarse-to-fine rasterization is used
            max_faces_per_bin=None  # this setting is for coarse rasterization
        )

        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings_sil
            ),
            shader=SoftSilhouetteShader(blend_params=self.blend_params)
        )

        # self.lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        self.lights = DirectionalLights(device=device, direction=[[0, 0, -1]])
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=self.cameras,
                lights=self.lights
            )
        )

    def forward(self, verts, faces):
        mesh = Meshes(
            verts=verts,
            faces=faces
        )

        return self.silhouette_renderer(meshes_world=mesh)

    def render(self, vert, faces):
        vert_shape = vert.shape
        sphere_verts_rgb = torch.full([1, vert_shape[0], 3], 1., device=faces.device, requires_grad=False)
        textures = TexturesVertex(verts_features=sphere_verts_rgb)

        mesh = Meshes(
            verts=[vert],
            faces=[faces],
            textures=textures
        )
        # mesh.extend(len(vert[0]))

        return self.renderer(meshes_world=mesh)
