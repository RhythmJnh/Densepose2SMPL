from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import numpy as np

import torch
import torch.nn as nn


def to_tensor(tensor, dtype=torch.float32):
    if torch.Tensor == type(tensor):
        return tensor.clone().detach()
    else:
        return torch.tensor(tensor, dtype)


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def max_grad_change(grad_arr):
    return grad_arr.abs().max()

class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist


def render(vertices,
           faces,
           cam, 
           focal=5000., 
           depth_or_color='color',
           H=None,
           W=None,
           vis_color=None,
           ):

    import trimesh
    out_mesh = trimesh.Trimesh(vertices, faces)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    out_mesh.apply_transform(rot)

    if vis_color is not None:
        out_mesh.visual.vertex_colors = vis_color

    import pyrender
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0,
                                                  alphaMode='OPAQUE',
                                                  baseColorFactor=(1.0, 1.0, 0.9, 1.0))

    mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    camera_center = cam.center.detach().cpu().numpy().squeeze()
    camera_transl = cam.translation.detach().cpu().numpy().squeeze()
    # Equivalent to 180 degrees around the y-axis. Transforms the fit to
    # OpenGL compatible coordinate system.
    camera_transl[0] *= -1.0
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_transl

    camera = pyrender.camera.IntrinsicsCamera(
            fx=focal, fy=focal, 
            cx=camera_center[0], cy=camera_center[1])
    scene.add(camera, pose=camera_pose)

    # create the lights
    from pyrender import DirectionalLight, Node
    light = DirectionalLight(color=np.ones(3), intensity=2.5)
    node = Node(light=light, matrix=np.eye(4))
    scene.add_node(node)

    r = pyrender.OffscreenRenderer(viewport_width=W, 
                                   viewport_height=H, 
                                   point_size=1.0)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)

    r.delete()

    if depth_or_color == 'depth':
        return depth
    else:
        return color