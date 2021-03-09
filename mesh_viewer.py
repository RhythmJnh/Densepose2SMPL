import numpy as np
import trimesh
import pyrender
import sys
import cv2

colors = {
    'pink': [.7, .7, .9],
    'purple': [.9, .7, .7],
    'cyan': [.7, .75, .5],
    'red': [1.0, 0.0, 0.0],

    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [.5, .7, .7],
    'blue': [.0, .0, 1.],

    'offwhite': [.8, .9, .9],
    'white': [1., 1., 1.],
    'orange': [.5, .65, .9],

    'grey': [.7, .7, .7],
    'black': np.zeros(3),
    'white': np.ones(3),

    'yellowg': [0.83, 1, 0],
}

__all__ = ['MeshViewer']

class MeshViewer(object):

    def __init__(self, width=1200, height=800, use_offscreen=False):
        super().__init__()

        self.use_offscreen = use_offscreen
        self.render_wireframe = False

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh

        self.scene = pyrender.Scene(bg_color=colors['white'], ambient_light=(0.3, 0.3, 0.3))

        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 5])
        self.camera_node = self.scene.add(pc, pose=camera_pose, name='pc-camera')

        self.figsize = (width, height)
        self.viewer = None

    def viewer_update(self):
        self.close_viewer()
        if self.use_offscreen:
            self.viewer = pyrender.OffscreenRenderer(*self.figsize)
            self.use_raymond_lighting(5.)
        else:
            self.viewer = pyrender.Viewer(self.scene,
                                          use_raymond_lighting=True,
                                          viewport_size=self.figsize,
                                          cull_faces=False,
                                          run_in_thread=False,
                                          viewer_flags={'rotate_axis':[0, 1, 0]})

    def set_background_color(self, color=colors['white']):
        self.scene.bg_color = color

    def update_camera_pose(self, camera_pose):
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def close_viewer(self):
        if self.viewer is None:
            return None
        if self.viewer.is_active:
            self.viewer.close_external()

    def set_meshes(self, meshes, group_name='static', poses=[]):
        for node in self.scene.get_nodes():
            if node.name is not None and '%s-mesh'%group_name in node.name:
                self.scene.remove_node(node)

        if len(poses) < 1:
            for mid, mesh in enumerate(meshes):
                if isinstance(mesh, trimesh.Trimesh):
                    mesh = pyrender.Mesh.from_trimesh(mesh)
                self.scene.add(mesh, '%s-mesh-%2d'%(group_name, mid))
        else:
            for mid, iter_value in enumerate(zip(meshes, poses)):
                mesh, pose = iter_value
                if isinstance(mesh, trimesh.Trimesh):
                    mesh = pyrender.Mesh.from_trimesh(mesh)
                self.scene.add(mesh, '%s-mesh-%2d'%(group_name, mid), pose)

    def set_static_meshes(self, meshes, poses=[]): self.set_meshes(meshes, group_name='static', poses=poses)
    def set_dynamic_meshes(self, meshes, poses=[]): self.set_meshes(meshes, group_name='dynamic', poses=poses)

    def _add_raymond_light(self):
        from pyrender.light import DirectionalLight
        from pyrender.node import Node

        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3,:3] = np.c_[x,y,z]
            nodes.append(Node(
                light=DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix
            ))
        return nodes

    def use_raymond_lighting(self, intensity = 1.0):
        if not self.use_offscreen:
            sys.stderr.write('Interactive viewer already uses raymond lighting!\n')
            return
        for n in self._add_raymond_light():
            n.light.intensity = intensity / 3.0
            if not self.scene.has_node(n):
                self.scene.add_node(n)#, parent_node=pc)

    def render(self, render_wireframe=None, RGBA=False):
        from pyrender.constants import RenderFlags

        flags = RenderFlags.SHADOWS_DIRECTIONAL
        if RGBA: flags |=  RenderFlags.RGBA
        if render_wireframe is not None and render_wireframe==True:
            flags |= RenderFlags.ALL_WIREFRAME
        elif self.render_wireframe:
            flags |= RenderFlags.ALL_WIREFRAME
        color_img, depth_img = self.viewer.render(self.scene, flags=flags)

        return color_img

    def save_snapshot(self, fname):
        if not self.use_offscreen:
            sys.stderr.write('Currently saving snapshots only works with off-screen renderer!\n')
            return
        color_img = self.render()
        cv2.imwrite(fname, color_img)


    def add_point_spheres(self, pts, colors=[0.9, 0.1, 0.1, 1.0], radius=0.03):
        sm = trimesh.creation.uv_sphere(radius=radius)
        sm.visual.vertex_colors = colors
        tfs = np.tile(np.eye(4), (len(pts), 1, 1))
        tfs[:, :3, 3] = pts
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        self.scene.add(joints_pcl)

    def add_point_sprites(self, pts):
        colors = np.random.uniform(size=pts.shape)
        ps = pyrender.Mesh.from_points(pts, colors=colors)
        self.scene.add(ps)

    def remove_object(self):
        for node in self.scene.get_nodes():
            if node.name is None:
                self.scene.remove_node(node)



