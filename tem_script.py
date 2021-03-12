import smplx
import numpy as np
from utils.daz import DazModel
from mesh_viewer import MeshViewer
import trimesh
import pickle
from my_use.camera import create_camera
import cv2
import torch
from my_use.utils import render
focal = 5000 * 28. / 234.7000
gender = 'female'
model_folder = '/home/jnh/workspace/workspace_bak/smplx/download/models'
dtype = torch.float32


if __name__ == '__main__':

    img = cv2.imread('/home/jnh/下载/package/00078.jpg')
    data = pickle.load(open('/home/jnh/下载/package/00078.pkl', 'rb'))
    go_pose = torch.from_numpy(data["global_orient"])
    transl = torch.from_numpy(data["transl"])
    pose = torch.from_numpy(data["pose_para"])
    beta = torch.from_numpy(data['betas'])

    model_params = dict(model_path='/home/jnh/workspace/workspace_bak/smplx/download/models')
    smpl = smplx.create(model_type='smpl', gender='female', **model_params)
    smpl_face = smpl.faces
    smpl_out = smpl(betas=beta, body_pose=pose, global_orient=go_pose, transl=transl)
    vertices = smpl_out.vertices.detach().cpu().numpy()[0]

    # load camera
    camera = create_camera(focal_length_x=focal,
                           focal_length_y=focal,
                           center=torch.from_numpy(np.array([256, 256])),
                           dtype=dtype)
    color = render(vertices, smpl_face, camera,
                   focal=focal, depth_or_color='color', H=512, W=512)
    color = color.astype(np.float32) / 255.0

    origin_img = img.astype(np.float32) / 255
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis].astype(np.float32) * 1.0
    show_img1 = origin_img * (1 - valid_mask) + color[:, :, :3] * valid_mask
    cv2.imshow('test', show_img1)
    cv2.waitKey()
    a = 1
    # mv = MeshViewer()
    # index_smpl_346 = np.loadtxt('./my_use/sample_index_346.txt').astype(np.int32)
    # index_daz_346 = np.loadtxt('./my_use/sample_index_daz_346.txt').astype(np.int32)
    # daz = DazModel('./model')
    # daz_face = daz.faces
    # daz_vers, _, _ = daz()
    # daz_vers = daz_vers.detach().cpu().numpy()[0]
    # daz_vers_sel = daz_vers[index_daz_346]
    #
    # model_params = dict(model_path='/home/jnh/workspace/workspace_bak/smplx/download/models')
    # smpl = smplx.create(model_type='smpl', gender='female', **model_params)
    # smpl_face = smpl.faces
    # smpl_out = smpl()
    # vertices = smpl_out.vertices.detach().cpu().numpy()[0]
    # smpl_sel = vertices[index_smpl_346]
    #
    # mv.add_point_spheres(smpl_sel)
    # mv.add_point_spheres(daz_vers_sel, colors=[.5, .65, .9])
    # mv.viewer_update()
    # a = 1