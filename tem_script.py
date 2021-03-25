import smplx
import numpy as np
from utils.daz import DazModel
from mesh_viewer import MeshViewer
import trimesh
import pickle
from my_use.camera import create_camera
import cv2
import torch
# from my_use.utils import render
focal = 5000 * 28. / 234.7000
gender = 'female'
model_folder = '/home/jnh/workspace/workspace_bak/smplx/download/models'
dtype = torch.float32
from main import drawPts2img
import sys
sys.path.append('/home/jnh/workspace/workspace_bak/cloth_texture_fusion/cloth_texture_render/face3d')
from mesh import render

if __name__ == '__main__':

    # with open('/home/jnh/workspace/transfer_motion/UVTextureConverter/UVTextureConverter/config/normal_faces.pickle', mode='rb') as f:
    #     normal_faces = np.vstack(pickle.load(f))
    # with open('/home/jnh/workspace/transfer_motion/UVTextureConverter/UVTextureConverter/config/normal.pickle', mode='rb') as f:
    #     normal_hash = pickle.load(f)

    img = cv2.imread('/media/jnh/Jnh_T5/ihp_data/daz2smpl_5000_data/mini_data/0ce6a276cc7403723c090e14e32ea0d1.jpg')
    data = pickle.load(open('/media/jnh/Jnh_T5/ihp_data/daz2smpl_5000_data/mini_data/0ce6a276cc7403723c090e14e32ea0d1.pickle', 'rb'))
    data = data['smpl_data']['result']
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
                           center=torch.from_numpy(np.array([[256, 256]])),
                           dtype=dtype)
    # color = render(vertices, smpl_face, camera,
    #                focal=focal, depth_or_color='color', H=512, W=512)
    # color = color.astype(np.float32) / 255.0
    #
    # origin_img = img.astype(np.float32) / 255
    # valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis].astype(np.float32) * 1.0
    # show_img1 = origin_img * (1 - valid_mask) + color[:, :, :3] * valid_mask
    # cv2.imshow('test', show_img1)
    # cv2.waitKey()
    # exit()
    pro_vers = camera(smpl_out.vertices)
    pro_vers = pro_vers.detach().cpu().numpy()[0]
    # drawPts2img(pro_vers, img)

    from my_use.densepose_methods import DensePoseMethods
    dpm = DensePoseMethods()
    map_idx = dpm.All_vertices - 1
    all_face = dpm.FacesDensePose
    face_idx = dpm.FaceIndices

    # smpl_uv = np.load('/home/jnh/workspace/transfer_motion/detectron2/projects/DensePose/tem_file/uv_table.npy')
    # smpl_uv = smpl_uv[map_idx]
    smpl_uv = np.vstack((dpm.U_norm, dpm.V_norm)).T
    smpl_uv[:, 1] = 1 - smpl_uv[:, 1]
    pro_vers = pro_vers[map_idx]
    final_uv_size = 200
    vt_coords = smpl_uv * final_uv_size
    # smpl_info = pickle.load(open('/home/jnh/workspace/workspace_bak/smplx/download/models/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl', 'rb'), encoding='Latin')
    # face = smpl_info['f']

    # d = pro_vers[:, 2]
    # d_max = abs(np.max(d))
    # d_min = abs(np.min(d))
    # pro_vers[:, 2] = ((d-d_min) / (d_max-d_min))[:]
    vt_coords_d = pro_vers[:, 2]
    vt_coords = np.concatenate((vt_coords, np.expand_dims(vt_coords_d, 1)), 1)

    all_tex = np.zeros((24, final_uv_size, final_uv_size, 3))

    for i in range(1, 25):
        face = all_face[face_idx == i]
        uv_texture_map, depth = render.render_texture(vt_coords, face, img.astype(np.float32) / 255, pro_vers,
                                                  face, final_uv_size, final_uv_size, c=3, mapping_type='nearest')

        all_tex[i-1] = uv_texture_map.transpose([1, 0, 2])

    tex = None
    for i in range(0, 4):
        tex_tmp = all_tex[6 * i]
        for i in range(1 + 6 * i, 6 + 6 * i):
            tex_tmp = np.concatenate((tex_tmp, all_tex[i]), axis=1)
        if tex is None:
            tex = tex_tmp
        else:
            tex = np.concatenate((tex, tex_tmp), axis=0)

    tex = tex.transpose([1, 0, 2])
    if 1:
        # cv2.namedWindow('uv_texture_map', 0)
        # cv2.imshow('uv_texture_map', tex)
        # cv2.waitKey()
        cv2.imwrite('test_tex.png', tex*255.)


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