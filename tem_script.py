import smplx
import numpy as np
from utils.daz import DazModel
from mesh_viewer import MeshViewer
import trimesh
import pickle

if __name__ == '__main__':

    data = pickle.load(open('/media/jnh/Jnh_T5/ihp_data/daz2smpl_5000_data/mini_data/0ce6a276cc7403723c090e14e32ea0d1.pickle', 'rb'))
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