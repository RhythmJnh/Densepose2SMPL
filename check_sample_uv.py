import os
import numpy as np
import pickle
import cv2
from utils.mesh_viewer import MeshViewer
from model.daz import DazModel
import trimesh



def getSmplVer(face_now, bc1, bc2, bc3, Vertices):

  p = Vertices[face_now[0],:] * bc1 +  \
      Vertices[face_now[1],:] * bc2 +  \
      Vertices[face_now[2],:] * bc3

  return p


def main(num):
    res = pickle.load(open('check_sample_uv.pkl', 'rb'))
    face = res['face']
    bary = res['bary']
    uvxy = res['uvxy'][0]

    test_png = cv2.imread('/home/jnh/workspace/Densepose2SMPL/test_data/ori/test.png')
    for i in range(1):
    # for i in range(uvxy.shape[0]):
        cv2.circle(test_png, (int(uvxy[num, 0]), int(uvxy[num, 1])), 5, (0, 0, 255, 255), -1)

    cv2.imshow('test', test_png)
    cv2.waitKey()
    daz = DazModel()
    vers, _, _ = daz()
    vers = vers.detach().cpu().numpy()[0]
    daz_mesh = trimesh.Trimesh(vers, daz.faces, process=False)

    sample_3d = []
    for i in range(1):
    # for i in range(len(face)):
        now_face = face[num]
        now_bary = bary[num]
        now_vers = getSmplVer(now_face, now_bary[0], now_bary[1], now_bary[2], vers)
        sample_3d.append(now_vers)

    mv = MeshViewer()
    mv.set_meshes([daz_mesh])
    mv.add_point_spheres(sample_3d, radius=0.01)
    mv.viewer_update()

    a = 1


if __name__ == '__main__':
    for i in range(20, 150):
        main(i)

