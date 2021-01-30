# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import numpy as np

import torch
import torch.nn as nn

from my_use.mesh_viewer import MeshViewer
from my_use import utils


def getSmplVer(face_now, bc1, bc2, bc3, Vertices):

  p = Vertices[face_now[0],:] * bc1 +  \
      Vertices[face_now[1],:] * bc2 +  \
      Vertices[face_now[2],:] * bc3 

  return p


@torch.no_grad()
def guess_init(model,
               edge_idxs,
               device=None,
               focal_length=5000,
               dtype=torch.float32,
               map_xy_now=None,
               face_now=None, 
               bary_now=None, 
               ):

    vers, _, _ = model()

    vertices = vers.detach().cpu().numpy()
    surface_pt3 = []
    for i in range(4):
      p = getSmplVer(face_now[i], bary_now[i][0], bary_now[i][1], \
                      bary_now[i][2], vertices[0])

      surface_pt3.append(p)
    surface_pt2 = torch.tensor([map_xy_now], dtype=dtype, device=device)
    surface_pt3 = torch.tensor(np.array([surface_pt3]), dtype=dtype, \
                               device=device)

    diff3d = []
    diff2d = []
    for edge in edge_idxs:
        diff3d.append(surface_pt3[:, edge[0]] - surface_pt3[:, edge[1]])
        diff2d.append(surface_pt2[:, edge[0]] - surface_pt2[:, edge[1]])

    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)

    est_d = focal_length * (height3d / height2d)

    # just set the z value
    batch_size = 1
    x_coord = torch.zeros([batch_size], device=device,
                          dtype=dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=1e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 ):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, body_model, body_pose=None, go_pose=None, betas=None,
                    ):
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            # 参数变化量上的终止容忍度
            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            # # 一阶最优终止容忍度
            # if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
            #         for var in params if var.grad is not None]):
            #     break

            if self.visualize and n % self.summary_steps == 0:
                if body_pose is not None:
                    vers, _, _ = body_model(global_orient=go_pose,
                                            body_pose=body_pose,
                                            betas=betas)
                else:
                    vers, _, _ = body_model(global_orient=go_pose)
                vertices = vers.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss


    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_xy=None, 
                               xy2face_inds=None,
                               xy2bary_coords=None,
                               loss=None,
                               plt=None,
                               show_loss=False,
                               show_loss_type=None,
                               create_graph=False,
                               body_pose=None,
                               go_pose=None,
                               betas=None,
                               ):

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            if body_pose is not None:
                vers, _, _ = body_model(global_orient=go_pose,
                                        body_pose=body_pose,
                                        betas=betas)
            else:
                vers, _, _ = body_model(global_orient=go_pose)
            total_loss = loss(vers, camera=camera,
                              gt_xy=gt_xy,
                              xy2face_inds=xy2face_inds,
                              xy2bary_coords=xy2bary_coords,
                              plt=plt,
                              show_loss=show_loss,
                              show_loss_type=show_loss_type,
                              body_pose=body_pose,
                              betas=betas,
                              )
            if backward:
                total_loss.backward(create_graph=create_graph)
            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                if body_pose is not None:
                    vers, _, _ = body_model(global_orient=go_pose,
                                            body_pose=body_pose,
                                            betas=betas)
                else:
                    vers, _, _ = body_model(global_orient=go_pose)
                vertices = vers.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))

class SMPLifyLoss(nn.Module):

    def __init__(self,
                 rho=100,
                 dtype=torch.float32,
                 shape_prior=None,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 ):

        super(SMPLifyLoss, self).__init__()

        self.shape_prior = shape_prior
        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_xy,
                xy2face_inds,
                xy2bary_coords,
                plt,
                show_loss, show_loss_type,
                body_pose=None,
                betas=None,
                ):

        vertices = body_model_output

        new_smpl_ver = getSmplVer(xy2face_inds[0], xy2bary_coords[0][0], \
          xy2bary_coords[0][1], xy2bary_coords[0][2], vertices[0])

        new_smpl_ver = torch.unsqueeze(new_smpl_ver, 1)
        for i in range(1, gt_xy.shape[1]):
          p = getSmplVer(xy2face_inds[i], xy2bary_coords[i][0], \
            xy2bary_coords[i][1], xy2bary_coords[i][2], vertices[0])
          new_smpl_ver = torch.cat((new_smpl_ver, torch.unsqueeze(p,1)), dim=1)
        new_smpl_ver = torch.unsqueeze(new_smpl_ver.permute(1, 0), 0)

        pro_new_ver = camera(new_smpl_ver)

        pro_new_ver_np = pro_new_ver.detach().cpu().numpy()[0]
        gt_xy_np = gt_xy.detach().cpu().numpy()[0]
        import cv2
        from main import drawPts2img
        test_png = cv2.imread('/home/jnh/workspace/Densepose2SMPL/test_data/dp/test_dp.png')
        for i in range(gt_xy_np.shape[0]):
            cv2.circle(test_png, (int(gt_xy_np[i, 0]), int(gt_xy_np[i, 1])), 2, (0, 0, 255, 255), -1)
        drawPts2img(pro_new_ver_np, test_png)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        pro_diff = self.robustifier(gt_xy - pro_new_ver)
        pro_loss = (torch.sum(pro_diff) *
                      self.data_weight ** 2)

        # Calculate the loss from the Pose prior
        pprior_loss = (body_pose.pow(2).sum() *
                       self.body_pose_weight ** 2)

        shape_loss = torch.sum(self.shape_prior(betas)) * self.shape_weight ** 2

        # print('pro_loss: {:.4f}'.format(pro_loss))
        # print('pprior_loss: {:.4f}'.format(pprior_loss))
        # print('shape_loss: {:.4f}'.format(shape_loss))

        # Loss curve
        if show_loss:
            import time
            if show_loss_type == 'pro_loss':
                plt.plot(time.time(), pro_loss.detach().cpu().numpy(), '.')
                plt.title('{}:{}'.format(show_loss_type, pro_loss.detach().cpu().numpy()))
                print('joint_loss: {:.4f}'.format(pro_loss))
            if show_loss_type == 'pprior_loss':
                plt.plot(time.time(), pprior_loss.detach().cpu().numpy(), '.')
                plt.title('{}:{}'.format(show_loss_type, pprior_loss.detach().cpu().numpy()))
                print('pprior_loss: {:.4f}'.format(pprior_loss))
            if show_loss_type == 'shape_loss':
                plt.plot(time.time(), shape_loss.detach().cpu().numpy(), '.')
                plt.title('{}:{}'.format(show_loss_type, shape_loss.detach().cpu().numpy()))
                print('shape_loss: {:.4f}'.format(shape_loss))

            plt.draw()
            plt.pause(0.1)

        total_loss = pro_loss + pprior_loss + shape_loss

        return total_loss

class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2, 
                 dtype=torch.float32, 
                 ):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))

        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_xy,
                xy2face_inds,
                xy2bary_coords,
                **kwargs
                ):

        vertices = body_model_output

        new_smpl_ver = getSmplVer(xy2face_inds[0], xy2bary_coords[0][0], \
          xy2bary_coords[0][1], xy2bary_coords[0][2], vertices[0])
        new_smpl_ver = torch.unsqueeze(new_smpl_ver, 1)
        for i in range(1,4):
          p = getSmplVer(xy2face_inds[i], xy2bary_coords[i][0], \
            xy2bary_coords[i][1], xy2bary_coords[i][2], vertices[0])
          new_smpl_ver = torch.cat((new_smpl_ver, torch.unsqueeze(p,1)), dim=1)
        new_smpl_ver = torch.unsqueeze(new_smpl_ver.permute(1, 0), 0)

        pro_new_ver = camera(new_smpl_ver)

        pro_error = torch.pow(pro_new_ver - gt_xy, 2)

        joint_loss = torch.sum(pro_error) * self.data_weight ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + depth_loss
