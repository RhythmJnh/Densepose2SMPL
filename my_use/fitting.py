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
               pose_embedding=None,
               vposer=None,
               dtype=torch.float32,
               model_type='smpl',
               map_xy_now=None, 
               face_now=None, 
               bary_now=None, 
               ):

    body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1)

    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                              dtype=body_pose.dtype,
                              device=body_pose.device)
    body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    output = model(body_pose=body_pose, return_verts=True,
                   return_full_pose=False)

    vertices = output.vertices.detach().cpu().numpy()
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

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None, 
                    ):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
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

            # 一阶最优终止容忍度
            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(
                        1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)

                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss


    def create_fitting_closure(self,
                               optimizer, body_model, camera,
                               target_vertex_tensor, index_smpl_346_tensor,
                               loss=None,
                               plt=None,
                               show_loss=False,
                               show_loss_type=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=True, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               ):

        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                pose_embedding, output_type='aa').view(
                    1, -1) if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            total_loss = loss(body_model_output, camera,
                              target_vertex_tensor, index_smpl_346_tensor,
                              body_model_faces=faces_tensor,
                              plt=plt,
                              show_loss=show_loss,
                              show_loss_type=show_loss_type,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              )
            if backward:
                total_loss.backward(create_graph=create_graph)
            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

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

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 interpenetration=True, dtype=torch.float32,
                 angle_prior=None,
                 shape_prior=None,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 bending_prior_weight=0.0,
                 shape_weight=0.0,
                 coll_loss_weight=0.0,
                 reduction='sum',
                 ):

        super(SMPLifyLoss, self).__init__()

        self.angle_prior = angle_prior
        self.shape_prior = shape_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))

        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))

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

    def forward(self, body_model_output, camera,
                target_vertex_tensor, index_smpl_346_tensor,
                body_model_faces, plt,
                show_loss, show_loss_type, use_vposer=True, 
                pose_embedding=None,
                ):

        vertices = body_model_output.vertices
        source_vertex_tensor = vertices[0, index_smpl_346_tensor]

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        pro_diff = self.robustifier(target_vertex_tensor-source_vertex_tensor)
        pro_loss = (torch.sum(pro_diff) *100 ** 2)

        # Calculate the loss from the Pose prior
        pprior_loss = (pose_embedding.pow(2).sum() *
                       self.body_pose_weight ** 2)

        shape_loss = torch.sum(self.shape_prior(
          body_model_output.betas)) * self.shape_weight ** 2

        # print('pro_loss: {:.4f}'.format(pro_loss))
        # print('pprior_loss: {:.4f}'.format(pprior_loss))
        # print('shape_loss: {:.4f}'.format(shape_loss))
        # print('con_loss: {:.4f}'.format(con_loss))

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

        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight

        pen_loss = 0.0

        total_loss = (pro_loss + pprior_loss + shape_loss +
                      angle_prior_loss + pen_loss)# + con_loss)

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

    def forward(self, body_model_output, camera,
                target_vertex_tensor, index_smpl_346_tensor,
                dtype=torch.float32,
                device='cuda',
                **kwargs
                ):

        vertices = body_model_output.vertices
        source_vertex_tensor = vertices[0, index_smpl_346_tensor]

        pro_error = torch.pow(target_vertex_tensor-source_vertex_tensor, 2)

        joint_loss = torch.sum(pro_error) * 100 ** 2

        depth_loss = 0.0
        # if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
        #         None):
        #     depth_loss = self.depth_loss_weight ** 2 * torch.sum((
        #         camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + depth_loss
