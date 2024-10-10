# -*- coding: UTF-8 -*-

import torch
import numpy as np
from utils.NODE import *
from utils.parameters import *
from utils.skew import *


class ABNODE_p2(ODEF):
    """
    Combining RGBlimp dynamics system and a neural network
    """

    def __init__(self, RGBlimp_params, device):
        super(ABNODE_p2, self).__init__()
        torch.set_default_dtype(torch.float64)
        self.nn_model = nn.ModuleList([nn.Linear(23, 256),
                                       nn.Tanh(),
                                       nn.Linear(256, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, 6)]) 

        torch.nn.init.xavier_uniform_(self.nn_model[0].weight, gain=0.08)
        torch.nn.init.xavier_uniform_(self.nn_model[2].weight, gain=0.08)
        torch.nn.init.xavier_uniform_(self.nn_model[4].weight, gain=0.08)
        
        self.device = device
        self.nn_model = self.nn_model.to(self.device)

        self.Rho = torch.tensor([RGBlimp_params['Rho']], dtype=torch.double, requires_grad=False).to(self.device)
        self.rho = torch.tensor([RGBlimp_params['rho']], dtype=torch.double, requires_grad=False).to(self.device)

        self.g = torch.tensor([RGBlimp_params['g']], dtype=torch.double, requires_grad=False).to(self.device)
        self.B = torch.tensor([RGBlimp_params['B']], dtype=torch.double, requires_grad=False).to(self.device)
        self.m = torch.tensor([RGBlimp_params['m']], dtype=torch.double, requires_grad=False).to(self.device)
        self.mb = torch.tensor([RGBlimp_params['mb']], dtype=torch.double, requires_grad=False).to(self.device)
        self.G = torch.tensor([RGBlimp_params['G']], dtype=torch.double, requires_grad=False).to(self.device)

        self.r = torch.tensor(RGBlimp_params['r'], dtype=torch.double, requires_grad=False).to(self.device)
        self.rb = torch.tensor(RGBlimp_params['rb'], dtype=torch.double, requires_grad=False).to(self.device)
        self.d = torch.tensor([RGBlimp_params['d']], dtype=torch.double, requires_grad=False).to(self.device)
        self.A = torch.tensor([RGBlimp_params['A']], dtype=torch.double, requires_grad=False).to(self.device)

        self.Ix = torch.tensor([RGBlimp_params['Ix']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Iy = torch.tensor([RGBlimp_params['Iy']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Iz = torch.tensor([RGBlimp_params['Iz']], dtype=torch.double, requires_grad=False).to(self.device)
        self.I = torch.diag(torch.tensor([self.Ix, self.Iy, self.Iz], dtype=torch.double, requires_grad=False)).to(
            self.device)

        self.K1 = torch.tensor([RGBlimp_params['K1']], dtype=torch.double, requires_grad=False).to(self.device)
        self.K2 = torch.tensor([RGBlimp_params['K2']], dtype=torch.double, requires_grad=False).to(self.device)
        self.K3 = torch.tensor([RGBlimp_params['K3']], dtype=torch.double, requires_grad=False).to(self.device)

        self.Cd0 = torch.tensor([RGBlimp_params['Cd0']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cda = torch.tensor([RGBlimp_params['Cda']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cdb = torch.tensor([RGBlimp_params['Cdb']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cl0 = torch.tensor([RGBlimp_params['Cl0']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cla = torch.tensor([RGBlimp_params['Cla']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Clb = torch.tensor([RGBlimp_params['Clb']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cmy0 = torch.tensor([RGBlimp_params['Cmy0']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cmya = torch.tensor([RGBlimp_params['Cmya']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cmyb = torch.tensor([RGBlimp_params['Cmyb']], dtype=torch.double, requires_grad=False).to(self.device)

        self.Cs0 = torch.tensor([RGBlimp_params['Cs0']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Csa = torch.tensor([RGBlimp_params['Csa']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Csb = torch.tensor([RGBlimp_params['Csb']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cmx0 = torch.tensor([RGBlimp_params['Cmx0']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cmxa = torch.tensor([RGBlimp_params['Cmxa']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cmxb = torch.tensor([RGBlimp_params['Cmxb']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cmz0 = torch.tensor([RGBlimp_params['Cmz0']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cmza = torch.tensor([RGBlimp_params['Cmza']], dtype=torch.double, requires_grad=False).to(self.device)
        self.Cmzb = torch.tensor([RGBlimp_params['Cmzb']], dtype=torch.double, requires_grad=False).to(self.device)

    def forward(self, z):
        z = z.type(torch.double)
        # z shape 1*23
        z = z.to(self.device)
        bs = z.size()[0]
        z = z.squeeze(1)
        x = z[:, :18]  # p,e,v,w,rb,rb_dot
        u = z[:, 18:]  # Fl,Fr,rb_dot_dot
        p = z[:, 0:3]
        e = z[:, 3:6]
        v = z[:, 6:9]
        w = z[:, 9:12]
        rb = z[:, 12:15]
        rb_dot = z[:, 15:18]

        rb_dot_dot = z[:, 20:23]
        eps = torch.tensor([1e-10], dtype=torch.double, requires_grad=False).to(self.device)
        V = torch.linalg.norm(v)
        alpha = torch.atan2(v[:, 2], v[:, 0] + eps)
        beta = torch.asin(v[:, 1] / (V + eps))

        D = 1 / 2 * self.Rho * V * V * self.A * (self.Cd0 + self.Cda * alpha * alpha + self.Cdb * beta * beta)
        S = 1 / 2 * self.Rho * V * V * self.A * (self.Cs0 + self.Csa * alpha * alpha + self.Csb * beta)
        L = 1 / 2 * self.Rho * V * V * self.A * (self.Cl0 + self.Cla * alpha + self.Clb * beta * beta)
        M1 = 1 / 2 * self.Rho * V * V * self.A * (self.Cmx0 + self.Cmxa * alpha + self.Cmxb * beta)
        M2 = 1 / 2 * self.Rho * V * V * self.A * (self.Cmy0 + self.Cmya * alpha + self.Cmyb * beta * beta* beta * beta)
        M3 = 1 / 2 * self.Rho * V * V * self.A * (self.Cmz0 + self.Cmza * alpha + self.Cmzb * beta)
        Damping = torch.cat(
            [(self.K1 * w[:, 0]).unsqueeze(1).unsqueeze(1), (self.K2 * w[:, 1]).unsqueeze(1).unsqueeze(1),
             (self.K3 * w[:, 2]).unsqueeze(1).unsqueeze(1)], 1)  # bs*3*1

        # Rotation Matrix bs*3*3
        Rbi_l1 = torch.cat([(torch.cos(e[:, 1]) * torch.cos(e[:, 2])).unsqueeze(1),
                            (torch.cos(e[:, 1]) * torch.sin(e[:, 2])).unsqueeze(1),
                            -torch.sin(e[:, 1]).unsqueeze(1)], 1).unsqueeze(1)  # bs*1*3
        Rbi_l2 = torch.cat([(torch.sin(e[:, 0]) * torch.sin(e[:, 1]) * torch.cos(e[:, 2]) - torch.cos(
            e[:, 0]) * torch.sin(e[:, 2])).unsqueeze(1),
                            (torch.sin(e[:, 0]) * torch.sin(e[:, 1]) * torch.sin(e[:, 2]) + torch.cos(
                                e[:, 0]) * torch.cos(e[:, 2])).unsqueeze(1),
                            (torch.sin(e[:, 0]) * torch.cos(e[:, 1])).unsqueeze(1)], 1).unsqueeze(1)
        Rbi_l3 = torch.cat([(torch.cos(e[:, 0]) * torch.sin(e[:, 1]) * torch.cos(e[:, 2]) + torch.sin(
            e[:, 0]) * torch.sin(e[:, 2])).unsqueeze(1),
                            (torch.cos(e[:, 0]) * torch.sin(e[:, 1]) * torch.sin(e[:, 2]) - torch.sin(
                                e[:, 0]) * torch.cos(e[:, 2])).unsqueeze(1),
                            (torch.cos(e[:, 0]) * torch.cos(e[:, 1])).unsqueeze(1)], 1).unsqueeze(1)
        Rbi = torch.cat([Rbi_l1, Rbi_l2, Rbi_l3], 1)

        # Rbv bs*3*3
        Rbv_l1 = torch.cat([(torch.cos(alpha) * torch.cos(beta)).unsqueeze(1),
                            (-torch.cos(alpha) * torch.sin(beta)).unsqueeze(1),
                            -torch.sin(alpha).unsqueeze(1)], 1).unsqueeze(1)
        Rbv_l2 = torch.cat([torch.sin(beta).unsqueeze(1),
                            torch.cos(beta).unsqueeze(1),
                            torch.zeros(bs, 1).to(self.device)], 1).unsqueeze(1)
        Rbv_l3 = torch.cat([(torch.sin(alpha) * torch.cos(beta)).unsqueeze(1),
                            (-torch.sin(alpha) * torch.sin(beta)).unsqueeze(1),
                            (torch.cos(alpha)).unsqueeze(1)], 1).unsqueeze(1)
        Rbv = torch.cat([Rbv_l1, Rbv_l2, Rbv_l3], 1)

        # Jib bs*3*3
        Jib_l1 = torch.cat([torch.ones(bs, 1).to(self.device),
                            (torch.sin(e[:, 0]) * torch.tan(e[:, 1])).unsqueeze(1),
                            (torch.cos(e[:, 0]) * torch.tan(e[:, 1])).unsqueeze(1)], 1).unsqueeze(1)
        Jib_l2 = torch.cat([torch.zeros(bs, 1).to(self.device),
                            torch.cos(e[:, 0]).unsqueeze(1),
                            -torch.sin(e[:, 0]).unsqueeze(1)], 1).unsqueeze(1)
        Jib_l3 = torch.cat([torch.zeros(bs, 1).to(self.device),
                            (torch.sin(e[:, 0]) / torch.cos(e[:, 1])).unsqueeze(1),
                            (torch.cos(e[:, 0]) / torch.cos(e[:, 1])).unsqueeze(1)], 1).unsqueeze(1)
        Jib = torch.cat([Jib_l1, Jib_l2, Jib_l3], 1)
        
        
        # lg bs*3
        lg = (self.m * self.r).reshape((1, 3)) + self.mb * rb
        # ff bs*3*1
        ff = ((self.m + self.mb) * torch.cross(v, w)).reshape((-1, 3, 1)) + \
             torch.cross(torch.cross(w, lg), w).reshape((-1, 3, 1)) + \
             Rbi @ torch.tensor([[0], [0], [(self.m + self.mb - self.B) * self.g]], dtype=torch.double,
                                requires_grad=False).to(self.device) + \
             Rbv @ torch.cat([-D.reshape([-1, 1]), S.reshape([-1, 1]), -L.reshape([-1, 1])], 1).unsqueeze(2)

        # ff bs*3
        ff = ff.squeeze(2)
        ff[:, 0] = ff[:, 0] + u[:, 0] + u[:, 1]

        # tt bs*3*1
        tt = torch.cross((self.I - self.mb * skew(rb,self.device).to(self.device) @ skew(rb,self.device).to(self.device)) @ w.unsqueeze(2),
                         w.unsqueeze(2)) + torch.cross(lg, torch.cross(v, w)).unsqueeze(2) + \
             torch.cross(lg, (Rbi @ torch.tensor([[0], [0], [self.g]], dtype=torch.double, requires_grad=False).to(
                 self.device)).squeeze(2)).unsqueeze(2) + \
             (Rbv @ (torch.cat([M1.reshape([-1, 1]), M2.reshape([-1, 1]), M3.reshape([-1, 1])], 1).unsqueeze(2)+Damping))
             
        # Rbv... bs*3*3 3*1 bs*3*1 bs*3
        tt[:, 2] = tt[:, 2] + ((u[:, 0] - u[:, 1]) * self.d).unsqueeze(1)
        tt[:, 1] = tt[:, 1] + ((u[:, 0] + u[:, 1]) * rb[:, 2]).unsqueeze(1)
        # tt bs*3
        tt = tt.squeeze(2)
        # H bs*6*6
        H_row1 = torch.stack(
            [((self.m + self.mb) * torch.eye(3).to(self.device)).repeat(bs, 1, 1), -skew(lg,self.device).to(self.device)],
            2).reshape(bs, 3, 6)
        H_row2 = torch.stack(
            [skew(lg,self.device).to(self.device), self.I - self.mb * skew(rb,self.device).to(self.device) @ skew(rb,self.device).to(self.device)],
            2).reshape(bs, 3, 6)
        H = torch.stack([H_row1, H_row2], 1).reshape(bs, 6, 6)

        # torchversion>1.8
        # temp_result=torch.linalg.solve(H,(torch.stack([ff,tt],1)).reshape(bs,6,1))
        # torchversion<1.8
        temp_result = torch.inverse(H) @ ((torch.stack([ff, tt], 1)).reshape(bs, 6, 1))
        # temp_result bs*6*1
        temp_result = temp_result.reshape(-1, 6)
        # temp_result bs*6

        # dot_p bs*3*3 @ bs*3 --bs*3
        dot_p = (Rbi.transpose(1, 2) @ v.unsqueeze(2)).squeeze(2)
        # dot_e --bs*3
        dot_e = (Jib @ w.unsqueeze(2)).squeeze(2)
        # dot_v --bs*3
        dot_v = temp_result[:, 0:3]
        dot_w = temp_result[:, 3:6]
        dot_u = torch.zeros((dot_w.shape[0], 5)).to(self.device)
        y = torch.cat([dot_p, dot_e, dot_v, dot_w, rb_dot, rb_dot_dot, dot_u], 1)
        temp = z
        for i, l in enumerate(self.nn_model):
            temp = l(temp)
        y[:, 6:12] = y[:, 6:12] + temp
        return y
