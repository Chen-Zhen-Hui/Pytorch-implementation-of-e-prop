import torch.nn as nn
import torch.nn.functional as F
from config import *

def p(x):
    print(x.shape)


class SRNN(nn.Module):
    def __init__(self):
        super(SRNN, self).__init__()
        # Parameters
        self.w_in = nn.Parameter(torch.Tensor(n_rec, n_in)).to(device)
        self.w_rec = nn.Parameter(torch.Tensor(n_rec, n_rec)).to(device)
        self.w_out = nn.Parameter(torch.Tensor(n_out, n_rec)).to(device)
        self.reset_parameters(w_init_gain)

        self.v = torch.zeros(n_b, n_rec).to(device)
        self.vo = torch.zeros(n_b, n_out).to(device)
        self.z = torch.zeros(n_b, n_rec).to(device)

        self.Fkz = torch.zeros(n_b, n_rec).to(device)  # kappa filter for z
        self.Faz = torch.zeros(n_b, n_rec).to(device)  # alpha filter for z
        self.Fax = torch.zeros(n_b, n_in).to(device)  # alpha filter for x
        self.L = torch.zeros(n_b, n_rec).to(device)  # Learning signal
        self.e_rec = torch.zeros(n_b, n_rec, n_rec).to(device)  # eligibility for w_rec
        self.e_in = torch.zeros(n_b, n_rec, n_in).to(device)  # eligibility for w_in
        self.h = torch.zeros(n_b, n_rec).to(device)  # pseudo
        self.Fke_rec = torch.zeros(n_b, n_rec, n_rec).to(device)  # kappa filter for e_rec
        self.Fke_in = torch.zeros(n_b, n_rec, n_in).to(device)  # kappa filter for e_in

    def reset_parameters(self, gain):
        torch.nn.init.kaiming_normal_(self.w_in)
        self.w_in.data = gain[0] * self.w_in.data
        torch.nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1] * self.w_rec.data
        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2] * self.w_out.data
        # Weight gradients
        self.w_in.grad = torch.zeros_like(self.w_in)
        self.w_rec.grad = torch.zeros_like(self.w_rec)
        self.w_out.grad = torch.zeros_like(self.w_out)

        
    def forward(self, x, do_training, yt):
        self.w_rec *= (1 - torch.eye(n_rec, n_rec, device=device))  # Cancel self connected neurons

        self.v = alpha * self.v * (1 - self.z) + torch.mm(self.z, self.w_rec.t()) + torch.mm(x, self.w_in.t())
        self.Faz = self.Faz * alpha + self.z  # n_b, n_r
        self.z = (self.v > thr).float()
        self.vo = kappa * self.vo + torch.mm(self.z, self.w_out.t())
        if classif:  # Apply a softmax function for classification problems
            yo = F.softmax(self.vo, dim=-1)
        else:
            yo = self.vo
        if do_training:
            self.h = gamma * torch.max(torch.zeros_like(self.v), 1 - torch.abs((self.v - thr) / thr))  # n_b, n_r

            self.L = torch.mm((yo - yt), self.w_out)  # n_b, n_r

            self.Fkz = self.Fkz * kappa + self.z  # n_b, n_r
            self.w_out.grad += torch.mm((yo - yt).t(), self.Fkz)

            self.e_rec = self.h.unsqueeze(2) * self.Faz.unsqueeze(1)
            self.Fke_rec = self.Fke_rec * kappa + self.e_rec  # n_b, n_r, n_r
            self.w_rec.grad += torch.sum(self.L.unsqueeze(2) * self.Fke_rec, dim=0)

            self.Fax = self.Fax * alpha + x  # n_b, n_i
            self.e_in = self.h.unsqueeze(2) * self.Fax.unsqueeze(1)
            self.Fke_in = self.Fke_in * kappa + self.e_in  # n_b, n_i, n_r
            self.w_in.grad += torch.sum(self.L.unsqueeze(2) * self.Fke_in, dim=0)
        return yo
