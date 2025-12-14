import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

import pandas as pd
import scipy
from scipy.linalg import orth, null_space
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt


import sklearn
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

import matplotlib.pyplot as plt
from losscape.create_landscape import create_2D_losscape
from plots import plot_accs, plot_grads

class NN(nn.Module):
    
    def __init__(self, input_dims, output_dims, hidden_dims):
        super().__init__()
        stdv1 = 1. / math.sqrt(hidden_dims)
        stdv2 = 1. / math.sqrt(hidden_dims)
        stdv3 = 1. / math.sqrt(output_dims)

        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, output_dims)

        self.b1 = nn.Parameter(torch.rand(hidden_dims) * 2 * stdv1 - stdv1)
        self.b2 = nn.Parameter(torch.rand(hidden_dims) * 2 * stdv2 - stdv2)
        self.b3 = nn.Parameter(torch.rand(output_dims) * 2 * stdv3 - stdv3)

        self.W1 = nn.Parameter(torch.rand(hidden_dims, input_dims) * 2 * stdv1 - stdv1)
        self.W2 = nn.Parameter(torch.rand(hidden_dims, hidden_dims) * 2 * stdv2 - stdv2)
        self.W3 = nn.Parameter(torch.rand(output_dims, hidden_dims) * 2 * stdv3 - stdv3)

        self.relu = torch.relu

    def forward(self, x):
        x = self.relu(torch.einsum("ij, bj->bi", self.W1, x) + self.b1)
        x = self.relu(torch.einsum("ij, bj->bi", self.W2,x) + self.b2)
        return torch.einsum("ij, bj->bi", self.W3, x) + self.b3

class NeuralNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, O1=None, O2=None, O3=None, U1=None, U2=None, U3=None, V1=None, V2=None, V3=None, device=None):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.device = device

        stdv1 = math.sqrt(1. / math.sqrt(hidden_dims))
        stdv2 = math.sqrt(1. / math.sqrt(hidden_dims))
        stdv3 = math.sqrt(1. / math.sqrt(output_dims))

        # outer products from previous rank
        self.O1 = O1
        self.O2 = O2
        self.O3 = O3

        # bias parameters
        self.b1 = nn.Parameter(torch.rand(hidden_dims) * 2 * stdv1**2 - stdv1**2)
        self.b2 = nn.Parameter(torch.rand(hidden_dims) * 2 * stdv2**2 - stdv2**2)
        self.b3 = nn.Parameter(torch.rand(output_dims) * 2 * stdv3**2 - stdv3**2)
    
       # whether we use orthonormal update
        self.ortho = False

        # take the previous singular vectors and construct change of basis into their nullspace
        self.first_layer = True
        self.second_layer = True
        self.third_layer = True

        if U1 is not None:

            self.ortho = True

            # construct the changes of basis for the left and right singular vectors
            self.B1, self.B2, self.B3, self.C1, self.C2, self.C3 = self.get_basis_matrices(U1, U2, U3, V1, V2, V3)

            # check if nullity is zero, stop updating those layers if that is the case
            if (self.B1.shape[0] * self.B1.shape[1]) == 0:
                self.first_layer = False
            
            if (self.B2.shape[0] * self.B2.shape[1]) == 0:
                self.second_layer = False

            if (self.B3.shape[0] * self.B3.shape[1]) == 0:
                self.third_layer = False

            if (self.C1.shape[0] * self.C1.shape[1]) == 0:
                self.first_layer = False

            if (self.C2.shape[0] * self.C2.shape[1]) == 0:
                self.second_layer = False

            if (self.C3.shape[0] * self.C3.shape[1]) == 0:
                self.third_layer = False       


        if self.ortho:
            self.O1.requires_grad_(False)
            self.O2.requires_grad_(False)
            self.O3.requires_grad_(False)

            # self.Ui and self.Vj are coordinates of the singular vectors, to be multiplied by the basis matrices
            if self.first_layer:
                self.U1 = nn.Parameter(1.0 * (torch.rand(np.max([1, self.B1.shape[1]]), 1) * 2 * stdv1 - stdv1))
                self.V1 = nn.Parameter(1.0 * (torch.rand(1, np.max([1, self.C1.shape[0]])) * 2 * stdv1 - stdv1))

            if self.second_layer:
                self.U2 = nn.Parameter(1.0 * (torch.rand(np.max([1, self.B2.shape[1]]), 1) * 2 * stdv2 - stdv2))
                self.V2 = nn.Parameter(1.0 * (torch.rand(1, np.max([1, self.C2.shape[0]])) * 2 * stdv2 - stdv2))
        
            if self.third_layer:
                self.U3 = nn.Parameter(1.0 * (torch.rand(np.max([1, self.B3.shape[1]]), 1) * 2 * stdv3 - stdv3))
                self.V3 = nn.Parameter(1.0 * (torch.rand(1, np.max([1, self.C3.shape[0]])) * 2 * stdv3 - stdv3))

        else:
            self.U1 = nn.Parameter(1.0 * (torch.rand(hidden_dims, 1) * 2 * stdv1 - stdv1))
            self.V1 = nn.Parameter(1.0 * (torch.rand(1, input_dims) * 2 * stdv1 - stdv1 ))

            self.U2 = nn.Parameter(1.0 * (torch.rand(hidden_dims, 1) * 2 * stdv2 - stdv2))
            self.V2 = nn.Parameter(1.0 * (torch.rand(1, hidden_dims)* 2 * stdv2 - stdv2 )) 

            self.U3 = nn.Parameter(1.0 * (torch.rand(output_dims, 1) * 2 * stdv3 - stdv3))
            self.V3 = nn.Parameter(1.0 * (torch.rand(1, hidden_dims) * 2 * stdv3 - stdv3))

            

        self.relu = torch.relu

    def get_basis_matrices(self, U1, U2, U3, V1, V2, V3):

        U1 = U1.cpu()
        B1 = torch.tensor(null_space(U1.T))
        B1 = B1.to(self.device)

        U2 = U2.cpu()
        B2 = torch.tensor(null_space(U2.T))
        B2 = B2.to(self.device)

        U3 = U3.cpu()
        B3 = torch.tensor(null_space(U3.T))
        B3 = B3.to(self.device)

        V1 = V1.cpu()
        C1 = torch.tensor(null_space(V1).T)
        C1 = C1.to(self.device)

        V2 = V2.cpu()
        C2 = torch.tensor(null_space(V2).T)
        C2 = C2.to(self.device)

        V3 = V3.cpu()
        C3 = torch.tensor(null_space(V3).T)
        C3 = C3.to(self.device)
        
        return B1, B2, B3, C1, C2, C3
       
    def normalize_params(self):
     
        print(self.U1.shape, self.U2.shape, self.U3.shape)
        print()
        print(self.V1.shape, self.V2.shape, self.V3.shape)

        self.U1.data.div_(torch.norm(self.U1, dim=0, keepdim=True))
        self.U2.data.div_(torch.norm(self.U2, dim=0, keepdim=True))
        self.U3.data.div_(torch.norm(self.U3, dim=0, keepdim=True))

        self.V1.data.div_(torch.norm(self.V1, dim=1, keepdim=True))
        self.V2.data.div_(torch.norm(self.V2, dim=1, keepdim=True))
        self.V3.data.div_(torch.norm(self.U3, dim=1, keepdim=True))

    def forward(self, x):
     
        if self.ortho:
            if self.first_layer:
                W1 = (self.B1 @ self.U1) @ (self.V1 @ self.C1)
            else:
                W1 = torch.zeros_like(self.O1)
            if self.second_layer:
                W2 = (self.B2 @ self.U2) @ (self.V2 @ self.C2)
            else:
                W2 = torch.zeros_like(self.O2)
            if self.third_layer:
                W3 = (self.B3 @ self.U3) @ (self.V3 @ self.C3)
            else:
                W3 = torch.zeros_like(self.O3)

        else:
            W1 = self.U1 @ self.V1
            W2 = self.U2 @ self.V2
            W3 = self.U3 @ self.V3

        # add the outer products from the previous ranks to current outer products
        if self.O1 is not None:
     
            W1 = W1 + self.O1
            W2 = W2 + self.O2
            W3 = W3 + self.O3
    
        x = self.relu(torch.einsum("ij, bj->bi", W1, x) + self.b1)
        x = self.relu(torch.einsum("ij, bj->bi", W2,x) + self.b2)
        return torch.einsum("ij, bj->bi", W3, x) + self.b3

    def get_matrices(self):

        if self.ortho:

            # construct current weight matrices from singular vectors
            if self.first_layer:
                W1 = (self.B1 @ self.U1) @ (self.V1 @ self.C1)
            else:
                W1 = torch.zeros_like(self.O1)
            if self.second_layer:
                W2 = (self.B2 @ self.U2) @ (self.V2 @ self.C2)
            else:
                W2 = torch.zeros_like(self.O2)
            if self.third_layer:
                W3 = (self.B3 @ self.U3) @ (self.V3 @ self.C3)
            else:
                W3 = torch.zeros_like(self.O3)


            # add outer products from previous ranks
            W1 += self.O1
            W2 += self.O2
            W3 += self.O3

            # construct the current singular vectors
            if self.first_layer:
                U1 = (self.B1 @ self.U1).detach().requires_grad_(False)
                V1 = (self.V1 @ self.C1).detach().requires_grad_(False)
            else:
                U1 = None
                V1 = None

            if self.second_layer:
                U2 = (self.B2 @ self.U2).detach().requires_grad_(False)
                V2 = (self.V2 @ self.C2).detach().requires_grad_(False)
            else:
                U2 = None
                V2 = None

            if self.third_layer:
                U3 = (self.B3 @ self.U3).detach().requires_grad_(False)
                V3 = (self.V3 @ self.C3).detach().requires_grad_(False)
            else:
                U3 = None
                V3 = None

        else:

            # construct current weight matrices
            W1 = self.U1 @ self.V1
            W2 = self.U2 @ self.V2
            W3 = self.U3 @ self.V3

            # collect current singular vectors
            U1, U2, U3, V1, V2, V3 = self.U1.detach().requires_grad_(False),\
                self.U2.detach().requires_grad_(False), self.U3.detach().requires_grad_(False), self.V1.detach().requires_grad_(False),\
                self.V2.detach().requires_grad_(False), self.V3.detach().requires_grad_(False)


        # detach and return everything for use in the next rank
        return W1.detach().requires_grad_(False), W2.detach().requires_grad_(False), W3.detach().requires_grad_(False), \
                    U1, U2, U3, V1, V2, V3 

    def get_parameter_norms(self):
        # construct current weight matrices from singular vectors
        if self.first_layer:
            layer_1_norms = torch.norm(self.U1) + torch.norm(self.V1)
        else:
            layer_1_norms = 0.0

        if self.second_layer:
            layer_2_norms = torch.norm(self.U2) + torch.norm(self.V2)
        else:
            layer_2_norms = 0.0
            
        if self.third_layer:
            layer_3_norms = torch.norm(self.U3) + torch.norm(self.V3)
        else:
            layer_3_norms = 0.0

        bias_norms = torch.norm(self.b1) + torch.norm(self.b2) + torch.norm(self.b3)

        return bias_norms + layer_1_norms + layer_2_norms + layer_3_norms


class DummyNeuralNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, rank, O1=None, O2=None, O3=None, U1=None, U2=None, U3=None, V1=None, V2=None, V3=None, device=None):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.device = device

        stdv1 = math.sqrt(1. / math.sqrt(hidden_dims))
        stdv2 = math.sqrt(1. / math.sqrt(hidden_dims))
        stdv3 = math.sqrt(1. / math.sqrt(output_dims))

        # outer products from previous rank
        self.O1 = O1
        self.O2 = O2
        self.O3 = O3

        # power law parameters for singular values

        # self.coeff = nn.Parameter(torch.tensor(1.0))
        # self.power = nn.Parameter(torch.tensor(-2.0))

        # bias parameters
        self.b1 = torch.rand(hidden_dims) * 2 * stdv1**2 - stdv1**2
        self.b2 = torch.rand(hidden_dims) * 2 * stdv2**2 - stdv2**2
        self.b3 = torch.rand(output_dims) * 2 * stdv3**2 - stdv3**2
    
       # whether we use orthonormal update
        self.ortho = False

        # take the previous singular vectors and construct change of basis into their nullspace
        self.first_layer = True
        self.second_layer = True
        self.third_layer = True

        if U1 is not None:

            self.ortho = True

            # construct the changes of basis for the left and right singular vectors
            self.B1, self.B2, self.B3, self.C1, self.C2, self.C3 = self.get_basis_matrices(U1, U2, U3, V1, V2, V3)
            # check if nullity is zero, stop updating those layers if that is the case
            if (self.B1.shape[0] * self.B1.shape[1]) == 0:
                self.first_layer = False
                # self.B1 = 1.0 * torch.eye(1).type(torch.double)

            if (self.B2.shape[0] * self.B2.shape[1]) == 0:
                self.second_layer = False
                # self.B2 = 1.0 * torch.eye(1).type(torch.double)

            if (self.B3.shape[0] * self.B3.shape[1]) == 0:
                self.third_layer = False
                # self.B3 = 1.0 * torch.eye(1).type(torch.double)

            if (self.C1.shape[0] * self.C1.shape[1]) == 0:
                self.first_layer = False
                # self.C1 = 1.0 * torch.eye(1).type(torch.double)

            if (self.C2.shape[0] * self.C2.shape[1]) == 0:
                self.second_layer = False
                # self.C2 = 1.0 * torch.eye(1).type(torch.double)

            if (self.C3.shape[0] * self.C3.shape[1]) == 0:
                self.third_layer = False
                # self.C3 = 1.0 * torch.eye(1).type(torch.double)
       


        if self.ortho:
            self.O1.requires_grad_(False)
            self.O2.requires_grad_(False)
            self.O3.requires_grad_(False)

            # self.Ui and self.Vj are coordinates of the singular vectors, to be multiplied by the basis matrices
            self.U1 = 1.0 * (torch.rand(np.max([1, self.B1.shape[1]]), 1) * 2 * stdv1 - stdv1)#.type(torch.double))
            self.V1 = 1.0 * (torch.rand(1, np.max([1, self.C1.shape[0]])) * 2 * stdv1 - stdv1) #.type(torch.double))

            self.U2 = 1.0 * (torch.rand(np.max([1, self.B2.shape[1]]), 1) * 2 * stdv2 - stdv2) #.type(torch.double))
            self.V2 = 1.0 * (torch.rand(1, np.max([1, self.C2.shape[0]])) * 2 * stdv2 - stdv2)#.type(torch.double))
            # print("Initial Norm: ", torch.norm(self.U2))
            # print("Initial Norm: ", torch.norm(self
            self.U3 = 1.0 * (torch.rand(np.max([1, self.B3.shape[1]]), 1) * 2 * stdv3 - stdv3)#.type(torch.double))
            self.V3 = 1.0 * (torch.rand(1, np.max([1, self.C3.shape[0]])) * 2 * stdv3 - stdv3)#.type(torch.double))

        else:
            self.U1 = 1.0 * (torch.rand(hidden_dims, 1) * 2 * stdv1 - stdv1) #.type(torch.doule))
            self.V1 = 1.0 * (torch.rand(1, input_dims) * 2 * stdv1 - stdv1 )#.type(torch.doube))

            self.U2 = 1.0 * (torch.rand(hidden_dims, 1) * 2 * stdv2 - stdv2) #.type(torch.doule))
            self.V2 = 1.0 * (torch.rand(1, hidden_dims)* 2 * stdv2 - stdv2 )#.type(torch.doule))
            # print("Initial Norm: ", torch.norm(self.U2))
            # print("Initial Norm: ", torch.norm(self.V2))

            self.U3 = 1.0 * (torch.rand(output_dims, 1) * 2 * stdv3 - stdv3) #.type(torch.double))
            self.V3 = 1.0 * (torch.rand(1, hidden_dims) * 2 * stdv3 - stdv3) #.type(torch.double))

            

        self.relu = torch.relu

        # self.normalize_params()

    # unstable
    def gram_schmidt(self, V):
        # Gram Schidt on Columns of V
        U = []
        for i, v in enumerate(V.T):
            u = torch.clone(v)
            for j in range(i):
                proj = (torch.dot(v, U[j]) / torch.dot(U[j], U[j])) * U[j]
                u -= proj
            u /= torch.norm(u)
            U.append(u)
        return torch.tensor(np.array(U)).T

    def get_basis_matrices(self, U1, U2, U3, V1, V2, V3):

        # n, m = U1.shape
        # A = torch.hstack([U1, torch.randn((n, n-m))])
        # B1 = self.gram_schmidt(A)[:, m:]
        # B1 = torch.tensor(orth(A)[:, m:])
        U1 = U1.cpu()
        B1 = torch.tensor(null_space(U1.T))
        B1 = B1.to(self.device)

        # n, m = U2.shape
        # A = torch.hstack([U2, torch.randn((n, n-m))])
        # B2 = self.gram_schmidt(A)[:, m:]
        # B2 = torch.tensor(orth(A)[:, m:])
        U2 = U2.cpu()
        B2 = torch.tensor(null_space(U2.T))
        B2 = B2.to(self.device)

        # n, m = U3.shape
        # A = torch.hstack([U3, torch.randn((n, n-m))])
        # B3 = self.gram_schmidt(A)[:, m:]
        # B3 = torch.tensor(orth(A)[:, m:])
        U3 = U3.cpu()
        B3 = torch.tensor(null_space(U3.T))
        B3 = B3.to(self.device)

        # n, m = V1.shape
        # A = torch.hstack([V1.T, torch.randn((n, n-m))])
        # C1 = self.gram_schmidt(A)[:, m:].T
        # C1 = torch.tensor(orth(A)[:, m:].T)
        V1 = V1.cpu()
        C1 = torch.tensor(null_space(V1).T)
        C1 = C1.to(self.device)

        # n, m = V2.T.shape
        # A = torch.hstack([V2.T, torch.randn((n, n-m))])
        # C2 = self.gram_schmidt(A)[:, m:].T
        # C2 = torch.tensor(orth(A)[:, m:].T)
        V2 = V2.cpu()
        C2 = torch.tensor(null_space(V2).T)
        C2 = C2.to(self.device)

        # n, m = V3.T.shape
        # A = torch.hstack([V3.T, torch.randn((n, n-m))])
        # C3 = self.gram_schmidt(A)[:, m:].T
        # C3 = torch.tensor(orth(A)[:, m:].T)
        V3 = V3.cpu()
        C3 = torch.tensor(null_space(V3).T)
        C3 = C3.to(self.device)
        
        return B1, B2, B3, C1, C2, C3
       
    def normalize_params(self):
     
        print(self.U1.shape, self.U2.shape, self.U3.shape)
        print()
        print(self.V1.shape, self.V2.shape, self.V3.shape)

        self.U1.data.div_(torch.norm(self.U1, dim=0, keepdim=True))
        self.U2.data.div_(torch.norm(self.U2, dim=0, keepdim=True))
        self.U3.data.div_(torch.norm(self.U3, dim=0, keepdim=True))

        self.V1.data.div_(torch.norm(self.V1, dim=1, keepdim=True))
        self.V2.data.div_(torch.norm(self.V2, dim=1, keepdim=True))
        self.V3.data.div_(torch.norm(self.U3, dim=1, keepdim=True))

    def forward(self, x):
     
        if self.ortho:
            if self.first_layer:
                W1 = (self.B1 @ self.U1) @ (self.V1 @ self.C1)
            else:
                W1 = torch.zeros_like(self.O1)
            if self.second_layer:
                W2 = (self.B2 @ self.U2) @ (self.V2 @ self.C2)
            else:
                W2 = torch.zeros_like(self.O2)
            if self.third_layer:
                W3 = (self.B3 @ self.U3) @ (self.V3 @ self.C3)
            else:
                W3 = torch.zeros_like(self.O3)

        else:
            W1 = self.U1 @ self.V1
            W2 = self.U2 @ self.V2
            W3 = self.U3 @ self.V3

        # add the outer products from the previous ranks to current outer products
        if self.O1 is not None:
     
            W1 = W1 + self.O1
            W2 = W2 + self.O2
            W3 = W3 + self.O3
    
        x = self.relu(torch.einsum("ij, bj->bi", W1, x) + self.b1)
        x = self.relu(torch.einsum("ij, bj->bi", W2,x) + self.b2)
        return torch.einsum("ij, bj->bi", W3, x) + self.b3

    def get_matrices(self):

        if self.ortho:

            # construct current weight matrices from singular vectors
            if self.first_layer:
                W1 = (self.B1 @ self.U1) @ (self.V1 @ self.C1)
            else:
                W1 = torch.zeros_like(self.O1)
            if self.second_layer:
                W2 = (self.B2 @ self.U2) @ (self.V2 @ self.C2)
            else:
                W2 = torch.zeros_like(self.O2)
            if self.third_layer:
                W3 = (self.B3 @ self.U3) @ (self.V3 @ self.C3)
            else:
                W3 = torch.zeros_like(self.O3)


            # add outer products from previous ranks
            W1 += self.O1
            W2 += self.O2
            W3 += self.O3

            # construct the current singular vectors
            if self.first_layer:
                U1 = (self.B1 @ self.U1).detach().requires_grad_(False)
                V1 = (self.V1 @ self.C1).detach().requires_grad_(False)
            else:
                U1 = None
                V1 = None

            if self.second_layer:
                U2 = (self.B2 @ self.U2).detach().requires_grad_(False)
                V2 = (self.V2 @ self.C2).detach().requires_grad_(False)
            else:
                U2 = None
                V2 = None

            if self.third_layer:
                U3 = (self.B3 @ self.U3).detach().requires_grad_(False)
                V3 = (self.V3 @ self.C3).detach().requires_grad_(False)
            else:
                U3 = None
                V3 = None

        else:

            # construct current weight matrices
            W1 = self.U1 @ self.V1
            W2 = self.U2 @ self.V2
            W3 = self.U3 @ self.V3

            # collect current singular vectors
            U1, U2, U3, V1, V2, V3 = self.U1.detach().requires_grad_(False),\
                self.U2.detach().requires_grad_(False), self.U3.detach().requires_grad_(False), self.V1.detach().requires_grad_(False),\
                self.V2.detach().requires_grad_(False), self.V3.detach().requires_grad_(False)


        # detach and return everything for use in the next rank
        return W1.detach().requires_grad_(False), W2.detach().requires_grad_(False), W3.detach().requires_grad_(False), \
                    U1, U2, U3, V1, V2, V3 
                    # self.b1.detach().requires_grad_(False), self.b2.detach().requires_grad_(False), self.b3.detach().requires_grad_(False)

    def get_parameter_norms(self):
        # construct current weight matrices from singular vectors
        if self.first_layer:
            layer_1_norms = torch.norm(self.U1) + torch.norm(self.V1)
        else:
            layer_1_norms = 0.0

        if self.second_layer:
            layer_2_norms = torch.norm(self.U2) + torch.norm(self.V2)
        else:
            layer_2_norms = 0.0
            
        if self.third_layer:
            layer_3_norms = torch.norm(self.U3) + torch.norm(self.V3)
        else:
            layer_3_norms = 0.0

        bias_norms = torch.norm(self.b1) + torch.norm(self.b2) + torch.norm(self.b3)

        return bias_norms + layer_1_norms + layer_2_norms + layer_3_norms



def train(model, num_epochs, train_loader, test_loader, lr, device):
    Loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scores = []
    grad_dict = {}
    count = 0

    for epoch in range(num_epochs):
        if epoch % 20 == 0:
            count += 1

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name in grad_dict.keys():
                        grad_dict[name] += torch.abs(param.grad)
                    else:
                        grad_dict[name] = torch.abs(param.grad)

            model.eval()
            test_loss = 0

            for batch, y in test_loader:  
                batch = batch.to(device)
                y = y.to(device)
                y = y.unsqueeze(-1)
                y_hat = model(batch)
       
                score = Loss(y, y_hat)              
                test_loss += score.item()

            avg_score = test_loss / len(test_loader)
            print("Validation Score: ", avg_score)
            scores.append(avg_score)
            model.train()

        for batch, y in train_loader:
            batch = batch.to(device)
            y = y.to(device)
            y = y.unsqueeze(-1)
          
            optimizer.zero_grad()
            y_hat = model(batch)
        
            loss = Loss(y, y_hat)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

    # for name, param in model.named_parameters():
    #     grad_dict[name] /= count

        
    return scores, grad_dict

def successive_train(model, num_epochs, train_loader, test_loader, num_updates=0, xs=None, y=None, lr_schedule=[1e-2] * 17, device="cuda"):
    # collect the singular vectors from each round of training
    U1, U2, U3, V1, V2, V3 = [], [], [], [], [], []
    all_scores = []
    successive_norms = []
    successive_grads = []

    for i in range(num_updates+ 1):
        
        # train the model and collecti outer products and singular vectors
        scores, grads = train(model, num_epochs[i], train_loader, test_loader, lr=lr_schedule[i], device=device)
        all_scores.extend(scores)
        O1, O2, O3, u1, u2, u3, v1, v2, v3 = model.get_matrices()
        successive_norms.append(model.get_parameter_norms())
        successive_grads.append(grads)
        
        if u1 is not None:
            U1.append(u1)

        if u2 is not None:
            U2.append(u2)
        
        if u3 is not None:
            U3.append(u3)

        if v1 is not None:
            V1.append(v1)
        
        if v2 is not None:
            V2.append(v2)
        
        if v3 is not None:
            V3.append(v3)

        if i < num_updates:
            print("Updating Model")
            b1, b2, b3 = model.b1, model.b2, model.b3
            model = NeuralNetwork(input_dims=8, hidden_dims=256, output_dims=1, O1=O1, O2=O2, O3=O3,  \
                                    U1=torch.hstack(U1), U2=torch.hstack(U2), U3=torch.hstack(U3), V1=torch.vstack(V1), \
                                    V2=torch.vstack(V2), V3=torch.vstack(V3), device=device)

            model.b1 = b1 
            model.b2 = b2 
            model.b3 = b3 
            model.to(device)

    # get the norms of each rank
    for i, norm in enumerate(successive_norms):
        
        print(f"Rank {i + 1} parameter norms: ", norm)

    return all_scores, successive_grads
    
class CreateDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, i):
        return self.data[i], self.labels[i]

    def __len__(self):
        return len(self.data)



if __name__ == "__main__":

    if torch.cuda.is_available():
        print("It's cuda time")
        device="cuda"
    else:
        device = "cpu"
    # iris = load_iris()
    c_housing = fetch_california_housing()
    data = c_housing["data"]
    targets = torch.tensor(c_housing["target"]).type(torch.float)

    # normalize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = torch.tensor(data).type(torch.float)

    # noise = 0.05 * torch.randn(data.shape)
    # data += noise

    # create train test split
    train_data, test_data, train_labels, test_labels = train_test_split(data, targets, train_size=0.7)

    train_dataset = CreateDataset(data=train_data, labels=train_labels)
    test_dataset = CreateDataset(data=test_data, labels=test_labels)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, pin_memory=True)

    
    # train a standard Neural Network
    # model = NN(input_dims=8, hidden_dims=8, output_dims=1).to(device)
    # nn_scores, grads = train(model, num_epochs=1000, train_loader=train_loader, test_loader=test_loader, device=device, lr=1e-2)

    # train a neural network successively
    model = NeuralNetwork(input_dims=8, hidden_dims=32, output_dims=1, device=device).to(device)
    successive_scores, successive_grads = successive_train(model, num_epochs=[500] * 15, train_loader=train_loader, test_loader=test_loader, num_updates=14, device=device)
    y_max = torch.max((torch.max(successive_grads[0]["U2"], torch.max(successive_grads[0]["V2"])))).cpu()

    for i in range(15):
        plot_grads(f"Rank {i+1} Gradient Norms", successive_grads[i], y_max = y_max)
    nn_scores = successive_scores

    plot_accs("Validation Loss Comparison", nn_scores, successive_scores, 20)
    plt.show()
    np.save("successive_scores.npy", successive_scores)
    np.save("nn_scores.npy", nn_scores)
    # create_2D_losscape(model, train_loader)







    def model_map(U1, V1, U2, V2, U3, V3, b1, b2, b3):
        Loss = nn.MSELoss()

        new_model = DummyNeuralNetwork(input_dims=8, hidden_dims=64, output_dims=1, rank=1).to('cuda')
        new_model.train()
        new_model.U1 = U1
        new_model.V1 = V1
        new_model.U2 = U2 
        new_model.V2 = V2
        new_model.U3 = U3
        new_model.V3 = V3
        new_model.b1 = b1
        new_model.b2 = b2
        new_model.b3 = b3
        
        total_cost = 0

        for batch, y in train_loader:
            batch = batch.to("cuda")
            y = y.to("cuda")
            y = y.unsqueeze(-1)
            y_hat = new_model(batch)
            cost = Loss(y, y_hat)
            total_cost += cost
        print(total_cost)
        return total_cost

    # params = (model.U1 ,model.V1, model.U2, model.V2, model.U3, model.V3, model.b1, model.b2, model.b3)
    # # print(model_map(*params))
    # J = torch.autograd.functional.jacobian(model_map, params, create_graph=True, strict=True)
    # print(J)
    
