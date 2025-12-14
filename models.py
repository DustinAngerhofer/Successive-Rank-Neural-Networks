import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

import scipy
from scipy.linalg import orth, null_space
from scipy.optimize import minimize, least_squares
import math

class NN(nn.Module):
    """
    a standard feedforward neural network that optionally uses the same initialization as SuccessiveNN, 
    to control for weight initialization when comparing the two.
    """
    
    def __init__(self, input_dims, output_dims, hidden_dims):
        super().__init__()
        stdv1 = 1. / math.sqrt(hidden_dims)
        stdv2 = 1. / math.sqrt(hidden_dims)
        stdv3 = 1. / math.sqrt(output_dims)

        # self.fc1 = nn.Linear(input_dims, hidden_dims)
        # self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        # self.fc3 = nn.Linear(hidden_dims, output_dims)

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

class SuccessiveNN(nn.Module):
    """
    A feed-forward neural network whose learnable parameters are rank 1 matrices, decomposed as an outer product u_i @ v_i.T.
    Once training is completed, the learned weights are saved as the matrices O_i, to be used as frozen weights in the next iteration,
    so that we are indeed only training a rank-1 update to the previous iteration. This implementation is not optimal in terms of 
    memory and speed. Rather, it is a convenient proof of concept.
    """
    def __init__(self, input_dims, hidden_dims, output_dims, O1=None, O2=None, O3=None, U1=None, U2=None, U3=None, V1=None, V2=None, V3=None, device=None):
        super().__init__()
        """
        O1, O2, O3: Frozen weight matrices from previous rank, if applicable
        U1, U2, U3: Tensor of previous right singular vectors, later updated to be learnable coordinate vectors of an orthonormal basis
        V1, V2, V3: Tensor of previous left singular vectors, later updated to be learnable coordinate vectors of an orthonormal basis
        """

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
    
       # whether we use orthonormal update. This only applies when rank is greater than 1, i.e. self.Ui is not None
        self.ortho = False

        # take the previous singular vectors and construct change of basis into their nullspace. Becomes false
        # if rank of a given layer is already full.
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
        # construct an orthonormal basis
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
    """
    Experimental model for generating model maps
    """
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

