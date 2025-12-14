import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

import scipy
from scipy.linalg import orth, null_space
from scipy.optimize import minimize, least_squares

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
from models import NN, SuccessiveNN, DummyNeuralNetwork



def train(model, num_epochs, train_loader, test_loader, lr, device):
    """
    Train a standard NN
    """
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
    """
    Train an NN rank-successively
    """
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
            model = SuccessiveNN(input_dims=8, hidden_dims=256, output_dims=1, O1=O1, O2=O2, O3=O3,  \
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
        device="cuda"
    else:
        device = "cpu"

    ##### Load the data #####
    # iris = load_iris()
    c_housing = fetch_california_housing()
    data = c_housing["data"]
    targets = torch.tensor(c_housing["target"]).type(torch.float)

    ##### normalize the data #####
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = torch.tensor(data).type(torch.float)

    # noise = 0.05 * torch.randn(data.shape)
    # data += noise

    ##### create data loaders #####
    train_data, test_data, train_labels, test_labels = train_test_split(data, targets, train_size=0.7)

    train_dataset = CreateDataset(data=train_data, labels=train_labels)
    test_dataset = CreateDataset(data=test_data, labels=test_labels)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, pin_memory=True)

    ##### Train models #####
    # train a standard Neural Network
    # model = NN(input_dims=8, hidden_dims=8, output_dims=1).to(device)
    # nn_scores, grads = train(model, num_epochs=1000, train_loader=train_loader, test_loader=test_loader, device=device, lr=1e-2)

    # train a neural network successively
    model = SuccessiveNN(input_dims=8, hidden_dims=32, output_dims=1, device=device).to(device)
    successive_scores, successive_grads = successive_train(model, num_epochs=[500] * 15, train_loader=train_loader, test_loader=test_loader, num_updates=14, device=device)
    y_max = torch.max((torch.max(successive_grads[0]["U2"], torch.max(successive_grads[0]["V2"])))).cpu()


    ##### plot gradient norms and validation losses #####
    for i in range(15):
        plot_grads(f"Rank {i+1} Gradient Norms", successive_grads[i], y_max = y_max)
    nn_scores = successive_scores

    plot_accs("Validation Loss Comparison", nn_scores, successive_scores, 20)
    plt.show()
    np.save("successive_scores.npy", successive_scores)
    np.save("nn_scores.npy", nn_scores)
    # create_2D_losscape(model, train_loader)







    def model_map(U1, V1, U2, V2, U3, V3, b1, b2, b3):
        """
        Compute the loss of a given parameterization in such a way that allows for computation of the jacobian.
        """
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
    
