# Successive-Rank-Neural-Networks

<img width="2000" height="1125" alt="580_SuccessiveNN-1" src="https://github.com/user-attachments/assets/cb061689-52b9-4490-907d-2e61581d3566" />

What we have done here is effectively Low-Rank Adapdation (LoRA), but without a pretrained model. We train a feed-forward neural network, but with the weight matrices initially restricted to rank 1. Once the model has converged, we freeze the current weights and add a new trainable rank. The final weights will thus be sums of rank 1 matrices, similar to a Singular Value Decomposition (SVD). The analogous "singular vectors", are constrained to be orthogonal, as they are in an SVD. The parallels to the SVD do not end here--we find that the model learns the most important features first, w.r.t. the loss, and is refined with each rank update with diminishing returns. We find that models trained in this way avoid over-fitting where a regular neural network does. See validation loss plot below.


<img width="640" height="480" alt="successive_vs_nn_normed_2" src="https://github.com/user-attachments/assets/803ea032-b4a2-4493-ad24-0894452a66ee" />
Notice the step-like nature of the successive loss. The model quickly converges to the optimal rank-1 update, after which the loss plateaus. The small spike in loss immediately after each plateau is the result of injecting new learnable parameters at the new rank-1 update. Notice further how the regular neural network overfits on the training data.
