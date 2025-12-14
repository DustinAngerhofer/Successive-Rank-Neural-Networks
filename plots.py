from losscape.create_landscape import create_2D_losscape
from matplotlib import pyplot as plt
import numpy as np

def plot_landscape(model, train_loader, ouput_vtp=True):
    create_2D_losscape(model, train_loader, ouput_vtp)

def plot_accs(title, acc1, acc2, epochs):
    plt.title(title)
    plt.semilogy(acc1, label='NN')
    plt.semilogy(acc2, label='Successive NN')
    plt.xlabel(f'{epochs} Epochs')
    plt.ylabel('Validation Loss')
    plt.grid()
    plt.legend()



def plot_grads(title, grad_dict, y_max):
    group_names = []
    groups = []
    indices = [-1]
    # try:
    #     U1 = grad_dict["U1"].squeeze(-1).detach().cpu().numpy()
    #     groups.append(U1)
    #     group_names.append("U1")
    #     indices.append(len(np.hstack(groups)) - 1)

    #     V1 = grad_dict["V1"].squeeze(0).detach().cpu().numpy()
    #     groups.append(V1)
    #     indices.append(len(np.hstack(groups)) - 1)
    #     group_names.append("V1")
    # except:
    #     print("No U1, V1")

    try:
        U2 = grad_dict["U2"].squeeze(-1).detach().cpu().numpy()
        groups.append(U2)
        indices.append(len(np.hstack(groups)) - 1)
        group_names.append("U2")

        V2 = grad_dict["V2"].squeeze(0).detach().cpu().numpy()
        groups.append(V2)
        indices.append(len(np.hstack(groups)) - 1)
        group_names.append("V2")
    except:
        print("No U2, V2")

    # try:
    #     U3 = grad_dict["U3"].squeeze(-1).detach().cpu().numpy()
    #     groups.append(U3)
    #     indices.append(len(np.hstack(groups)) - 1)
    #     group_names.append("U3")

    #     V3 = grad_dict["V3"].squeeze(0).detach().cpu().numpy()
    #     groups.append(V3)
    #     indices.append(len(np.hstack(groups)) - 1)
    #     group_names.append("V3")

    # except:
    #     print("No U3, V3")

    
    # try:
    #     b1 = grad_dict["b1"].detach().cpu().numpy()
    #     groups.append(b1)
    #     indices.append(len(np.hstack(groups)) - 1)
    #     group_names.append("b1")
    # except:
    #     print("No b1")

    # try:
    #     b2 = grad_dict["b2"].detach().cpu().numpy()
    #     groups.append(b2)
    #     indices.append(len(np.hstack(groups)) - 1)
    #     group_names.append("b2")
    # except:
    #     print("No b2")

    # try:
    #     b3 = grad_dict["b3"].detach().cpu().numpy()
    #     groups.append(b3)
    #     indices.append(len(np.hstack(groups)) - 1)
    #     group_names.append("b3")

    # except:
    #     print("No b3")

    colors = np.array(["C2", "C3", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
    colors_inds = np.hstack([(i-1) * np.ones(indices[i] - indices[i-1]) for i in range(1, len(indices))]).astype(int)

    colors = colors[colors_inds]

    y = np.hstack(groups)
    x = np.arange(len(y))

    fig, ax = plt.subplots()

    # fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

    # extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
    # im = ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
    # ax.set_yticks([])
    # ax.set_xlim(extent[0], extent[1])

    # ax2.plot(x,y)
    # cbar = fig.colorbar(im, ax=ax)
    # plt.tight_layout()
    # plt.show()

    ax.bar(x, y, color=colors)
    ax.legend(np.array(ax.patches)[indices[1:]], group_names)
    ax.set_ylabel("Average Gradient Norms")
    ax.set_xlabel("Parameter Groups")
    # ax.vlines(x=[len(U1) - 1, len(U1) + len(V1)-1], ls="--", ymin=-np.abs(y).mean(), ymax=np.abs(y).mean(), color="k")
    # ax.set_xticklabels(labels)
    # plt.grid()
    ax.set_title(title)
    ax.set_ylim([0, y_max])
    plt.grid()
    plt.savefig(title + ".png")
    plt.show()



