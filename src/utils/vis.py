from matplotlib import pyplot as plt

def plot_heat_cool(y, heat_hat, cool_hat, fig_path):
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    axs[0].plot(y[:, 0].cpu().numpy(), label="heat")
    axs[0].plot(heat_hat.cpu().detach().numpy(), label="heat_hat")
    axs[0].legend()
    axs[0].title.set_text('Heat')
    axs[1].plot(y[:, 1].cpu().numpy(), label="cool")
    axs[1].plot(cool_hat.cpu().detach().numpy(), label="cool_hat")
    axs[1].legend()
    axs[1].title.set_text('Cool')
    plt.savefig(fig_path)

def plot_heat_cool_seq_batch(y, heat_hat, cool_hat, fig_path):
    visual_num = 5
    fig, axs = plt.subplots(visual_num, 2, figsize=(15, 10))
    for i in range(visual_num):
        sin_y = y[i]
        axs[i, 0].plot(sin_y[:, 0].cpu().numpy(), label="heat")
        axs[i, 1].plot(sin_y[:, 1].cpu().numpy(), label="cool")
        for label in heat_hat:
            axs[i, 0].plot(heat_hat[label][i].cpu().detach().numpy(), label=label)
            axs[i, 1].plot(cool_hat[label][i].cpu().detach().numpy(), label=label)
        axs[i, 0].legend()
        axs[i, 1].legend()
    axs[0, 0].title.set_text('Heat')
    axs[0, 1].title.set_text('Cool')
    plt.tight_layout()
    plt.savefig(fig_path)
    # plt.show()