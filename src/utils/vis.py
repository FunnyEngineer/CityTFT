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