from matplotlib import pyplot as plt

def plot_heat_cool(y, heat_hat, cool_hat):
    plt.figure(figsize=(20, 10))
    plt.plot(y[:, 0].cpu().numpy(), label="heat")
    plt.plot(y[:, 1].cpu().numpy(), label="cool")
    plt.plot(heat_hat.cpu().detach().numpy(), label="heat_hat")
    plt.plot(cool_hat.cpu().detach().numpy(), label="cool_hat")
    plt.legend()
    plt.show()