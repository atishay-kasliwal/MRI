import matplotlib.pyplot as plt

def plot_slice(volume, slice_index, title=None, cmap='gray'):
    plt.imshow(volume[:, :, slice_index], cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show() 