"""Use the `SOM2D` class from `som.py` to cluster the RGB color space."""

import torch
from matplotlib import pyplot as plt

from som import SOM2D


def rgb_to_hex(rgb_values):
    for v in rgb_values:
        c = '%02x%02x%02x' % tuple(v)
        yield c


def hex_to_rgb(hex_values):
    for v in hex_values:
        v = v.lstrip('#')
        lv = len(v)
        c = [int(v[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]
        yield c


def plot_and_show(title):
    plt.imshow(colorsom.weights.reshape(*gridshape, 3))
    plt.title(title)
    for i, iloc in enumerate(colorsom.competition(data)):
        plt.text(iloc[1],
                 iloc[0],
                 cnames[i],
                 ha='center',
                 va='center',
                 bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.show()


hex_colors = [
    '000000', '0000ff', '00007f', '1f86ff', '5466aa', '997fff', '00ff00',
    'ff0000', '00ffff', 'ff00ff', 'ffff00', 'ffffff', '545454', '7f7f7f',
    'a8a8a8'
]
cnames = [
    'black', 'blue', 'darkblue', 'skyblue', 'greyblue', 'lilac', 'green',
    'red', 'cyan', 'violet', 'yellow', 'white', 'darkgrey', 'mediumgrey',
    'lightgrey'
]

# convert the hex to torch tensor of rgb values
colors = list(hex_to_rgb(hex_colors))
data = torch.tensor(colors) / 255.0

# instantiate the model
gridshape = (20, 30)
colorsom = SOM2D(gridshape, 3)

# display the unorganized map
plot_and_show('Unorganized grid')

# unsupervised training
colorsom.fit(data, epochs=100, verbose=False)

# display the organized map
plot_and_show('ColorSOMe (see what I did there?)')
