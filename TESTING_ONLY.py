from Standard_Imports import *

num_trials = 10
dimension = 5

start_tile = np.zeros((num_trials, 4, 4))
start_indices = np.indices(start_tile.shape)[:, :, 0, 0]

random_startpoints = np.random.randint(0, 4, (2, num_trials))

start_indices[1:, :] = random_startpoints

start_tile[tuple(start_indices)] = 1
start_mask = np.tile(start_tile, (dimension, dimension))
pad_width = ((0, 0), (2, 2), (2, 2))
start_mask = np.pad(start_mask, pad_width=pad_width, mode="constant", constant_values=0)

move_choices = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]

all_start_indices = np.indices(start_mask.shape)[:, start_mask.astype(bool)]
num_moves = all_start_indices.shape[1]


moves = np.array(move_choices)[np.random.choice(len(move_choices), num_moves)]
moves = moves.T


all_move_indices = all_start_indices.copy()
all_move_indices[1:] += moves

move_mask = np.zeros_like(start_mask)
move_mask[tuple(all_move_indices)] = 0.5

figure = plt.figure()
axes_h = figure.add_subplot(121)
axes_phi = figure.add_subplot(122)

start_plot = axes_h.matshow(start_mask[1], cmap="plasma")
move_plot = axes_phi.matshow(move_mask[2] + start_mask[2], cmap="plasma")


plt.show()
