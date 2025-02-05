from Standard_Imports import *

num_trials = 3
dim = 5
x_offset = np.random.randint(0, 4, num_trials)

y_offset = np.random.randint(0, 4, num_trials)

pad = 2

coords = (np.repeat(np.arange(num_trials), (dim) ** 2) * (4 * (dim + 1)) ** 2,)

coords += (
    (
        np.tile(np.repeat(np.arange(dim) * 4 + pad, dim), num_trials)
        + np.repeat(y_offset, dim**2)
    )
    * 4
    * (dim + 1)
)

coords += np.tile(np.tile(np.arange(dim) * 4 + pad, dim), num_trials) + np.repeat(
    x_offset, dim**2
)


move_choices = [
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
]
random_moves = np.array(move_choices)[
    np.random.choice(len(move_choices), dim * dim * num_trials)
]
move_coords = coords + random_moves[:, 0] * 4 * (dim + 1) + random_moves[:, 1]


data = np.zeros((num_trials, 4 * (dim + 1), 4 * (dim + 1)))
trial, y, x = (
    coords // ((4 * (dim + 1)) ** 2),
    (coords % ((4 * (dim + 1)) ** 2)) // (4 * (dim + 1)),
    (coords % ((4 * (dim + 1)) ** 2)) % (4 * (dim + 1)),
)

data[trial, y, x] = 0.5

trial2, y2, x2 = (
    move_coords // ((4 * (dim + 1)) ** 2),
    (move_coords % ((4 * (dim + 1)) ** 2)) // (4 * (dim + 1)),
    (move_coords % ((4 * (dim + 1)) ** 2)) % (4 * (dim + 1)),
)
data[trial2, y2, x2] = 1


fig, axs = plt.subplots(2, 1)
axs[0].matshow(data[0], cmap="plasma")

axs[1].matshow(data[1], cmap="plasma")
plt.show()
