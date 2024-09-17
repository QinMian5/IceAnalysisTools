# Author: Mian Qin
# Date Created: 7/3/24
from pathlib import Path
import pickle
from tqdm import tqdm
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
import MDAnalysis as mda
from skimage import measure

from utils import read_solid_like_atoms, wrap_position_vector_array


def calc_scale_offset(x_range: tuple[float, float], n_x: int) \
        -> tuple[float, float]:
    start, end = x_range
    scale = (end - start) / (n_x - 1)
    offset = start
    return scale, offset


@lru_cache
def generate_grid(x_range: tuple[float, float], y_range: tuple[float, float], z_range: tuple[float, float],
                  n_x: int = 50, n_y: int = 50, n_z: int = 50):
    x_scale, x_offset = calc_scale_offset(x_range, n_x)
    y_scale, y_offset = calc_scale_offset(y_range, n_y)
    z_scale, z_offset = calc_scale_offset(z_range, n_z)
    scale = np.array([x_scale, y_scale, z_scale]).reshape(1, 3)
    offset = np.array([x_offset, y_offset, z_offset]).reshape(1, 3)
    x, y, z = np.linspace(*x_range, n_x), np.linspace(*y_range, n_y), np.linspace(*z_range, n_z)
    X, Y, Z = np.meshgrid(x, y, z)
    pos_grid = np.stack((X, Y, Z), axis=3)
    return pos_grid, scale, offset


def calc_concentration(pos_ice: np.ndarray, pos_grid: np.ndarray,
                       box_size: np.ndarray, ksi: float) -> np.ndarray:
    """

    :param pos_ice: Positions of O atoms in ice-like molecules. Shape: (N, 3)
    :param pos_grid: Positions of grids. Shape: (X, Y, Z, 3)
    :param box_size: Size of the box. Shape: (3)
    :param ksi:
    :return: Shape: (X, Y ,Z)
    """
    position_vector_array = np.expand_dims(pos_grid, axis=3) - np.expand_dims(pos_ice, axis=(0, 1, 2))
    wrapped_position_vector_array = wrap_position_vector_array(position_vector_array, box_size)
    distance_array = np.linalg.norm(wrapped_position_vector_array, axis=4)
    concentration = np.sum(1 / ((np.sqrt(2 * np.pi) * ksi) ** 3) * np.exp(-distance_array ** 2 / (2 * ksi ** 2)),
                           axis=3)
    return concentration


def generate_interface_from_concentration(concentration: np.ndarray, level=0.015) \
        -> tuple[np.ndarray, np.ndarray]:
    if level > concentration.max() or level < concentration.min():
        nodes, faces = np.zeros((0, 3)), np.zeros((0, 3))
    else:
        nodes, faces, normals, values = measure.marching_cubes(concentration, level=level)
    return nodes, faces


def calc_instantaneous_interface(pos_ice: np.ndarray, pos_grid: np.ndarray, box_size: np.ndarray, scale, offset,
                                 ksi=2.4):
    concentration = calc_concentration(pos_ice, pos_grid, box_size, ksi)
    nodes, faces = generate_interface_from_concentration(concentration)
    nodes = nodes * scale + offset
    return nodes, faces


def calc_mean_interface_in_t_range(u: mda.Universe, solid_like_atoms_dict: dict[str, list[str]], pos_grid, scale,
                                   offset, t_range: tuple[float, float], ksi=3.5 / 2) \
        -> tuple[NDArray[tuple[int, 3]], NDArray[tuple[int, 3]]]:
    acc_concentration = np.zeros(pos_grid.shape[:3])
    acc_n = 0
    for ts in tqdm(u.trajectory[::10]):
        t = ts.time
        if not t_range[0] <= t <= t_range[1]:
            continue
        solid_like_atoms_id = solid_like_atoms_dict[f"{t:.1f}"]
        if solid_like_atoms_id:
            solid_like_atoms = u.select_atoms(f"bynum {' '.join(solid_like_atoms_id)}")
            pos_ice = solid_like_atoms.positions
            concentration: NDArray[tuple[int, int, int]] = calc_concentration(pos_ice, pos_grid, u.dimensions[:3], ksi)
            acc_concentration += concentration
        acc_n += 1
    mean_concentration = acc_concentration / acc_n
    nodes, faces = generate_interface_from_concentration(mean_concentration)
    nodes = nodes * scale + offset
    return nodes, faces


def main():
    rho = 1.0
    data_dir = Path(f"/home/qinmian/data/gromacs/pseudoice/data/{rho}/prd/melting/result")
    key = "op_900"
    u = mda.Universe(data_dir / "conf.gro", data_dir / key / "trajout.xtc")
    x_max, y_max, z_max = u.dimensions[:3]
    solid_like_atoms_dict = read_solid_like_atoms(data_dir / key / "solid_like_atoms.index")
    x_range, y_range, z_range = (0, x_max), (0, y_max), (20, 65)
    pos_grid, scale, offset = generate_grid(x_range, y_range, z_range)
    nodes, faces = calc_mean_interface_in_t_range(u, solid_like_atoms_dict, pos_grid, scale, offset, (3000, 5000))
    with open("data.pickle", "wb") as file:
        pickle.dump([nodes, faces], file)


if __name__ == "__main__":
    main()
