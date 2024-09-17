# Author: Mian Qin
# Date Created: 7/2/24
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from MDAnalysis.auxiliary.XVG import XVGReader
import MDAnalysis as mda
import networkx as nx


def read_xvg(file_path: str | Path) -> pd.DataFrame:
    with XVGReader(str(file_path)) as aus:
        data = aus._auxdata_values
        df = pd.DataFrame(data)
    return df


def read_solid_like_atoms(file_path: Path) -> dict[str, list[str]]:
    solid_like_atoms_dict = {}
    with open(file_path) as file:
        for line in file:
            line = line.strip().split()
            if len(line) == 0:  # End of file
                break
            t = float(line[0])
            indices = [str(x) for x in line[1:]]
            solid_like_atoms_dict[f"{t:.1f}"] = indices
    return solid_like_atoms_dict


def combine_index(index_dict_list: list[dict]) -> dict[str, list[str]]:
    combined_index_dict = {}
    t_all = set()
    for index_dict in index_dict_list:
        t_all.update(index_dict.keys())
    t_all = sorted(t_all, key=lambda x: float(x))
    for t in t_all:
        indices = set()
        for index_dict in index_dict_list:
            indices.update(index_dict.get(t, []))
        indices = sorted(indices, key=lambda x: int(x))
        combined_index_dict[t] = indices
    return combined_index_dict


def write_index_dict(index_dict: dict[str, list[str]], file_path: Path):
    with open(file_path, 'w') as file:
        for t, indices in index_dict.items():
            file.write(f"{t} {' '.join(indices)}\n")
    print(f"Write indices to {file_path.resolve()}")


def wrap_position_vector_array(position_vector_array: np.ndarray, box_size: np.ndarray):
    half_box_size = box_size / 2
    lt = position_vector_array < -half_box_size
    gt = position_vector_array > half_box_size
    position_vector_array += lt * box_size
    position_vector_array -= gt * box_size
    return position_vector_array


def generate_graph(pos_atoms: np.ndarray, ids: np.ndarray, box_size, type2indices: dict[str, list[int]],
                   threshold_edge=3.5) -> nx.Graph:
    """

    :param pos_atoms: Positions of atoms. Shape: (N, 3)
    :param ids: Indices of atoms. Shape: (N)
    :param box_size:
    :param type2indices: Ice/Water/Surface
    :param threshold_edge: In Angstrom
    :return:
    """
    position_vector_array = np.expand_dims(pos_atoms, axis=0) - np.expand_dims(pos_atoms, axis=1)
    wrapped_position_vector_array = wrap_position_vector_array(position_vector_array, box_size)
    distance_array = np.linalg.norm(wrapped_position_vector_array, axis=2)

    i_array, j_array = np.where(distance_array < threshold_edge)
    non_duplicate_index = np.where(i_array < j_array)
    i_array = i_array[non_duplicate_index]
    j_array = j_array[non_duplicate_index]

    i_ids_array = ids[i_array]
    j_ids_array = ids[j_array]

    G = nx.Graph()
    for t, indices in type2indices.items():
        G.add_nodes_from(indices, type=t)
    G.add_edges_from(list(zip(i_ids_array, j_ids_array)))
    return G


def update_graph(G: nx.Graph, type2indices: dict[str, Iterable[int]]):
    for t, indices in type2indices.items():
        for index in indices:
            G.nodes[index]["type"] = t
    return G


def main():
    ...


if __name__ == "__main__":
    main()
