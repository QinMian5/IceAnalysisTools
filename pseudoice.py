# Author: Mian Qin
# Date Created: 7/2/24
from pathlib import Path
from tqdm import tqdm
import json
import pickle
import os
import sys
import inspect

import numpy as np
from numpy.typing import NDArray
import MDAnalysis as mda
import matplotlib.pyplot as plt
import networkx as nx
import scipy

from qbar6 import classify
from chillplus import combine_indices
from interface import generate_grid, calc_mean_interface_in_t_range, calc_instantaneous_interface
from utils import read_solid_like_atoms, write_index_dict, generate_graph, update_graph, combine_index

_filename_job_params = "job_params.json"
_filename_op = "order_parameters.dat"
_filename_pp = "post_processing.dat"
_filename_job = "job.sh"
_filename_job_post = "job_post.sh"
_filename_op_out = "op.out"
_filename_trj = "trajout.xtc"
_filename_index = "solid_like_atoms.index"
_filename_index2 = "solid_like_atoms_with_PI.index"
_filename_index3 = "solid_like_atoms_chillplus.index"
_filename_index4 = "solid_like_atoms_corrected.index"
_filename_index5 = "solid_like_atoms_chillplus_water.index"
_filename_index_new_ice = "new_ice.index"
_filename_index_new_water = "new_water.index"
_folder_name_pp2 = "post_processing_with_PI"
_folder_name_pp3 = "post_processing_chillplus"
_path_data_dir = Path("./data")

run_env = os.environ.get("RUN_ENVIRONMENT")
if run_env == "wsl":
    home_path = Path("/home/qinmian")
elif run_env == "chestnut":
    home_path = Path("/home/mianqin")
elif run_env == "mianqin_PC":
    home_path = Path("//wsl.localhost/Ubuntu-22.04/home/qinmian")
else:
    raise RuntimeError(f"Unknown environment: {run_env}")

def _get_root_dir(rho, process) -> Path:
    root_dir = home_path / f"data/gromacs/pseudoice/data/{rho}/prd/{process}"
    return root_dir


def _load_params(rho, process):
    data_dir = _get_root_dir(rho, process)
    with open(data_dir / _filename_job_params) as file:
        job_params = json.load(file)
    return job_params


def _filter_solid_like_atoms(solid_like_atoms_dict: dict):
    for k, v, in solid_like_atoms_dict.items():
        for i in range(len(v)):
            if int(v[i]) > 11892:
                solid_like_atoms_dict[k] = v[:i]
                break
    return solid_like_atoms_dict


def _correct_surface_water2ice(G: nx.Graph, max_iter=20) -> np.ndarray:
    _type_to_index = {"ice": 0, "water": 1, "surface": 2, "maybe_ice": 3}

    new_ice_array = np.zeros(len(G.nodes))
    node_type_array = np.zeros((len(G.nodes), len(_type_to_index)), dtype=int)
    for i, node in enumerate(G.nodes):
        node_type = G.nodes[node]["type"]
        node_type_array[i][_type_to_index[node_type]] = 1

    adj_matrix_first_shell = nx.to_scipy_sparse_array(G)
    adj_matrix2 = adj_matrix_first_shell @ adj_matrix_first_shell
    adj_matrix_two_shell = adj_matrix_first_shell + adj_matrix2
    adj_matrix_two_shell.setdiag(0)  # Exclude self
    adj_matrix_two_shell.data = np.minimum(adj_matrix_two_shell.data, 1)
    adj_matrix_two_shell.eliminate_zeros()
    is_surface = node_type_array[:, _type_to_index["surface"]]
    n_surface_two_shell = adj_matrix_two_shell @ is_surface
    cond_surface = n_surface_two_shell >= 1
    for i in range(max_iter):
        # adj_matrix_second_shell = adj_matrix_two_shell - adj_matrix_first_shell
        # adj_matrix_second_shell.eliminate_zeros()
        is_ice = node_type_array[:, _type_to_index["ice"]]
        is_water = node_type_array[:, _type_to_index["water"]]
        n_ice_two_shell = adj_matrix_two_shell @ is_ice
        n_water_two_shell = adj_matrix_two_shell @ is_water
        n_mobile_two_shell = n_ice_two_shell + n_water_two_shell
        # info = np.stack((is_ice, is_water, is_surface, n_ice_two_shell, n_water_two_shell, n_surface_two_shell), axis=1)
        # info_water = info[info[:, 1] == 1]

        mobile_ice_percentage_two_shell = n_ice_two_shell / np.maximum(n_mobile_two_shell, 1)
        cond_maybe_ice = cond_surface & (mobile_ice_percentage_two_shell > 0.25)

        maybe_ice = is_water & cond_maybe_ice
        node_type_array[:, _type_to_index["maybe_ice"]] = maybe_ice
        node_type_array[:, _type_to_index["water"]] -= maybe_ice
        is_water = node_type_array[:, _type_to_index["water"]]
        is_maybe_ice = node_type_array[:, _type_to_index["maybe_ice"]]
        n_water_two_shell = adj_matrix_two_shell @ is_water
        # n_water_first_shell = adj_matrix_first_shell @ is_water

        mobile_water_percentage_two_shell = n_water_two_shell / np.maximum(n_mobile_two_shell, 1)
        cond_maybe_ice_to_ice = cond_surface & (mobile_water_percentage_two_shell < 0.25)
        # info_mobile_water = mobile_water_percentage_two_shell[is_maybe_ice.astype(bool)]
        maybe_ice_to_ice = is_maybe_ice & cond_maybe_ice_to_ice
        if maybe_ice_to_ice.sum() == 0:
            break
        maybe_ice_to_water = is_maybe_ice & ~maybe_ice_to_ice
        node_type_array[:, _type_to_index["ice"]] += maybe_ice_to_ice
        node_type_array[:, _type_to_index["water"]] += maybe_ice_to_water
        node_type_array[:, _type_to_index["maybe_ice"]] = 0
        new_ice_array += maybe_ice_to_ice
    else:
        print(f"{inspect.currentframe().f_code.co_name}: Failed to converge after {max_iter} iterations.")
    node_index_array = np.array(G.nodes())
    water_to_ice_array = node_index_array[new_ice_array.astype(bool)]
    return water_to_ice_array


def _correct_bulk(G: nx.Graph, max_iter=10) -> tuple[np.ndarray, np.ndarray]:
    _type_to_index = {"ice": 0, "water": 1, "surface": 2}

    new_ice_array = np.zeros(len(G.nodes))
    new_water_array = np.zeros(len(G.nodes))
    node_type_array = np.zeros((len(G.nodes), len(_type_to_index)), dtype=int)
    for i, node in enumerate(G.nodes):
        node_type = G.nodes[node]["type"]
        node_type_array[i][_type_to_index[node_type]] = 1

    adj_matrix_first_shell = nx.to_scipy_sparse_array(G)
    adj_matrix2 = adj_matrix_first_shell @ adj_matrix_first_shell
    adj_matrix_two_shell = adj_matrix_first_shell + adj_matrix2
    adj_matrix_two_shell.setdiag(0)  # Exclude self
    adj_matrix_two_shell.data = np.minimum(adj_matrix_two_shell.data, 1)
    adj_matrix_two_shell.eliminate_zeros()
    is_surface = node_type_array[:, _type_to_index["surface"]]
    n_surface_first_shell = adj_matrix_first_shell @ is_surface
    cond_surface = n_surface_first_shell >= 1
    for i in range(max_iter):
        # adj_matrix_second_shell = adj_matrix_two_shell - adj_matrix_first_shell
        # adj_matrix_second_shell.eliminate_zeros()
        is_ice = node_type_array[:, _type_to_index["ice"]]
        is_water = node_type_array[:, _type_to_index["water"]]
        n_ice_two_shell = adj_matrix_two_shell @ is_ice
        n_water_two_shell = adj_matrix_two_shell @ is_water
        n_mobile_two_shell = n_ice_two_shell + n_water_two_shell
        cond_bulk = ~cond_surface & (n_mobile_two_shell >= 8)

        water_to_ice = is_water & cond_bulk & (n_water_two_shell <= 2)
        ice_to_water = is_ice & cond_bulk & (n_ice_two_shell <= 2)

        if water_to_ice.sum() + ice_to_water.sum() == 0:
            break
        node_type_array[:, _type_to_index["ice"]] += water_to_ice
        node_type_array[:, _type_to_index["water"]] -= water_to_ice
        node_type_array[:, _type_to_index["water"]] += ice_to_water
        node_type_array[:, _type_to_index["ice"]] -= ice_to_water
        new_ice_array += water_to_ice
        new_water_array += ice_to_water
    else:
        print(f"{inspect.currentframe().f_code.co_name}: Failed to converge after {max_iter} iterations.")
    node_index_array = np.array(G.nodes())
    water_to_ice_array = node_index_array[new_ice_array.astype(bool)]
    ice_to_water_array = node_index_array[new_water_array.astype(bool)]
    return water_to_ice_array, ice_to_water_array


def correct_ice_index(path_to_conf: Path, path_to_traj: Path, path_to_index: Path, t_range: tuple) -> tuple[dict, dict]:
    u = mda.Universe(path_to_conf, path_to_traj)
    ice_index_dict = _filter_solid_like_atoms(read_solid_like_atoms(path_to_index))
    new_ice_index_dict = {}
    new_water_index_dict = {}
    for ts in tqdm(u.trajectory, desc="Frame"):
        # if f"{ts.time:.1f}" != "4020.0":
        #     continue
        if ts.time < t_range[0] or ts.time > t_range[1]:
            new_ice_index_dict[f"{ts.time:.1f}"] = []
            new_water_index_dict[f"{ts.time:.1f}"] = []
            continue
        ice_index = ice_index_dict[f"{ts.time:.1f}"]
        O_atoms_all = u.select_atoms("name OW")
        if len(ice_index) == 0:
            O_atoms_water = O_atoms_all
            O_atoms_ice = O_atoms_all - O_atoms_water
        else:
            O_atoms_ice = u.select_atoms(f"bynum {' '.join(ice_index)}")
            O_atoms_water = O_atoms_all - O_atoms_ice
        O_atoms_surface = u.select_atoms("name O_PI")
        pos_ice = O_atoms_ice.positions
        pos_water = O_atoms_water.positions
        pos_surface = O_atoms_surface.positions
        box_size = u.dimensions[:3]
        pos_atoms = np.concatenate((pos_ice, pos_water, pos_surface))
        ids = np.concatenate((O_atoms_ice.ids, O_atoms_water.ids, O_atoms_surface.ids))
        type2indices = {"ice": O_atoms_ice.ids,
                        "water": O_atoms_water.ids,
                        "surface": O_atoms_surface.ids}
        G = generate_graph(pos_atoms, ids, box_size, type2indices)

        surface_water_to_ice_array = _correct_surface_water2ice(G)
        G = update_graph(G, {"ice": surface_water_to_ice_array})
        bulk_water_to_ice_array, bulk_ice_to_water_array = _correct_bulk(G)
        water_to_ice_array = np.concatenate((surface_water_to_ice_array, bulk_water_to_ice_array))
        ice_to_water_array = bulk_ice_to_water_array
        new_ice_index = [str(x) for x in water_to_ice_array]
        new_water_index = [str(x) for x in ice_to_water_array]

        # bulk_water_to_ice_array, bulk_ice_to_water_array = _correct_bulk(G)
        # new_ice_array = np.concatenate((surface_water_to_ice_array, bulk_water_to_ice_array), axis=0)
        # new_water_array = bulk_ice_to_water_array
        # new_ice_index = [str(x) for x in new_ice_array]
        # new_water_index = [str(x) for x in new_water_array]

        new_ice_index_dict[f"{ts.time:.1f}"] = new_ice_index
        new_water_index_dict[f"{ts.time:.1f}"] = new_water_index
    return new_ice_index_dict, new_water_index_dict


def main_mean_interface(rho, process):
    job_params = _load_params(rho, process)
    result_dir = _get_root_dir(rho, process) / "result"
    u = mda.Universe(result_dir / "conf.gro")
    x_max, y_max, z_max = u.dimensions[:3]
    x_range, y_range, z_range = (0, x_max), (0, y_max), (20, 65)
    pos_grid, scale, offset = generate_grid(x_range, y_range, z_range, n_x=50, n_y=50, n_z=40)
    for key, value in job_params.items():
        op = int(key.split("_")[1])
        print(key)
        u = mda.Universe(result_dir / "conf.gro", result_dir / key / "trajout.xtc")
        solid_like_atoms_dict = _filter_solid_like_atoms(read_solid_like_atoms(result_dir / key / _filename_index4))
        t_start = job_params[key]["RAMP_TIME"] + 200
        t_end = job_params[key]["RAMP_TIME"] + job_params[key]["PRD_TIME"]
        t_range = (t_start, t_end)
        nodes, faces = calc_mean_interface_in_t_range(u, solid_like_atoms_dict, pos_grid, scale, offset, t_range)
        with open(result_dir / key / "interface.pickle", "wb") as file:
            pickle.dump([nodes, faces], file)


def post_processing_chillplus(rho, process):
    job_params = _load_params(rho, process)
    root_dir = _get_root_dir(rho, process)
    for job_name in job_params:
        data_dir = root_dir / "data" / job_name / _folder_name_pp3
        ice_index_dict = combine_indices(data_dir)
        save_path = root_dir / "data" / job_name / _folder_name_pp3 / _filename_index
        write_index_dict(ice_index_dict, save_path)


def post_processing_with_PI(rho, process):
    job_params = _load_params(rho, process)
    root_dir = _get_root_dir(rho, process)
    for job_name in job_params:
        data_dir = root_dir / "data" / job_name / _folder_name_pp2
        solid_like_atoms_dict = _filter_solid_like_atoms(read_solid_like_atoms(data_dir / _filename_index))
        save_path = root_dir / "data" / job_name / _folder_name_pp2 / _filename_index2
        write_index_dict(solid_like_atoms_dict, save_path)


def main_correct_ice_index(rho, process):
    job_params = _load_params(rho, process)
    root_dir = _get_root_dir(rho, process)
    result_dir = root_dir / "result"
    for job_name in reversed(job_params):
        # if job_name != "op_1200":
        #     continue
        print(f"Current job: {job_name}")
        job_dir = result_dir / job_name
        path_to_conf = result_dir / "conf.gro"
        path_to_traj = job_dir / "trajout.xtc"
        path_to_index = job_dir / _filename_index3
        save_path = root_dir / "result" / job_name / _filename_index4
        new_ice_save_path = job_dir / _filename_index_new_ice
        new_water_save_path = job_dir / _filename_index_new_water
        t_start = job_params[job_name]["RAMP_TIME"] + 200
        t_end = job_params[job_name]["RAMP_TIME"] + job_params[job_name]["PRD_TIME"]
        t_range = (t_start, t_end)
        new_ice_index_dict, new_water_index_dict = correct_ice_index(path_to_conf, path_to_traj, path_to_index, t_range)
        write_index_dict(new_ice_index_dict, new_ice_save_path)
        write_index_dict(new_water_index_dict, new_water_save_path)

        ice_index_dict = _filter_solid_like_atoms(read_solid_like_atoms(path_to_index))
        combined_ice_dict = combine_index([ice_index_dict, new_ice_index_dict])
        write_index_dict(combined_ice_dict, save_path)


def _get_water_index(indices: list[str]):
    indices = indices.copy()
    indices.append("11893")
    water_index = []
    for i in range(len(indices) - 1):
        first = int(indices[i])
        second = int(indices[i + 1])
        for index in range(first + 4, second, 4):
            water_index.append(str(index))
    return water_index


def main_write_water_index(rho, process):
    job_params = _load_params(rho, process)
    root_dir = _get_root_dir(rho, process)
    result_dir = root_dir / "result"
    for job_name in job_params:
        # if job_name != "op_1200":
        #     continue
        print(f"Current job: {job_name}")
        job_dir = result_dir / job_name
        path_to_index = job_dir / _filename_index3
        water_index_path = job_dir / _filename_index5
        ice_index_dict = read_solid_like_atoms(path_to_index)
        water_index_dict = {}
        for t, indices in ice_index_dict.items():
            water_index = _get_water_index(indices)
            water_index_dict[t] = water_index
        write_index_dict(water_index_dict, water_index_path)


def main():
    process_list = ["icing_constant_ramp_rate"]
    for rho in [0.75]:
        for process in process_list:
            print(f"rho = {rho}, process = {process}")
            # post_processing_chillplus(rho, process)
            main_correct_ice_index(rho, process)
            main_write_water_index(rho, process)
            main_mean_interface(rho, process)


if __name__ == "__main__":
    main()
