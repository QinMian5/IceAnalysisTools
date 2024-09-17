# Author: Mian Qin
# Date Created: 7/2/24
from pathlib import Path

from utils import read_solid_like_atoms

_filename_hex_ice = "Hex.index"
_filename_cubic_ice = "Cubic.index"
_filename_int_ice = "IntIce.index"


def combine_indices(root_dir: Path):
    hex_ice_index_dict: dict[str, list[str]] = read_solid_like_atoms(root_dir / _filename_hex_ice)
    cubic_ice_index_dict: dict[str, list[str]] = read_solid_like_atoms(root_dir / _filename_cubic_ice)
    int_ice_index_dict: dict[str, list[str]] = read_solid_like_atoms(root_dir / _filename_int_ice)
    ice_index_dict: dict[str, list[str]] = {}
    for t in hex_ice_index_dict:
        hex_ice_index = hex_ice_index_dict[t]
        cubic_ice_index = cubic_ice_index_dict[t]
        int_ice_index = int_ice_index_dict[t]
        ice_index = hex_ice_index + cubic_ice_index + int_ice_index
        assert len(set(ice_index)) == len(ice_index), "Duplicate indices"
        ice_index.sort(key=lambda x: int(x))
        ice_index_dict[t] = ice_index
    return ice_index_dict


def main():
    ...


if __name__ == "__main__":
    main()
