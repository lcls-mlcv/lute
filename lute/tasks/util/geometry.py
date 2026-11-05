"""A set of utilities for managing detector geometry and conversions between formats.

Functions:
    from_crystfel(fname: str): Parse a CrystFEL geometry file.
"""

import os
from typing import Dict, List, Literal, Tuple, TypedDict, Union, cast

import numpy as np
import numpy.typing as npt

__author__ = "Gabriel Dorlhiac"


class CrystfelPanelGeometry(TypedDict):
    fs: Tuple[float, float, float]
    ss: Tuple[float, float, float]
    res: float
    corner_x: float
    corner_y: float
    coffset: float
    min_fs: int
    max_fs: int
    min_ss: int
    max_ss: int
    no_index: Literal[0, 1]


class CrystfelDetectorGeometry(TypedDict):
    panels: Dict[str, CrystfelPanelGeometry]
    rigid_groups: Dict[str, List[str]]
    rigid_group_collections: Dict[str, List[str]]


class PixelMaps(TypedDict):
    x_map: npt.NDArray[np.float64]
    y_map: npt.NDArray[np.float64]
    z_map: npt.NDArray[np.float64]
    radius_map: npt.NDArray[np.float64]
    phi_map: npt.NDArray[np.float64]


def parse_crystfel_geometry_file(file_path: Union[str, os.PathLike]):
    """Parse a CrystFEL geometry file.

    Read a text ".geom" file and return the dictionary of geometry components.

    Args:
    """
    detector: CrystfelDetectorGeometry = {
        "panels": {},
        "rigid_groups": {},
        "rigid_group_collections": {},
    }
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # Remove comments
            if line[0] == ";":
                continue
            if "=" not in line:
                continue

            # CrystFEL fmt: object = val
            obj_value = line.split("=")
            # object fmt: "panel/parameter"
            obj: List[str] = obj_value[0].split("/")  # May be len 1 or 2
            value: str = obj_value[1]

            if len(obj) == 1:  # e.g. rigid_group_quad ...
                if "collection" in obj[0].strip():
                    collection_name: str = obj[0].strip().split("_")[-1]
                    detector["rigid_group_collections"][
                        collection_name
                    ] = value.strip().split(",")
                else:
                    group_name = obj[0].strip().split("_")[-1]
                    detector["rigid_groups"][group_name] = value.strip().split(",")
            elif len(obj) == 2:  # e.g. p0a0/fs = ...
                pname = obj[0].strip()
                panel: CrystfelPanelGeometry
                if pname in detector["panels"]:
                    panel = detector["panels"][pname]
                else:
                    panel = {
                        "fs": (0, 0, 0),
                        "ss": (0, 0, 0),
                        "res": 10000,
                        "corner_x": 100,
                        "corner_y": 100,
                        "coffset": 0.0,
                        "min_fs": 0,
                        "max_fs": 123,
                        "min_ss": 0,
                        "max_ss": 123,
                        "no_index": 0,
                    }
                    detector["panels"][pname] = panel
                if "fs" in obj[1].strip()[-2:]:
                    if "max" in obj[1]:
                        panel["max_fs"] = int(value)
                    elif "min" in obj[1]:
                        panel["min_fs"] = int(value)
                    else:
                        strcoords = value.split()
                        # -1x -2y -3z
                        fcoords = (
                            float(strcoords[0].strip("x")),
                            float(strcoords[1].strip("y")),
                            float(strcoords[2].strip("z")) if len(strcoords) > 2 else 0,
                        )
                        panel["fs"] = fcoords
                elif "ss" in obj[1].strip()[-2:]:
                    if "max" in obj[1]:
                        panel["max_ss"] = int(value)
                    elif "min" in obj[1]:
                        panel["min_ss"] = int(value)
                    else:
                        strcoords = value.split()
                        # -1x -2y -3z
                        fcoords = (
                            float(strcoords[0].strip("x")),
                            float(strcoords[1].strip("y")),
                            float(strcoords[2].strip("z")) if len(strcoords) > 2 else 0,
                        )
                        panel["ss"] = fcoords
                elif "res" in obj[1].strip():
                    panel["res"] = float(value)
                elif "corner" in obj[1].strip():
                    if "x" in obj[1]:
                        panel["corner_x"] = float(value)
                    elif "y" in obj[1]:
                        panel["corner_y"] = float(value)
                elif "no_index" in obj[1]:
                    panel["no_index"] = cast(Literal[0, 1], int(value))
                elif "coffset" in obj[1]:
                    panel["coffset"] = float(value)

        return detector


def crystfel_to_pixel_map(geometry_desc: CrystfelDetectorGeometry) -> PixelMaps:
    """Compute pixel maps including pixel radius and phi from CrystFEL geometry.

    Args:
        geometry_desc (CrystfelDetectorGeometry): A description of the detector
            geometry parsed from a CyrstFEL geometry description.

    Returns:
        pixel_maps (PixelMaps): The per-pixel coordinate maps as well as radius
            and phi mappings.
    """
    max_fs_in_slab: int = np.array(
        [geometry_desc["panels"][k]["max_fs"] for k in geometry_desc["panels"]]
    ).max()
    max_ss_in_slab: int = np.array(
        [geometry_desc["panels"][k]["max_ss"] for k in geometry_desc["panels"]]
    ).max()

    x_map: npt.NDArray[np.float64] = np.zeros(
        shape=(max_ss_in_slab + 1, max_fs_in_slab + 1), dtype=np.float32
    )
    y_map: npt.NDArray[np.float64] = np.zeros(
        shape=(max_ss_in_slab + 1, max_fs_in_slab + 1), dtype=np.float32
    )
    z_map: npt.NDArray[np.float64] = np.zeros(
        shape=(max_ss_in_slab + 1, max_fs_in_slab + 1), dtype=np.float32
    )

    panel_name: str
    for panel_name in geometry_desc["panels"]:
        if "coffset" in geometry_desc["panels"][panel_name]:
            first_panel_camera_length: float = geometry_desc["panels"][panel_name][
                "coffset"
            ]
        else:
            first_panel_camera_length = 0.0

        ss_grid: npt.NDArray[np.int64]
        fs_grid: npt.NDArray[np.int64]
        ss_grid, fs_grid = np.meshgrid(
            np.arange(
                geometry_desc["panels"][panel_name]["max_ss"]
                - geometry_desc["panels"][panel_name]["min_ss"]
                + 1
            ),
            np.arange(
                geometry_desc["panels"][panel_name]["max_fs"]
                - geometry_desc["panels"][panel_name]["min_fs"]
                + 1
            ),
            indexing="ij",
        )

        fsx: float
        fsy: float
        # fsz: float
        fsx, fsy, _ = geometry_desc["panels"][panel_name]["fs"]

        ssx: float
        ssy: float
        # ssz: float
        ssx, ssy, _ = geometry_desc["panels"][panel_name]["ss"]
        y_panel: npt.NDArray[np.float64] = (
            ss_grid * ssy
            + fs_grid * fsy
            + geometry_desc["panels"][panel_name]["corner_y"]
        )
        x_panel: npt.NDArray[np.float64] = (
            ss_grid * ssx
            + fs_grid * fsx
            + geometry_desc["panels"][panel_name]["corner_x"]
        )
        x_map[
            geometry_desc["panels"][panel_name]["min_ss"] : geometry_desc["panels"][
                panel_name
            ]["max_ss"]
            + 1,
            geometry_desc["panels"][panel_name]["min_fs"] : geometry_desc["panels"][
                panel_name
            ]["max_fs"]
            + 1,
        ] = x_panel
        y_map[
            geometry_desc["panels"][panel_name]["min_ss"] : geometry_desc["panels"][
                panel_name
            ]["max_ss"]
            + 1,
            geometry_desc["panels"][panel_name]["min_fs"] : geometry_desc["panels"][
                panel_name
            ]["max_fs"]
            + 1,
        ] = y_panel
        z_map[
            geometry_desc["panels"][panel_name]["min_ss"] : geometry_desc["panels"][
                panel_name
            ]["max_ss"]
            + 1,
            geometry_desc["panels"][panel_name]["min_fs"] : geometry_desc["panels"][
                panel_name
            ]["max_fs"]
            + 1,
        ] = first_panel_camera_length

    r_map: npt.NDArray[np.float64] = np.sqrt(np.square(x_map) + np.square(y_map))
    phi_map: npt.NDArray[np.float64] = np.arctan2(y_map, x_map)

    return {
        "x_map": x_map,
        "y_map": y_map,
        "z_map": z_map,
        "radius_map": r_map,
        "phi_map": phi_map,
    }
