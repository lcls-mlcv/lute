from __future__ import annotations

# flake8: noqa: F821 -- __future__ annota.

import os
from typing import List, Optional, Tuple, Union, cast

import lmfit  # type: ignore
import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates  # type: ignore


def generate_concentric_sample_pts(
    peak_radii: Union[npt.NDArray[np.int64], List[int]],
    center: List[float],
    num_pts: int = 200,
) -> npt.NDArray[np.float64]:
    """Generate sample points along concentric circles.

    Args:
        peak_radii (npt.NDArray[np.int64]): Radii indices.

        center (List[float]): Center_x, Center_y for the concentric circles.

        num_pts (int): Number of sample points.

    Returns:
        coords (npt.NDArray[np.float64]): X/Y coordinates of sample points.
    """
    # X,Y labelling seems backwards
    cx: float = center[0]
    cy: float = center[1]
    # Reshape linear radii (peak indices) for broadcasting
    radii: np.ndarray[Tuple[int, int], np.dtype[np.int64]] = np.array(
        [peak_radii], dtype=np.int64
    ).reshape(-1, 1)
    theta: np.ndarray[Tuple[int], np.dtype[np.float64]] = np.linspace(
        0.0, 2 * np.pi, 200, dtype=np.float64
    )

    coords_x: np.ndarray[Tuple[int, int], np.dtype[np.float64]] = (
        radii * np.cos(theta) + cx
    ).astype(np.float64)
    coords_y: np.ndarray[Tuple[int, int], np.dtype[np.float64]] = (
        radii * np.sin(theta) + cy
    ).astype(np.float64)

    # Reshape for optimization routines
    coords: npt.NDArray[np.float64] = np.zeros((2, num_pts * len(peak_radii)))
    coords[1] = coords_x.reshape(-1)
    coords[0] = coords_y.reshape(-1)

    return coords


def geometry_optimize_residual(
    params: lmfit.Parameters, powder: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Target function for OptimizeAgBhGeometryExhaustive geometry fitting.

    Args:
        params (lmfit.Parameters): Parameters. [center_x, center_y, peaks...]
            Center values are floats. Peaks are integers.

        powder (npt.NDArray[np.float64]): Powder image.

    Returns:
        pixel_values (npt.NDArray[np.float64]): Residuals for fitting.
    """
    # Unpack the parameters
    params_l: List[Union[float, int]] = [val.value for _, val in params.items()]
    center_guess: List[float] = params_l[:2]
    # Indices are in radii units since they are for a radial profile
    indices: List[int] = cast(List[int], params_l[2:])
    coords: npt.NDArray[np.float64] = generate_concentric_sample_pts(
        indices, center_guess
    )

    # Use residual for fitting - difference between intensity in ring
    # and powder max
    pixel_values: npt.NDArray[np.float64] = map_coordinates(powder, coords)
    pixel_values -= np.max(powder)
    return pixel_values


def generate_geom_file(
    exp: str,
    run: int,
    ds: Union[psana.DataSource, psana.MPIDataSource],  # type: ignore
    det: psana.Detector,  # type: ignore
    input_file: str,
    output_file: str,
    det_dist: Optional[float] = None,
    pv_camera_length: Optional[str] = None,
) -> None:
    """Generate a Crystfel .geom file from either a psana or CrystFEL geometry.

    This function also sets the coffset field for each panel based
    on the estimated detector distance:
        coffset [m] = 1e-3 * (distance - clen)
    Supplying det_dist will override the value pulled from the deployed
    geometry for this run, which is used to compute the coffset parameter.

    Args:
        exp (str): Experiment name

        run (int): Run number

        ds (psana.DataSource | psana.MPIDataSource): psana DataSource object.

        det (psana.Detector): psana Detector object.

        input_file (str): Input .geom or .data file

        output_file (str): Output .geom file

        det_dist (Optional[float]): Estimated sample-detector distance in mm

        pv_camera_length (Optional[str]): PV associated with the camera length
    """
    from psgeom import camera, sensors  # type: ignore

    if input_file.split(".")[-1] == "data":
        geom = camera.CompoundAreaCamera.from_psana_file(input_file)
    elif input_file.split(".")[-1] == "geom":
        if (
            "epix10k2m" in str(det.name).lower()
            or "epix10ka2m" in str(det.name).lower()
        ):
            geom = camera.CompoundAreaCamera.from_crystfel_file(
                input_file, element_type=sensors.Epix10kaSegment
            )
        else:
            geom = camera.CompoundAreaCamera.from_crystfel_file(input_file)
    else:
        raise RuntimeError(
            "Provided a geometry file that did not end in .data or .geom! Cannot guess type."
        )

    if det_dist is None:
        det_dist = -1 * np.mean(det.coords_z(run)) / 1e3
    pv_cl: str
    if pv_camera_length is None:
        if (
            "epix10k2m" in str(det.name).lower()
            or "epix10ka2m" in str(det.name).lower()
        ):
            if "mfx" in str(det.name).lower():
                pv_cl = "MFX:ROB:CONT:POS:Z"
            else:
                raise RuntimeError(f"Cannot guess camera length PV for: {det.name}")
        elif "jungfrau4m" in str(det.name).lower():
            if "cxi" in str(det.name).lower():
                pv_cl = "CXI:DS1:MMS:06.RBV"
            else:
                raise RuntimeError(f"Cannot guess camera length PV for: {det.name}")
        elif "rayonix" in str(det.name).lower():
            if "mfx" in str(det.name).lower():
                pv_cl = "MFX:DET:MMS:04.RBV"
            else:
                raise RuntimeError(f"Cannot guess camera length PV for: {det.name}")
        else:
            raise RuntimeError(f"Cannot guess camera length PV for: {det.name}")
    else:
        pv_cl = pv_camera_length
    coffset: float = (det_dist - ds.env().epicsStore().value(pv_cl)) / 1000.0

    geom.to_crystfel_file(output_file, coffset=coffset)


def modify_crystfel_header(input_file, output_file):
    """Modify the header of a psgeom-generated Crystfel (.geom) file.

    This function performs the following modifications:
        1. Uncomment lines indicating the mask file and LCLS parameters.
        2. Add entries indicating the location of peaks in the cxi files.

    Args:
        input_file (str): Input .geom file generated by psgeom

        output_file (str): Output modified .geom file
    """
    outfile = open(output_file, "w")

    with open(input_file, "r") as infile:
        for line in infile.readlines():

            # uncomment by removing semicolon and space
            if line[0] == ";":
                if line.split()[1] in [
                    "clen",
                    "photon_energy",
                    "adu_per_eV",
                    "mask",
                    "mask_good",
                    "mask_bad",
                ]:
                    outfile.write(line[2:])
                else:
                    outfile.write(line)

            # add these header lines for latest crystfel
            elif "data = /entry_1/data_1/data" in line:
                outfile.write(line)
                outfile.write("\n")
                outfile.write("peak_list = /entry_1/result_1\n")
                outfile.write("peak_list_type = cxi\n")

            else:
                outfile.write(line)

    outfile.close()


def modify_crystfel_coffset_res(
    input_file: str, output_file: str, coffset: float, res: float
) -> None:
    """Overwrite coffset and res entries in a CrystFEL .geom file.

    This is a hack to fix Rayonix geom files generated using the wrong pixel size.

    Args:
        input_file (str): Input .geom file

        output_file (str): Output modified .geom file

        coffset (float): coffset (camera offset) value in meters.

        res (float): Pixel resolution in um
    """
    outfile = open(output_file, "w")

    with open(input_file, "r") as infile:
        for line in infile.readlines():

            if "coffset" in line:
                start = line.split("=")[0].strip(" ")
                outfile.write(f"{start} = {coffset}\n")

            elif "res = " in line:
                start = line.split("=")[0].strip(" ")
                outfile.write(f"{start} = {res}\n")

            else:
                outfile.write(line)

    outfile.close()


def deploy_geometry(
    out_dir: str,
    exp: str,
    run: int,
    ds: Union[psana.DataSource, psana.MPIDataSource],  # type: ignore
    det: psana.Detector,  # type: ignore
    pixel_size_mm: float,
    center: Tuple[float, float],
    distance: float,
    pv_camera_length: Optional[str] = None,
) -> None:
    """Write new geometry files (.geom and .data for CrystFEL and psana).

    Should be called with an optimized center and distance.

    Args:
        out_dir (str) Path to output directory.

        exp (str): Experiment name

        run (int): Run number

        ds (psana.DataSource | psana.MPIDataSource): psana DataSource object.

        det (psana.Detector): psana Detector object.

        pixel_size_mm (float): Detector pixel size in mm.

        center (Tuple[float, float]): Beam center for new geometry.

        distance (float): Detector distance for new geometry.

        pv_camera_length (str | None): PV associated with camera length.
    """
    import PSCalib  # type: ignore

    # retrieve original geometry
    geom: PSCalib.GeometryAcces.GeometryAccess = det.geometry(run)
    top: PSCalib.GeometryObject.GeometryObject = geom.get_top_geo()
    children: List[PSCalib.GeometryObject.GeometryObject] = top.get_list_of_children()
    child: PSCalib.GeometryObject.GeometryObject = children[0]

    pixel_size_um: float = pixel_size_mm * 1e3

    # determine and deploy shifts in x,y,z
    cy, cx = det.point_indexes(run, pxy_um=(0, 0), fract=True)
    dx = pixel_size_um * (center[0] - cx)  # convert from pixels to microns
    dy = pixel_size_um * (center[1] - cy)  # convert from pixels to microns
    dz = np.mean(-1 * det.coords_z(run)) - 1e3 * distance  # convert from mm to microns
    geom.move_geo(child.oname, 0, dx=-dy, dy=-dx, dz=dz)

    # write optimized geometry files
    psana_file_path: str = f"{out_dir}/r{run:04d}_end.data"
    geom.save_pars_in_file(psana_file_path)

    cfel_file_path: str = f"{out_dir}/r{run:04d}.geom"
    tmp_cfel_file_path: str = f"{cfel_file_path}.tmp"
    generate_geom_file(
        exp,
        run,
        ds,
        det,
        psana_file_path,
        tmp_cfel_file_path,
        distance,
        pv_camera_length,
    )
    modify_crystfel_header(tmp_cfel_file_path, cfel_file_path)
    os.remove(tmp_cfel_file_path)

    # Rayonix check
    # if self.diagnostics.psi.get_pixel_size() != self.diagnostics.psi.det.pixel_size(
    #    run
    # ):
    #    print(
    #        "Original geometry is wrong due to hardcoded Rayonix pixel size. Correcting geom file now..."
    #    )
    #    coffset = (
    #        self.distance - self.diagnostics.psi.get_camera_length(pv_camera_length)
    #    ) / 1e3  # convert from mm to m
    #    res = 1e3 / self.diagnostics.psi.get_pixel_size()  # convert from mm to um
    #    os.rename(crystfel_file, temp_file)
    #    modify_crystfel_coffset_res(temp_file, crystfel_file, coffset, res)
    #    os.remove(psana_file)
    #    os.remove(temp_file)
