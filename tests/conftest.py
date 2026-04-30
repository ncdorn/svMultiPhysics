import numpy as np

import pytest
import os
import shutil
import platform
import subprocess
import meshio

is_not_Darwin = True
if platform.system() == "Darwin": is_not_Darwin = False

this_file_dir = os.path.abspath(os.path.dirname(__file__))
cpp_exec = os.path.join(this_file_dir, "..", "build", "svMultiPhysics-build", "bin", "svmultiphysics")
cpp_exec_p = os.path.join(this_file_dir, "..", "build-petsc", "svMultiPhysics-build", "bin", "svmultiphysics")

# Relative tolerances for each tested field
RTOL = {
    "Action_potential": 1.0e-10,
    "Cauchy_stress": 1.0e-4,
    "Concentration": 1.0e-10,
    "Def_grad": 1.0e-10,
    "Divergence": 1.0e-9,
    "Displacement": 1.0e-10,
    "Jacobian": 1.0e-10,
    "Pressure": 1.0e-6,
    "Stress": 1.0e-4,
    "Strain": 1.0e-10,
    "Temperature": 1.0e-10,
    "Traction": 1.0e-6,
    "Velocity": 1.0e-7,
    "VonMises_stress": 1.0e-3,
    "Vorticity": 1.0e-7,
    "WSS": 1.0e-8,
}

# Number of processors to test
PROCS = [1, 3, 4]


# Fixture to parametrize the number of processors for all tests
@pytest.fixture(params=PROCS)
def n_proc(request):
    return request.param


def resolve_case_dir(folder):
    if os.path.isabs(folder):
        return folder
    return os.path.join(this_file_dir, folder)


def run_by_name(folder, name, t_max, n_proc=1):
    """
    Run a test case and return results
    Args:
        folder: location from which test will be executed
        name: name of svMultiPhysics input file (.xml)
        t_max: time step to compare
        n_proc: number of processors

    Returns:
    Simulation results
    """

    folder = resolve_case_dir(folder)

    # remove old results folders if they exist
    dir_path = os.path.join(folder, str(n_proc) + "-procs")
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    # run simulation
    if is_not_Darwin:
        if "petsc" in folder:
            cmd = " ".join(
            [
                "mpirun",
                "--oversubscribe" if n_proc > 1 else "",
                "-np",
                str(n_proc),
                cpp_exec_p,
                name,
            ]
            )
        else:
            cmd = " ".join(
            [
                "mpirun",
                "--oversubscribe" if n_proc > 1 else "",
                "-np",
                str(n_proc),
                cpp_exec,
                name,
            ]
            )
    else:
        if "petsc" in folder or "trilinos" in folder: 
            return
        else:
            cmd = " ".join(
                [
                    "mpirun",
                    "--oversubscribe" if n_proc > 1 else "",
                    "-np",
                    str(n_proc),
                    cpp_exec,
                    name,
                ]
                )

    subprocess.call(cmd, cwd=folder, shell=True)

    # read results
    fname = os.path.join(
        folder, str(n_proc) + "-procs", "result_" + str(t_max).zfill(3) + ".vtu"
    )
    if not os.path.exists(fname):
        raise RuntimeError("No svMultiPhysics output: " + fname)
    return meshio.read(fname)


def run_expect_failure(folder, name="solver.xml", n_proc=1):
    """
    Run a test case expected to fail and return the completed process.
    Args:
        folder: location from which test will be executed
        name: name of svMultiPhysics input file (.xml)
        n_proc: number of processors
    """

    folder = resolve_case_dir(folder)

    if is_not_Darwin:
        if "petsc" in folder:
            cmd = " ".join(
                [
                    "mpirun",
                    "--oversubscribe" if n_proc > 1 else "",
                    "-np",
                    str(n_proc),
                    cpp_exec_p,
                    name,
                ]
            )
        else:
            cmd = " ".join(
                [
                    "mpirun",
                    "--oversubscribe" if n_proc > 1 else "",
                    "-np",
                    str(n_proc),
                    cpp_exec,
                    name,
                ]
            )
    else:
        if "petsc" in folder or "trilinos" in folder:
            pytest.skip("Failure-path test not run for PETSc/Trilinos cases on Darwin")
        else:
            cmd = " ".join(
                [
                    "mpirun",
                    "--oversubscribe" if n_proc > 1 else "",
                    "-np",
                    str(n_proc),
                    cpp_exec,
                    name,
                ]
            )

    return subprocess.run(cmd, cwd=folder, shell=True, capture_output=True, text=True)


def run_with_reference(
    base_folder,
    test_folder,
    fields,
    n_proc=1,
    t_max=1,
    name_ref=None,
    name_inp="solver.xml",
):
    """
    Run a test case and compare it to a stored reference solution
    Args:
        folder: location from which test will be executed
        fields: array fields to compare (e.g. ["Pressure", "Velocity"])
        n_proc: number of processors
        t_max: time step to compare
        name_inp: name of svMultiPhysics input file (.xml)
        name_ref: name of refence file (.vtu)
    """
    # default reference name
    if not name_ref:
        name_ref = "result_" + str(t_max).zfill(3) + ".vtu"

    # run simulation
    folder = os.path.join("cases", base_folder, test_folder)
    
    if is_not_Darwin:
        res = run_by_name(folder, name_inp, t_max, n_proc)
    else:
        if "petsc" in folder or "trilinos" in folder: 
            return
        else:
            res = run_by_name(folder, name_inp, t_max, n_proc)

    # read reference
    fname = os.path.join(resolve_case_dir(folder), name_ref)
    ref = meshio.read(fname)

    # check results
    msg = ""
    for f in fields:
        # extract field
        if f not in res.point_data.keys():
            raise ValueError("Field " + f + " not in simulation result")
        a = res.point_data[f]

        if f not in ref.point_data.keys():
            raise ValueError("Field " + f + " not in reference result")
        b = ref.point_data[f]

        # truncate last dimension if solution is 2D but reference is 3D
        if len(a.shape) == 2:
            if a.shape[1] == 2 and b.shape[1] == 3:
                assert not np.any(b[:, 2])
                b = b[:, :2]

        # pick tolerance for current field
        if f not in RTOL:
            raise ValueError("No tolerance defined for field " + f)
        rtol = RTOL[f]

        # relative difference (as computed in np.isclose)
        # note that we consider rtol as absolute zero (and as relative tolerance)
        a_fl = a.flatten()
        b_fl = b.flatten()
        rel_diff = np.abs(a_fl - b_fl) - rtol - rtol * np.abs(b_fl)

        # throw error if not all results are within relative tolerance
        close = rel_diff <= 0.0
        if not np.all(close):
            # portion of individual results that are above the tolerance
            wrong = 1 - np.sum(close) / close.size

            # location of maximum relative difference
            i_max = rel_diff.argmax()

            # maximum relative difference
            max_rel = rel_diff[i_max]

            # maximum absolute difference at same location
            max_abs = np.abs(a_fl[i_max] - b_fl[i_max])

            # throw error message for pytest
            msg += "Test failed in field " + f + "."
            msg += " Results differ by more than rtol=" + str(rtol)
            msg += " in {:.1%}".format(wrong)
            msg += " of results."
            msg += " Max. rel. difference is"
            msg += " {:.1e}".format(max_rel)
            msg += " (abs. {:.1e}".format(max_abs) + ")\n"
    # check all fields first and then throw error if any failed
    if msg:
        raise AssertionError(msg)


def run_with_displacement_mean_reduction(
    base_folder,
    baseline_folder,
    test_folder,
    n_proc=1,
    t_max=1,
    min_reduction=0.05,
    name_inp="solver.xml",
):
    """
    Run a baseline and comparison case and assert the mean displacement
    magnitude of the comparison run is reduced by at least min_reduction.
    """
    baseline_path = os.path.join(this_file_dir, "cases", base_folder, baseline_folder)
    test_path = os.path.join(this_file_dir, "cases", base_folder, test_folder)

    baseline = run_by_name(baseline_path, name_inp, t_max, n_proc)
    test = run_by_name(test_path, name_inp, t_max, n_proc)

    if "Displacement" not in baseline.point_data:
        raise ValueError("Field Displacement not in baseline result")
    if "Displacement" not in test.point_data:
        raise ValueError("Field Displacement not in comparison result")

    baseline_mean = np.linalg.norm(baseline.point_data["Displacement"], axis=1).mean()
    test_mean = np.linalg.norm(test.point_data["Displacement"], axis=1).mean()

    if baseline_mean <= 0.0:
        raise AssertionError("Baseline mean displacement magnitude must be positive")

    reduction = 1.0 - test_mean / baseline_mean
    if reduction < min_reduction:
        raise AssertionError(
            "Mean displacement magnitude reduction was "
            f"{reduction:.2%}, below the required {min_reduction:.2%}. "
            f"Baseline={baseline_mean:.6e}, comparison={test_mean:.6e}"
        )
