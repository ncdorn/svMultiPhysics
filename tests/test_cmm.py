import os

from .conftest import (
    run_expect_failure,
    run_with_displacement_mean_reduction,
    run_with_reference,
)

# Common folder for all tests in this file
base_folder = "cmm"

# Fields to test
fields = ["Stress", "Displacement", "Pressure", "Velocity", "Traction", "WSS"]


def test_pipe_3d(n_proc):
    folder = "pipe_3d"
    inflate_folder = os.path.join(folder, "2a-inflate")
    t_max = 3
    run_with_reference(base_folder, inflate_folder, ["Displacement"], 1, t_max)

    inflate_cmm_folder = os.path.join(folder, "3a-inflate-cmm")
    t_max = 5
    run_with_reference(base_folder, inflate_cmm_folder, fields[1::], n_proc, t_max)

    prestress_folder = os.path.join(folder, "2b-prestress")
    t_max = 3
    run_with_reference(base_folder, prestress_folder, fields[0:2], 1, t_max)

    prestress_cmm_folder = os.path.join(folder, "3b-prestress-cmm")
    t_max = 5
    run_with_reference(base_folder, prestress_cmm_folder, fields[1::], n_proc, t_max)


def test_pipe_3d_cmm_robin_zero_uniform(n_proc):
    folder = os.path.join("pipe_3d", "3c-inflate-cmm-robin-zero-uniform")
    run_with_reference(base_folder, folder, fields[1::], n_proc, t_max=5, name_ref="../3a-inflate-cmm/result_005.vtu")


def test_pipe_3d_cmm_robin_zero_spatial(n_proc):
    folder = os.path.join("pipe_3d", "3d-inflate-cmm-robin-zero-spatial")
    run_with_reference(base_folder, folder, fields[1::], n_proc, t_max=5, name_ref="../3a-inflate-cmm/result_005.vtu")


def test_pipe_3d_robin_type_rejected_for_cmm():
    folder = os.path.join("cases", base_folder, "pipe_3d", "3e-inflate-cmm-invalid-robin")
    result = run_expect_failure(folder, n_proc=1)

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "Type = CMM" in output
    assert "Robin support for deformable CMM walls" in output


def test_pipe_3d_cmm_robin_nonzero_uniform_reduces_displacement():
    run_with_displacement_mean_reduction(
        base_folder,
        os.path.join("pipe_3d", "3a-inflate-cmm"),
        os.path.join("pipe_3d", "3f-inflate-cmm-robin-uniform-nonzero"),
        n_proc=1,
        t_max=5,
        min_reduction=0.05,
    )


def test_iliac_artery_variable_wall_props(n_proc):
    folder = "iliac_artery_variable_wall_props"
    inflate_folder = os.path.join(folder, "2-inflate")
    t_max = 3
    run_with_reference(base_folder, inflate_folder, fields[1:2], 1, t_max)

    inflate_cmm_folder = os.path.join(folder, "3-inflate-cmm")
    t_max = 3
    run_with_reference(base_folder, inflate_cmm_folder, fields[1::], n_proc, t_max)
