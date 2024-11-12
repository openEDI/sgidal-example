from pathlib import Path
from oedisi.types.data_types import (
    Topology,
    PowersReal,
    PowersImaginary,
    VoltagesMagnitude,
    VoltagesReal,
    VoltagesImaginary,
)
import pandas as pd
import os


def load_timestep(filename, timestep):
    df = pd.read_feather(filename).drop("time", axis=1)
    return {"ids": list(df.columns), "values": list(df.iloc[timestep, :])}


def write_test_data(outputsdir, targetdir, timestep):
    topology = Topology.parse_file(outputsdir / "topology.json")
    power_real = load_timestep(outputsdir / "measured_power_real.feather", timestep)
    power_imag = load_timestep(outputsdir / "measured_power_imag.feather", timestep)
    voltage_mag = load_timestep(
        outputsdir / "measured_voltage_magnitude.feather", timestep
    )
    voltage_real = load_timestep(outputsdir / "voltage_real.feather", timestep)
    voltage_imag = load_timestep(outputsdir / "voltage_imag.feather", timestep)

    with open(targetdir / "topology.json", "w") as f:
        f.write(topology.json())
    with open(targetdir / "power_real.json", "w") as f:
        f.write(PowersReal(**power_real, equipment_ids=[]).json())
    with open(targetdir / "power_imag.json", "w") as f:
        f.write(PowersImaginary(**power_imag, equipment_ids=[]).json())
    with open(targetdir / "voltage_magnitude.json", "w") as f:
        f.write(VoltagesMagnitude(**voltage_mag).json())
    with open(targetdir / "voltage_real.json", "w") as f:
        f.write(VoltagesReal(**voltage_real).json())
    with open(targetdir / "voltage_imaginary.json", "w") as f:
        f.write(VoltagesImaginary(**voltage_imag).json())


test_data_dir = "wls_federate/tests/large_smartds_no_noise_3"
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)
write_test_data(Path("outputs_large_no_noise"), Path(test_data_dir), 3)

test_data_dir = "wls_federate/tests/large_smartds_noise_3"
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)
write_test_data(Path("outputs_large_noise"), Path(test_data_dir), 3)

test_data_dir = "wls_federate/tests/large_smartds_no_noise_40"
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)
write_test_data(Path("outputs_large_no_noise"), Path(test_data_dir), 40)

test_data_dir = "wls_federate/tests/large_smartds_noise_40"
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)
write_test_data(Path("outputs_large_noise"), Path(test_data_dir), 40)
