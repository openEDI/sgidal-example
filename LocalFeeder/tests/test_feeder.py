import logging
import os
import sys

import numpy as np
import pandas as pd
import plotille
import pytest
import xarray as xr
from oedisi.types.data_types import (
    EquipmentNodeArray,
    InverterControl,
    InverterControlMode,
    VVControl,
    VWControl,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FeederSimulator
import sender_cosim
from sender_cosim import agg_to_ids


@pytest.fixture(scope="session", autouse=True)
def init_federate_simulation():
    federate_config = FeederSimulator.FeederConfig(
        **{
            "use_smartds": False,
            "profile_location": "gadal_ieee123/profiles",
            "opendss_location": "gadal_ieee123/qsts",
            "sensor_location": "gadal_ieee123/sensors.json",
            "start_date": "2017-01-01 00:00:00",
            "number_of_timesteps": 96,
            "run_freq_sec": 900,
            "topology_output": "../../outputs/topology.json",
            "name": "feeder",
        }
    )
    logging.info("Initializating simulation data")
    _ = FeederSimulator.FeederSimulator(federate_config)
    return


@pytest.fixture()
def federate_config():
    return FeederSimulator.FeederConfig(
        **{
            "use_smartds": False,
            "profile_location": "gadal_ieee123/profiles",
            "opendss_location": "gadal_ieee123/qsts",
            "sensor_location": "gadal_ieee123/sensors.json",
            "existing_feeder_file": "opendss/master.dss",
            "start_date": "2017-01-01 00:00:00",
            "number_of_timesteps": 96,
            "run_freq_sec": 900,
            "topology_output": "../../outputs/topology.json",
            "name": "feeder",
        }
    )


def plot_y_matrix(Y):
    Y_max = np.max(np.abs(Y))

    width = Y.shape[0] // 2
    height = Y.shape[1] // 4
    cvs = plotille.Canvas(width, height, mode="rgb")
    arrayY = np.log(np.abs(Y.toarray())[: (4 * height), : (2 * width)] + 1)
    flatY = arrayY.reshape(arrayY.size)
    max_flatY = flatY.max()
    cvs.braille_image([int(255 * np.abs(y) / max_flatY) for y in flatY])
    print(f"Y max: {Y_max}")
    print(cvs.plot())


def test_ordering(federate_config):
    logging.info("Loading sim")
    sim = FeederSimulator.FeederSimulator(federate_config)
    Y = sim.get_y_matrix()

    with pytest.raises(AssertionError, match=".*DISABLED_RUN.*"):
        sim.solve(0, 0)
    with pytest.raises(AssertionError, match=".*DISABLED_RUN.*"):
        _ = sim.get_voltages_actual()

    sim.snapshot_run()
    base_voltages = sim.get_voltages_snapshot()
    assert np.max(base_voltages) > 100
    Y2 = sim.get_y_matrix()
    assert np.max(np.abs(Y - Y2)) < 0.001

    sim.snapshot_run()
    sim.solve(0, 0)
    feeder_voltages = sim.get_voltages_actual()
    assert np.max(feeder_voltages) > 100
    with pytest.raises(AssertionError, match=".*SOLVE_AT_TIME.*"):
        _ = sim.get_voltages_snapshot()

    sim.solve(10, 0)
    Y_load = sim.get_load_y_matrix()
    plot_y_matrix(Y)
    plot_y_matrix(Y_load - Y)


def rtheta_to_xy(r, theta):
    x = np.array(r * np.cos(theta))
    y = np.array(r * np.sin(theta))
    return x, y


def test_voltages(federate_config):
    logging.info("Loading sim")
    sim = FeederSimulator.FeederSimulator(federate_config)
    startup(sim)

    # Get Voltages
    base = sim.get_base_voltages()
    sim.disabled_run()
    sim.initial_disabled_solve()
    disabled_solve = sim.get_disabled_solve_voltages()
    sim.snapshot_run()
    snapshot = sim.get_voltages_snapshot()
    sim.solve(0, 0)
    actuals = sim.get_voltages_actual()

    # Plot magnitudes
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    # fig.set_x_limits(min_=-3, max_=3)
    fig.set_y_limits(min_=0, max_=2)
    fig.color_mode = "byte"
    fig.plot(
        range(len(base)), np.abs(disabled_solve / base).data, lc=50, label="Disabled"
    )
    fig.plot(range(len(base)), np.abs(snapshot / base).data, lc=75, label="Snapshot")
    fig.plot(range(len(base)), np.abs(actuals / base).data, lc=100, label="Actuals")
    print("\n" + fig.show(legend=True))

    # Plot magnitudes
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    # fig.set_x_limits(min_=-3, max_=3)
    fig.set_y_limits(min_=0.9, max_=1.1)
    fig.color_mode = "byte"
    fig.plot(
        range(len(base)), np.abs(disabled_solve / base).data, lc=50, label="Disabled"
    )
    fig.plot(range(len(base)), np.abs(snapshot / base).data, lc=75, label="Snapshot")
    fig.plot(range(len(base)), np.abs(actuals / base).data, lc=100, label="Actuals")
    print("\n" + fig.show(legend=True))

    # Plot angles better
    r = xr.DataArray(range(len(base)), {"ids": base.ids.data}) + 100
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    fig.set_x_limits(min_=-400, max_=400)
    fig.set_y_limits(min_=-400, max_=400)
    fig.color_mode = "byte"
    fig.scatter(*rtheta_to_xy(r, np.angle(base)), lc=50, label="Disabled")
    fig.scatter(*rtheta_to_xy(r, np.angle(snapshot)), lc=75, label="Snapshot")
    fig.scatter(*rtheta_to_xy(r, np.angle(actuals)), lc=100, label="Actuals")
    print("\n" + fig.show(legend=True))

    # Plot angles better
    r = xr.DataArray(range(len(base)), {"ids": base.ids.data}) + 100
    fig = plotille.Figure()
    fig.width = 90
    fig.height = 30
    fig.set_x_limits(min_=0, max_=400)
    fig.set_y_limits(min_=-50, max_=50)
    fig.color_mode = "byte"
    fig.scatter(*rtheta_to_xy(r, np.angle(base)), lc=50, label="Disabled")
    fig.scatter(*rtheta_to_xy(r, np.angle(snapshot)), lc=75, label="Snapshot")
    fig.scatter(*rtheta_to_xy(r, np.angle(actuals)), lc=100, label="Actuals")
    print("\n" + fig.show(legend=True))


def test_xarray_translation():
    x = xr.DataArray(
        [0 + 1j, 1 + 2j, 2 + 3j, 3 + 4j],
        dims=("ids",),
        coords={
            "ids": ("ids", ["1", "1", "2", "3"]),
            "equipment_ids": ("ids", ["0", "1", "2", "3"]),
        },
    )
    powerreal, powerimag = sender_cosim.xarray_to_powers(x)
    assert powerreal.ids == ["1", "1", "2", "3"]
    assert powerreal.values == [0.0, 1.0, 2.0, 3.0]
    assert powerimag.values == [1.0, 2.0, 3.0, 4.0]


def test_simulation(federate_config):
    logging.info("Loading sim")
    sim = FeederSimulator.FeederSimulator(federate_config)
    startup(sim)
    Y = sim.get_y_matrix()
    plot_y_matrix(Y)
    sim.snapshot_run()  # You need to re-enable!
    getting_and_concatentating_data(sim)
    initial_data(sim, federate_config)
    sim.snapshot_run()
    simulation_middle(sim, Y)


def startup(sim):
    assert sim._feeder_file is not None
    assert sim._AllNodeNames is not None
    assert sim._circuit is not None
    assert sim._AllNodeNames is not None


def getting_and_concatentating_data(sim):
    PQ_load = sim.get_PQs_load(static=True)
    PQ_PV = sim.get_PQs_pv(static=True)
    PQ_gen = sim.get_PQs_gen(static=True)
    PQ_cap = sim.get_PQs_cap(static=True)

    pv_real, pv_imag = sender_cosim.xarray_to_powers(PQ_PV)
    gen_real, gen_imag = sender_cosim.xarray_to_powers(PQ_gen)
    test_real = sender_cosim.concat_measurement_arrays(pv_real, gen_real)
    assert test_real.values[5] == PQ_PV.data[5].real
    assert test_real.ids[5] == PQ_PV.ids.data[5]

    ids = xr.DataArray(sim._AllNodeNames, coords={"ids": sim._AllNodeNames})
    PQ_injections_all = (
        agg_to_ids(PQ_load, ids)
        + agg_to_ids(PQ_PV, ids)
        + agg_to_ids(PQ_gen, ids)
        + agg_to_ids(PQ_cap, ids)
    )
    assert sorted(list(PQ_injections_all.ids.data)) == sorted(sim._AllNodeNames)


def initial_data(sim, federate_config):
    initial_data = sender_cosim.get_initial_data(sim, federate_config)
    # Plot magnitudes
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    # fig.set_x_limits(min_=-3, max_=3)
    fig.set_y_limits(min_=0, max_=3000)
    fig.color_mode = "byte"
    fig.plot(
        range(len(initial_data.topology.base_voltage_magnitudes.ids)),
        np.sort(initial_data.topology.base_voltage_magnitudes.values),
        lc=100,
        label="Base",
    )
    print("\n" + fig.show(legend=True))

    # Plot angles better
    r = np.array(range(len(initial_data.topology.base_voltage_magnitudes.ids))) + 100
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    fig.set_x_limits(min_=-400, max_=400)
    fig.set_y_limits(min_=-400, max_=400)
    fig.color_mode = "byte"
    fig.scatter(
        *rtheta_to_xy(r, initial_data.topology.base_voltage_angles.values),
        lc=50,
        label="Base",
    )
    print("\n" + fig.show(legend=True))

    df = pd.DataFrame(
        {
            "voltages": initial_data.topology.base_voltage_magnitudes.values,
            "angles": initial_data.topology.base_voltage_angles.values,
        }
    )
    logging.info(df.describe())
    df = pd.DataFrame(
        {"injections": initial_data.topology.injections.power_real.values}
    )
    logging.info(df.describe())
    assert initial_data is not None


def plot_complex_array(data, label="Voltages"):
    print("Index vs Complex Magnitude")
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    # fig.set_x_limits(min_=-3, max_=3)
    fig.set_y_limits(min_=0, max_=np.max(np.abs(data)) * 1.1)
    fig.color_mode = "byte"
    fig.plot(range(len(data)), np.abs(data), lc=100, label=label)
    print("\n" + fig.show(legend=True))

    print("Index vs Complex Angle (radial plot)")
    r = np.array(range(len(data))) + 100
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    fig.set_x_limits(min_=-400, max_=400)
    fig.set_y_limits(min_=-400, max_=400)
    fig.color_mode = "byte"
    fig.scatter(*rtheta_to_xy(r, np.angle(data)), lc=50, label=label)
    print("\n" + fig.show(legend=True))

    print("Log magnitude with angle (radial plot)")
    r = np.log(np.array(np.abs(data)) + 1.0)
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 30
    fig.set_x_limits(min_=-10, max_=10)
    fig.set_y_limits(min_=-10, max_=10)
    fig.color_mode = "byte"
    fig.scatter(*rtheta_to_xy(r, np.angle(data)), lc=50, label=label)
    print("\n" + fig.show(legend=True))


def simulation_middle(sim, Y):
    # this one may need to be properly ordered
    logging.info(f"Current directory : {os.getcwd()}")
    sim.solve(0, 0)

    current_data = sender_cosim.get_current_data(sim, Y)
    assert len(current_data.injections.power_real.values) == len(
        current_data.injections.power_real.ids
    )

    current_data_again = sender_cosim.get_current_data(sim, Y)
    assert np.allclose(current_data_again.feeder_voltages, current_data.feeder_voltages)

    assert "113" in current_data.PQ_injections_all.equipment_ids.data
    df = pd.DataFrame(
        {
            "p": current_data.PQ_injections_all.real,
            "q": current_data.PQ_injections_all.imag,
            "voltages": np.abs(current_data.feeder_voltages),
            "phases": np.angle(current_data.feeder_voltages),
        }
    )
    logging.info(df.describe())
    print("Feeder Voltages")
    plot_complex_array(current_data.feeder_voltages.data, label="Feeder Voltages")
    assert np.max(current_data.feeder_voltages) > 50

    plot_complex_array(current_data.PQ_injections_all.data, label="PQ injection")

    diff = np.abs(
        current_data.PQ_injections_all - (-current_data.calculated_power)
    ) * np.exp(
        1j
        * (
            np.angle(current_data.PQ_injections_all)
            - np.angle(-current_data.calculated_power)
        )
    )

    plot_complex_array(diff.data, label="Calculated - Injected")

    bad_bus_names = sender_cosim.where_power_unbalanced(
        current_data.PQ_injections_all, current_data.calculated_power
    )
    assert len(bad_bus_names) == 0


def equipment_indices_on_equipment_node_array(
    equipment_node_array: EquipmentNodeArray, equipment_type: str
):
    return map(
        lambda iv: iv[0],
        filter(
            lambda iv: iv[1].startswith(equipment_type),
            enumerate(equipment_node_array.equipment_ids),
        ),
    )


def test_controls(federate_config):
    logging.info("Loading sim")
    sim = FeederSimulator.FeederSimulator(federate_config)
    Y = sim.get_y_matrix()
    sim.snapshot_run()  # Needed to bring out of disabled state
    sim.solve(8, 0)
    sim.just_solve()
    current_data = sender_cosim.get_current_data(sim, Y)
    assert len(current_data.injections.power_real.values) == len(
        current_data.injections.power_real.ids
    )

    bad_bus_names = sender_cosim.where_power_unbalanced(
        current_data.PQ_injections_all, current_data.calculated_power
    )
    assert len(bad_bus_names) == 0

    # Find first with equipment type = PVSystem
    power_real = current_data.injections.power_real
    pv_system_indices = list(
        equipment_indices_on_equipment_node_array(power_real, "PVSystem")
    )
    max_index = np.argmax(np.abs([power_real.values[i] for i in pv_system_indices]))
    pv_system_index = pv_system_indices[max_index]
    pv_system_index = None
    for i in range(len(power_real.ids)):
        if power_real.ids[i] == "113.1" and power_real.equipment_ids[i].startswith(
            "PVSystem"
        ):
            pv_system_index = i
            break
    assert pv_system_index is not None

    print(
        f"{power_real.ids[pv_system_index]} {power_real.equipment_ids[pv_system_index]}"
    )
    # Try setting current power to half of that.
    assert abs(power_real.values[pv_system_index]) > 0.01
    sim.change_obj(
        [
            FeederSimulator.Command(
                obj_name="PVSystem.113",
                obj_property="%Pmpp",
                val="5",  # power_real.values[pv_system_index] / 2,
            )
        ]
    )
    # dss.Text.Command("PVsystem.113.PF=0.01")
    # Check properties in AllPropertyNames in CktElement or just Element
    # Solve and observe current power. It should change.
    sim.just_solve()
    new_data = sender_cosim.get_current_data(sim, Y)
    new_power_real = new_data.injections.power_real
    print(power_real.values[pv_system_index])
    print(power_real.ids[pv_system_index])
    print(new_power_real.values[pv_system_index])
    print(new_power_real.ids[pv_system_index])
    assert (
        np.abs(
            new_power_real.values[pv_system_index] - power_real.values[pv_system_index]
        )
        > 1
    )
    (bad_indices,) = np.where(
        np.abs(np.array(new_power_real.values) - np.array(power_real.values)) > 1
    )

    for i in bad_indices:
        print(
            f"Old: {power_real.equipment_ids[i]} {power_real.ids[i]} "
            f"{power_real.values[i]}"
        )
        print(
            f"New: {new_power_real.equipment_ids[i]} {new_power_real.ids[i]} "
            f"{new_power_real.values[i]}"
        )

    assert bad_indices == [pv_system_index]
    # Run another time step. What happens?
    sim.solve(9, 0)
    next_data = sender_cosim.get_current_data(sim, Y)
    next_power_real = next_data.injections.power_real
    print(f"8,0: {power_real.values[pv_system_index]}")
    print(f"8,0: {new_power_real.values[pv_system_index]}")
    print(f"9,0: {next_power_real.values[pv_system_index]}")


def test_inv_control(federate_config):
    logging.info("Loading sim")
    sim = FeederSimulator.FeederSimulator(federate_config)
    test_inverter_data = InverterControl(
        pvsystem_list=["PVSystem.113"],
        vvcontrol=VVControl(
            deltaq_factor=0.7,
            voltage=[0.5, 0.945, 0.97, 0.99, 1.045, 1.50],
            reactive_response=[1, 1, 1, 1, 1, 1],
        ),
    )

    _ = sim.get_y_matrix()
    sim.snapshot_run()  # Needed to bring out of disabled state
    sim.solve(8, 0)

    old_voltages = sim.get_voltages_actual()

    sim.apply_inverter_control(test_inverter_data)
    sim.just_solve()

    new_voltages = sim.get_voltages_actual()

    assert float(
        np.sum(np.abs(new_voltages.loc["113.1"] - old_voltages.loc["113.1"]))
    ) > 0.01 * float(np.abs(old_voltages.loc["113.1"]))

    sim.apply_inverter_control(test_inverter_data)
    sim.just_solve()

    even_newer_voltages = sim.get_voltages_actual()

    assert (
        float(
            np.sum(np.abs(new_voltages.loc["113.1"] - even_newer_voltages.loc["113.1"]))
        )
        < 1
    )

    test_inverter_data = InverterControl(
        pvsystem_list=["PVSystem.113"],
        vvcontrol=VVControl(
            deltaq_factor=0.7,
            voltage=[0.5, 0.945, 0.97, 0.99, 1.045, 1.50],
            reactive_response=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    )

    sim.apply_inverter_control(test_inverter_data)
    sim.just_solve()

    nocontrol_voltages = sim.get_voltages_actual()

    assert (
        float(
            np.sum(np.abs(old_voltages.loc["113.1"] - nocontrol_voltages.loc["113.1"]))
        )
        < 1
    )


def test_inv_control_full(federate_config):
    logging.info("Loading sim")
    sim = FeederSimulator.FeederSimulator(federate_config)
    test_inverter_data = InverterControl(
        vvcontrol=VVControl(
            voltage=[0.5, 0.945, 0.97, 0.99, 1.045, 1.50],
            reactive_response=[1, 1, 1, 1, 1, 1],
        ),
    )

    _ = sim.get_y_matrix()
    sim.snapshot_run()  # Needed to bring out of disabled state
    sim.solve(8, 0)

    old_voltages = sim.get_voltages_actual()

    sim.apply_inverter_control(test_inverter_data)
    sim.just_solve()

    new_voltages = sim.get_voltages_actual()

    assert float(
        np.sum(np.abs(new_voltages.loc["113.1"] - old_voltages.loc["113.1"]))
    ) > 0.01 * float(np.abs(old_voltages.loc["113.1"]))

    assert float(
        np.sum(np.abs(new_voltages.loc["87.3"] - old_voltages.loc["87.3"]))
    ) > 0.01 * float(np.abs(old_voltages.loc["87.3"]))


def test_inv_combined_control(federate_config):
    logging.info("Loading sim")
    sim = FeederSimulator.FeederSimulator(federate_config)

    _ = sim.get_y_matrix()
    sim.snapshot_run()  # Needed to bring out of disabled state
    sim.solve(8, 0)

    old_voltages = sim.get_voltages_actual()

    test_inverter_data = InverterControl(
        vvcontrol=VVControl(
            deltaq_factor=0.7,
            voltage=[0.5, 0.945, 0.97, 0.99, 1.045, 1.50],
            reactive_response=[1, 1, 1, 1, 1, 1],
        ),
        vwcontrol=VWControl(
            deltap_factor=1.0,
            voltage=[0.5, 0.945, 0.97, 0.99, 1.045, 1.50],
            power_response=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ),
        mode=InverterControlMode.voltvar_voltwatt,
    )

    sim.apply_inverter_control(test_inverter_data)
    sim.just_solve()

    new_voltages = sim.get_voltages_actual()

    assert float(
        np.sum(np.abs(new_voltages.loc["113.1"] - old_voltages.loc["113.1"]))
    ) > 0.01 * float(np.abs(old_voltages.loc["113.1"]))

    assert float(
        np.sum(np.abs(new_voltages.loc["87.3"] - old_voltages.loc["87.3"]))
    ) > 0.01 * float(np.abs(old_voltages.loc["87.3"]))


def test_pv_setpoints(federate_config):
    logging.info("Loading sim")
    sim = FeederSimulator.FeederSimulator(federate_config)
    sim.change_obj(
        [
            FeederSimulator.Command(
                obj_name="PVSystem.113",
                obj_property="irradiance",
                val="1",
            ),
            FeederSimulator.Command(
                obj_name="PVSystem.113",
                obj_property="Pmpp",
                val="40",
            ),
        ]
    )
    sim.set_pv_output("113", 20, 5)
    sim.snapshot_run()
    power = (
        -sim.get_PQs_pv(static=True)
        .groupby("equipment_ids")["PVSystem.113"]
        .sum()
        .item()
    )
    assert np.isclose(power.real, 20), f"Real power is {power.real}"
    assert np.isclose(power.imag, 5), f"Reactive power is {power.imag}"

    sim.change_obj(
        [
            FeederSimulator.Command(
                obj_name="PVSystem.113",
                obj_property="Pmpp",
                val="8",
            )
        ]
    )
    sim.snapshot_run()
    sim.set_pv_output("113", 20, 5)
    sim.snapshot_run()
    power = (
        -sim.get_PQs_pv(static=True)
        .groupby("equipment_ids")["PVSystem.113"]
        .sum()
        .item()
    )
    assert np.isclose(power.real, 8), f"Real power is {power.real}"
    assert np.isclose(power.imag, 2), f"Reactive power is {power.imag}"


def test_incidence_matrix(federate_config):
    sim = FeederSimulator.FeederSimulator(federate_config)
    incidences = sim.get_incidences()

    core_bus_names = set(name.split(".")[0] for name in sim._AllNodeNames)
    assert all(
        bus_name.split(".")[0] in core_bus_names
        for bus_name in incidences.from_equipment
    ), f"Could not find the following buses: {list(filter(lambda bus_name: bus_name.split('.')[0] not in core_bus_names, incidences.from_equipment))}"
    assert all(
        bus_name.split(".")[0] in core_bus_names for bus_name in incidences.to_equipment
    ), f"Could not find the following buses: {list(filter(lambda bus_name: bus_name.split('.')[0] not in core_bus_names, incidences.to_equipment))}"
    assert len(incidences.from_equipment) == len(incidences.to_equipment)
    if incidences.equipment_type is not None:
        assert len(incidences.equipment_type) == len(incidences.from_equipment)
    if incidences.ids is not None:
        assert len(incidences.ids) == len(incidences.from_equipment)
