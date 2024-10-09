"""HELICS wrapper for OpenDSS feeder simulation."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

import helics as h
import numpy as np
import numpy.typing as npt
import xarray as xr
from FeederSimulator import FeederConfig, FeederSimulator
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    AdmittanceMatrix,
    AdmittanceSparse,
    CommandList,
    EquipmentNodeArray,
    Injection,
    InverterControlList,
    MeasurementArray,
    PowersImaginary,
    PowersReal,
    Topology,
    VoltagesAngle,
    VoltagesImaginary,
    VoltagesMagnitude,
    VoltagesReal,
)
from scipy.sparse import coo_matrix

logger = logging.getLogger("uvicorn.error")

def numpy_to_y_matrix(array: npt.NDArray[np.complex64]):
    """Convert 2d numpy array to list of lists."""
    return [[(element.real, element.imag) for element in row] for row in array]


def sparse_to_admittance_sparse(array: coo_matrix, unique_ids: List[str]):
    """Convert coo sparse array to AdmittanceSparse type."""
    return AdmittanceSparse(
        from_equipment=[unique_ids[i] for i in array.row],
        to_equipment=[unique_ids[i] for i in array.col],
        admittance_list=[(data.real, data.imag) for data in array.data],
    )


def get_true_phases(angle):
    """Round complex angles to predefined set of phases."""
    for test_angle in map(lambda x: x * np.pi / 3, range(-3,4)):
        if np.abs(angle - test_angle) <= np.pi / 6:
            return angle
    raise ValueError(f"angle {angle} not close to -pi to pi")


def xarray_to_dict(data):
    """Convert xarray to dict with values and ids for JSON serialization."""
    coords = {key: list(data.coords[key].data) for key in data.coords.keys()}
    return {"values": list(data.data), **coords}


def xarray_to_powers(data, **kwargs):
    """Conveniently turn xarray into PowersReal and PowersImaginary."""
    powersreal = PowersReal(**xarray_to_dict(data.real), **kwargs)
    powersimag = PowersImaginary(**xarray_to_dict(data.imag), **kwargs)
    return powersreal, powersimag


def concat_measurement_arrays(*ps: MeasurementArray):
    """Concatenate list of measurements into one."""
    accuracy = None
    if all((p.accuracy is not None for p in ps)):
        accuracy = [e for p in ps for e in p.accuracy]

    assert all(ps[0].units == p.units for p in ps)

    bad_data_threshold = None
    if all((p.bad_data_threshold is not None for p in ps)):
        bad_data_threshold = [e for p in ps for e in p.bad_data_threshold]

    assert all(ps[0].time == p.time for p in ps)

    if all((isinstance(p, EquipmentNodeArray) for p in ps)):
        equipment_ids = [e for p in ps for e in p.equipment_ids]

        return ps[0].__class__(
            values=[v for p in ps for v in p.values],
            ids=[id for p in ps for id in p.ids],
            equipment_ids=equipment_ids,
            units=ps[0].units,
            accuracy=accuracy,
            bad_data_threshold=bad_data_threshold,
            time=ps[0].time,
        )
    else:
        return ps[0].__class__(
            values=[v for p in ps for v in p.values],
            ids=[id for p in ps for id in p.ids],
            units=ps[0].units,
            accuracy=accuracy,
            bad_data_threshold=bad_data_threshold,
            time=ps[0].time,
        )


def get_powers(PQ_load, PQ_PV, PQ_gen, PQ_cap):
    """Turn xararys into PowersReal and PowersImaginary."""
    PQ_load_real, PQ_load_imag = xarray_to_powers(PQ_load)
    PQ_PV_real, PQ_PV_imag = xarray_to_powers(PQ_PV)
    PQ_gen_real, PQ_gen_imag = xarray_to_powers(PQ_gen)
    PQ_cap_real, PQ_cap_imag = xarray_to_powers(PQ_cap)

    power_real = concat_measurement_arrays(
        PQ_load_real, PQ_PV_real, PQ_gen_real, PQ_cap_real
    )
    power_imag = concat_measurement_arrays(
        PQ_load_imag, PQ_PV_imag, PQ_gen_imag, PQ_cap_imag
    )
    return power_real, power_imag


@dataclass
class InitialData:
    """Initial data from start of simulation."""

    Y: Any
    topology: Topology


def get_initial_data(sim: FeederSimulator, config: FeederConfig):
    """Get and calculate InitialData from simulation."""
    incidences = sim.get_incidences()
    Y = sim.get_y_matrix()
    unique_ids = sim._AllNodeNames

    if config.use_sparse_admittance:
        admittancematrix = sparse_to_admittance_sparse(Y, unique_ids)
    else:
        admittancematrix = AdmittanceMatrix(
            admittance_matrix=numpy_to_y_matrix(Y.toarray()), ids=unique_ids
        )

    slack_ids = [
        sim._AllNodeNames[i]
        for i in range(sim._source_indexes[0], sim._source_indexes[-1] + 1)
    ]

    base_voltages = sim.get_base_voltages()
    base_voltagemagnitude = VoltagesMagnitude(
        values=list(np.abs(base_voltages).data), ids=list(base_voltages.ids.data)
    )

    # We have to do snapshot run so we can re-enable things properly.
    # Technically we don't have to solve.
    sim.snapshot_run()
    PQ_load = sim.get_PQs_load(static=True)
    PQ_PV = sim.get_PQs_pv(static=True)
    PQ_gen = sim.get_PQs_gen(static=True)
    PQ_cap = sim.get_PQs_cap(static=True)

    sim.solve(0, 0)
    power_real, power_imaginary = get_powers(-PQ_load, -PQ_PV, -PQ_gen, -PQ_cap)
    injections = Injection(power_real=power_real, power_imaginary=power_imaginary)

    feeder_voltages = sim.get_voltages_actual()
    feeder_angles: npt.NDArray[np.float64] = np.angle(feeder_voltages.data)
    phases = list(map(get_true_phases, feeder_angles))
    base_voltageangle = VoltagesAngle(values=phases, ids=list(feeder_voltages.ids.data))

    topology = Topology(
        admittance=admittancematrix,
        base_voltage_angles=base_voltageangle,
        injections=injections,
        base_voltage_magnitudes=base_voltagemagnitude,
        slack_bus=slack_ids,
        incidences=incidences,
    )
    return InitialData(Y=Y, topology=topology)


def agg_to_ids(x: xr.core.dataarray.DataArray, ids):
    """Aggregate xarray to ids. Specialized to equipment node arrays."""
    target = xr.zeros_like(ids, dtype=np.float64)
    if x.shape == (0,):
        return target

    _, x_grouped = xr.align(ids, x.groupby("ids").sum(), join="left", fill_value=0.0)
    return x_grouped


@dataclass
class CurrentData:
    """Current data at time t. ``arr.ids`` gives bus ids."""

    feeder_voltages: xr.core.dataarray.DataArray
    PQ_injections_all: xr.core.dataarray.DataArray
    calculated_power: xr.core.dataarray.DataArray
    injections: Injection
    load_y_matrix: Any


def get_current_data(sim: FeederSimulator, Y):
    """Construct current data from simulation after having solved."""
    feeder_voltages = sim.get_voltages_actual()
    PQ_load = sim.get_PQs_load(static=False)
    PQ_PV = sim.get_PQs_pv(static=False)
    PQ_gen = sim.get_PQs_gen(static=False)
    PQ_cap = sim.get_PQs_cap(static=False)

    # Assumes everything is controllable!
    power_real, power_imaginary = get_powers(-PQ_load, -PQ_PV, -PQ_gen, -PQ_cap)
    injections = Injection(power_real=power_real, power_imaginary=power_imaginary)

    ids = xr.DataArray(
        sim._AllNodeNames,
        coords={
            "ids": sim._AllNodeNames,
        },
    )
    PQ_injections_all = (
        agg_to_ids(PQ_load, ids)
        + agg_to_ids(PQ_PV, ids)
        + agg_to_ids(PQ_gen, ids)
        + agg_to_ids(PQ_cap, ids)
    )

    PQ_injections_all = PQ_injections_all.assign_coords(
        equipment_ids=("ids", list(map(lambda x: x.split(".")[0], sim._AllNodeNames)))
    )
    calculated_power = (
        feeder_voltages * (Y.conjugate() @ feeder_voltages.conjugate()) / 1000
    )

    PQ_injections_all[sim._source_indexes] = -calculated_power[sim._source_indexes]

    Y_load = sim.get_load_y_matrix()
    return CurrentData(
        feeder_voltages=feeder_voltages,
        PQ_injections_all=PQ_injections_all,
        calculated_power=calculated_power,
        injections=injections,
        load_y_matrix=Y_load,
    )


def where_power_unbalanced(PQ_injections_all, calculated_power, tol=1):
    """Find errors where PQ_injectinos does not match calculated power."""
    errors = PQ_injections_all + calculated_power
    (indices,) = np.where(np.abs(errors) > tol)
    return errors.ids[indices]


def go_cosim(
    sim: FeederSimulator,
    config: FeederConfig,
    input_mapping: Dict[str, str],
    broker_config: BrokerConfig,
):
    """Run HELICS federate with FeederSimulation.

    TODO: Maybe this should be a class or a coroutine or something cleaner.
    There are many options.
    """
    deltat = 0.01
    fedinitstring = "--federates=1"

    logger.info("Creating Federate Info")
    fedinfo = h.helicsCreateFederateInfo()

    h.helicsFederateInfoSetBroker(fedinfo, broker_config.broker_ip)
    h.helicsFederateInfoSetBrokerPort(fedinfo, broker_config.broker_port)

    h.helicsFederateInfoSetCoreName(fedinfo, config.name)
    h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
    h.helicsFederateInfoSetCoreInitString(fedinfo, fedinitstring)
    h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_delta, deltat)
    vfed = h.helicsCreateValueFederate(config.name, fedinfo)

    pub_voltages_real = h.helicsFederateRegisterPublication(
        vfed, "voltages_real", h.HELICS_DATA_TYPE_STRING, ""
    )
    pub_voltages_imag = h.helicsFederateRegisterPublication(
        vfed, "voltages_imag", h.HELICS_DATA_TYPE_STRING, ""
    )
    pub_voltages_magnitude = h.helicsFederateRegisterPublication(
        vfed, "voltages_magnitude", h.HELICS_DATA_TYPE_STRING, ""
    )
    pub_powers_real = h.helicsFederateRegisterPublication(
        vfed, "powers_real", h.HELICS_DATA_TYPE_STRING, ""
    )
    pub_powers_imag = h.helicsFederateRegisterPublication(
        vfed, "powers_imag", h.HELICS_DATA_TYPE_STRING, ""
    )
    pub_topology = h.helicsFederateRegisterPublication(
        vfed, "topology", h.HELICS_DATA_TYPE_STRING, ""
    )
    pub_injections = h.helicsFederateRegisterPublication(
        vfed, "injections", h.HELICS_DATA_TYPE_STRING, ""
    )
    pub_available_power = h.helicsFederateRegisterPublication(
        vfed, "available_power", h.HELICS_DATA_TYPE_STRING, ""
    )
    pub_load_y_matrix = h.helicsFederateRegisterPublication(
        vfed, "load_y_matrix", h.HELICS_DATA_TYPE_STRING, ""
    )
    pub_pv_forecast = h.helicsFederateRegisterPublication(
        vfed, "pv_forecast", h.HELICS_DATA_TYPE_STRING, ""
    )

    command_set_key = (
        "unused/change_commands"
        if "change_commands" not in input_mapping
        else input_mapping["change_commands"]
    )
    sub_command_set = vfed.register_subscription(command_set_key, "")
    sub_command_set.set_default("[]")
    sub_command_set.option["CONNECTION_OPTIONAL"] = True

    inv_control_key = (
        "unused/inv_control"
        if "" not in input_mapping
        else input_mapping["inv_control"]
    )
    sub_invcontrol = vfed.register_subscription(inv_control_key, "")
    sub_invcontrol.set_default("[]")
    sub_invcontrol.option["CONNECTION_OPTIONAL"] = True

    pv_set_key = (
        "unused/pv_set" if "pv_set" not in input_mapping else input_mapping["pv_set"]
    )

    sub_pv_set = vfed.register_subscription(pv_set_key, "")
    sub_pv_set.set_default("[]")
    sub_pv_set.option["CONNECTION_OPTIONAL"] = True

    h.helicsFederateEnterExecutingMode(vfed)
    initial_data = get_initial_data(sim, config)

    topology_dict = initial_data.topology.dict()
    topology_dict["bus_coords"] = sim.get_bus_coords()
    topology_json = json.dumps(topology_dict)
    logger.info("Sending topology and saving to topology.json")
    with open(config.topology_output, "w") as f:
        f.write(topology_json)
    pub_topology.publish(topology_json)

    # Publish the forecasted PV outputs as a list of MeasurementArray
    logger.info("Evaluating the forecasted PV")
    forecast_data = sim.forcast_pv(int(config.number_of_timesteps))
    PVforecast = [MeasurementArray(**xarray_to_dict(forecast),
                    units="kW").json() for forecast in forecast_data]
    pub_pv_forecast.publish(json.dumps(PVforecast))

    granted_time = -1
    request_time = 0
    initial_timestamp = datetime.strptime(
        config.start_date, "%Y-%m-%d %H:%M:%S"
    )

    while request_time < int(config.number_of_timesteps):
        granted_time = h.helicsFederateRequestTime(vfed, request_time)
        assert (
            granted_time <= request_time + deltat
        ), f"granted_time: {granted_time} past {request_time}"
        if granted_time >= request_time - deltat:
            request_time += 1

        current_index = int(granted_time)  # floors
        current_timestamp = datetime.strptime(
            config.start_date, "%Y-%m-%d %H:%M:%S"
        ) + timedelta(seconds=granted_time * config.run_freq_sec)
        floored_timestamp = datetime.strptime(
            config.start_date, "%Y-%m-%d %H:%M:%S"
        ) + timedelta(seconds=current_index * config.run_freq_sec)

        change_obj_cmds = CommandList.parse_obj(sub_command_set.json)
        sim.change_obj(change_obj_cmds.__root__)

        inverter_controls = InverterControlList.parse_obj(sub_invcontrol.json)
        for inv_control in inverter_controls.__root__:
            sim.apply_inverter_control(inv_control)

        pv_sets = sub_pv_set.json
        for pv_set in pv_sets:
            sim.set_pv_output(pv_set[0].split(".")[1], pv_set[1], pv_set[2])

        current_hour = 24*(floored_timestamp.date() - initial_timestamp.date()).days + floored_timestamp.hour
        logger.info(
            f"Solve at hour {current_hour} second "
            f"{60*floored_timestamp.minute + floored_timestamp.second}"
        )

        sim.snapshot_run()
        sim.solve(
            current_hour,
            60 * floored_timestamp.minute + floored_timestamp.second,
        )

        current_data = get_current_data(sim, initial_data.Y)

        bad_bus_names = where_power_unbalanced(
            current_data.PQ_injections_all, current_data.calculated_power
        )
        if len(bad_bus_names) > 0:
            raise ValueError(
                f"""
            Bad buses at {bad_bus_names.data}

            OpenDSS PQ
            {current_data.PQ_injections_all.loc[bad_bus_names]}

            PowerBalance PQ
            {current_data.calculated_power.loc[bad_bus_names]}
            """
            )

        logger.debug(
            f"Publish load {current_data.feeder_voltages.ids.data[0]} "
            f"{current_data.feeder_voltages.data[0]}"
        )
        voltage_magnitudes = np.abs(current_data.feeder_voltages)
        pub_voltages_magnitude.publish(
            VoltagesMagnitude(
                **xarray_to_dict(voltage_magnitudes),
                time=current_timestamp,
            ).json()
        )
        pub_voltages_real.publish(
            VoltagesReal(
                **xarray_to_dict(current_data.feeder_voltages.real),
                time=current_timestamp,
            ).json()
        )
        pub_voltages_imag.publish(
            VoltagesImaginary(
                **xarray_to_dict(current_data.feeder_voltages.imag),
                time=current_timestamp,
            ).json()
        )
        pub_powers_real.publish(
            PowersReal(
                **xarray_to_dict(current_data.PQ_injections_all.real),
                time=current_timestamp,
            ).json()
        )
        pub_powers_imag.publish(
            PowersImaginary(
                **xarray_to_dict(current_data.PQ_injections_all.imag),
                time=current_timestamp,
            ).json()
        )
        pub_injections.publish(current_data.injections.json())
        pub_available_power.publish(
            MeasurementArray(
                **xarray_to_dict(sim.get_available_pv()),
                time=current_timestamp,
                units="kWA",
            ).json()
        )

        if config.use_sparse_admittance:
            pub_load_y_matrix.publish(
                sparse_to_admittance_sparse(
                    current_data.load_y_matrix, sim._AllNodeNames
                ).json()
            )
        else:
            pub_load_y_matrix.publish(
                AdmittanceMatrix(
                    admittance_matrix=numpy_to_y_matrix(
                        current_data.load_y_matrix.toarray()
                    ),
                    ids=sim._AllNodeNames,
                ).json()
            )

        logger.info("end time: " + str(datetime.now()))

    h.helicsFederateDisconnect(vfed)
    h.helicsFederateFree(vfed)
    h.helicsCloseLibrary()


def run_simulator(broker_config: BrokerConfig):
    """Load static_inputs and input_mapping and run JSON."""
    with open("static_inputs.json") as f:
        parameters = json.load(f)
    with open("input_mapping.json") as f:
        input_mapping = json.load(f)
    config = FeederConfig(**parameters)
    sim = FeederSimulator(config)
    go_cosim(sim, config, input_mapping, broker_config)


if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))
