"""
Basic State Estimation Federate

Uses weighted least squares to estimate the voltage angles.

First `call_h` calculates the residual from the voltage magnitude and angle,
and `call_H` calculates a jacobian. Then `scipy.optimize.least_squares`
is used to solve.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Union

import helics as h
import numpy as np
import scipy.sparse
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    AdmittanceMatrix,
    AdmittanceSparse,
    Complex,
    PowersImaginary,
    PowersReal,
    Topology,
    VoltagesAngle,
    VoltagesMagnitude,
)
from pydantic import BaseModel
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def estimated_pqv(knownP, knownQ, knownV, Y, deltaK, VabsK, num_node):
    """Calculate estimated P, Q, and V."""
    h1 = (VabsK[knownV]).reshape(-1, 1)
    Vp = VabsK * np.exp(1j * deltaK)
    S = Vp * (Y.conjugate() @ Vp.conjugate())
    P, Q = S.real, S.imag
    h2, h3 = P[knownP].reshape(-1, 1), Q[knownQ].reshape(-1, 1)
    h = np.concatenate((h1, h2, h3), axis=0)
    return h.reshape(-1)


def calculate_jacobian(X0, z, num_node, knownP, knownQ, knownV, Y):
    """Calculate the Jacobian matrix for the weighted least squares algorithm.

    Called H in literature."""
    deltaK, VabsK = X0[:num_node], X0[num_node:]
    num_knownV = len(knownV)
    # Calculate original H1
    H11, H12 = np.zeros((num_knownV, num_node)), np.zeros(num_knownV * num_node)
    H12[np.arange(num_knownV) * num_node + knownV] = 1
    H1 = np.concatenate((H11, H12.reshape(num_knownV, num_node)), axis=1)
    Vp = VabsK * np.exp(1j * deltaK)
    ##### S = np.diag(Vp) @ Y.conjugate() @ Vp.conjugate()
    ######  Take gradient with respect to V
    H_pow2 = scipy.sparse.diags_array(Vp) @ Y.conjugate() @ scipy.sparse.diags_array(
        np.exp(-1j * deltaK)
    ) + scipy.sparse.diags_array(np.exp(1j * deltaK) * (Y.conjugate() @ Vp.conjugate()))
    # Take gradient with respect to delta
    H_pow1 = (
        1j
        * scipy.sparse.diags_array(Vp)
        @ (
            scipy.sparse.diags_array(Y.conjugate() @ Vp.conjugate())
            - Y.conjugate() @ scipy.sparse.diags_array(Vp.conjugate())
        )
    )

    if isinstance(Y, scipy.sparse.sparray):
        H2 = scipy.sparse.hstack((H_pow1.real, H_pow2.real))[knownP, :]
        H3 = scipy.sparse.hstack((H_pow1.imag, H_pow2.imag))[knownQ, :]
        assert isinstance(H2, scipy.sparse.sparray), f"H2 has type {type(H2)}"
        assert isinstance(H3, scipy.sparse.sparray), f"H3 has type {type(H3)}"
        H = scipy.sparse.vstack((H1, H2, H3))
    else:
        H2 = np.concatenate((H_pow1.real, H_pow2.real), axis=1)[knownP, :]
        H3 = np.concatenate((H_pow1.imag, H_pow2.imag), axis=1)[knownQ, :]
        H = np.concatenate((H1, H2, H3), axis=0)
    return -H


def residual(X0, z, num_node, knownP, knownQ, knownV, Y):
    delta, Vabs = X0[:num_node], X0[num_node:]
    h = estimated_pqv(knownP, knownQ, knownV, Y, delta, Vabs, num_node)
    logger.debug("X0")
    logger.debug(X0)
    logger.debug("z")
    logger.debug(z)
    logger.debug("h")
    logger.debug(h)
    return z - h


def get_y(admittance: Union[AdmittanceMatrix, AdmittanceSparse], ids: List[str]):
    if type(admittance) == AdmittanceMatrix:
        assert ids == admittance.ids
        return matrix_to_numpy(admittance.admittance_matrix)
    elif type(admittance) == AdmittanceSparse:
        node_map = {name: i for (i, name) in enumerate(ids)}
        return scipy.sparse.coo_array(
            (
                [v[0] + 1j * v[1] for v in admittance.admittance_list],
                (
                    [node_map[r] for r in admittance.from_equipment],
                    [node_map[c] for c in admittance.to_equipment],
                ),
            )
        )


def matrix_to_numpy(admittance: List[List[Complex]]):
    "Convert list of list of our Complex type into a numpy matrix"
    return np.array([[x[0] + 1j * x[1] for x in row] for row in admittance])


def get_indices(topology: Topology, measurement, extra_nodes=set()):
    "Get list of indices in the topology for each index of the input measurement"
    inv_map = {v: i for i, v in enumerate(topology.base_voltage_magnitudes.ids)}
    ordinary_indices = [inv_map[v] for v in measurement.ids]
    extra_indices = [inv_map[v] for v in extra_nodes.difference(measurement.ids)]
    return ordinary_indices + extra_indices


def get_zero_injection_indices(topology: Topology):
    zero_nodes = set(topology.base_voltage_magnitudes.ids)
    zero_nodes = zero_nodes.difference(topology.injections.power_real.ids)
    zero_nodes = zero_nodes.difference(topology.injections.power_imaginary.ids)
    zero_nodes = zero_nodes.difference(topology.injections.current_real.ids)
    zero_nodes = zero_nodes.difference(topology.injections.current_imaginary.ids)
    zero_nodes = zero_nodes.difference(topology.injections.impedance_real.ids)
    zero_nodes = zero_nodes.difference(topology.injections.impedance_imaginary.ids)
    zero_nodes = zero_nodes.difference(topology.slack_bus)
    return zero_nodes


class AlgorithmParameters(BaseModel):
    tol: float = 5e-7
    base_power: Optional[float] = 100.0

    class Config:
        use_enum_values = True


def state_estimator(
    parameters: AlgorithmParameters,
    topology,
    P,
    Q,
    V,
    initial_ang=0,
    initial_V=1,
    slack_index=0,
):
    """Estimates voltage magnitude and angle from topology, partial power injections
    P + Q i, and lossy partial voltage magnitude.

    Parameters
    ----------
    parameters : AlgorithmParameters
        Miscellaneous parameters for algorithm: tolerance, unit-system, etc.
    topology : Topology
        topology includes: Y-matrix, some initial phases, and unique ids
    P : PowersReal (inherited from MeasurementArray)
        Real power injection with unique ids
    Q : PowersImaginary (inherited from MeasurementArray)
        Reactive power injection with unique ids
    V : VoltagesMagnitude (inherited from MeasurementArray)
        Voltage magnitude with unique ids
    """
    base_voltages = np.array(topology.base_voltage_magnitudes.values)
    num_node = len(base_voltages)
    logging.debug("Number of Nodes")
    logging.debug(num_node)

    zero_power_nodes = get_zero_injection_indices(topology)
    knownP = get_indices(topology, P, extra_nodes=zero_power_nodes)
    knownQ = get_indices(topology, Q, extra_nodes=zero_power_nodes)
    knownV = get_indices(topology, V)

    P_array = np.zeros(len(knownP))
    P_array[: len(P.values)] = P.values
    Q_array = np.zeros(len(knownQ))
    Q_array[: len(Q.values)] = Q.values
    z = np.concatenate(
        (
            V.values / base_voltages[knownV],
            -P_array / parameters.base_power,
            -Q_array / parameters.base_power,
        ),
        axis=0,
    )
    Y = get_y(topology.admittance, topology.base_voltage_magnitudes.ids)
    # Hand-crafted unit conversion (check it, it works)
    Y = (
        scipy.sparse.diags_array(base_voltages)
        @ Y
        @ scipy.sparse.diags_array(base_voltages)
    ) / (parameters.base_power * 1000)
    tol = parameters.tol

    if type(initial_ang) != np.ndarray:
        delta = np.full(num_node, initial_ang)
    else:
        delta = initial_ang
    assert delta.shape == (
        num_node,
    ), f"Initial angles shape {delta.shape} does not match {num_node}"

    if type(initial_V) != np.ndarray:
        Vabs = np.full(num_node, initial_V)
    else:
        Vabs = initial_V
    assert Vabs.shape == (
        num_node,
    ), f"Initial Vabs shape {Vabs.shape} does not match {num_node}"

    X0 = np.concatenate((delta, Vabs))
    logging.debug(X0)

    # Weights are ignored since errors are sampled from Gaussian
    # Real dimension of solutions is
    # 2 * num_node - len(knownP) - len(knownV) - len(knownQ)
    ls_result = least_squares(
        residual,
        X0,
        jac=calculate_jacobian,
        method="trf",
        verbose=2,
        ftol=tol,
        xtol=tol,
        gtol=tol,
        args=(z, num_node, knownP, knownQ, knownV, Y),
    )
    solution = ls_result.x
    vmagestDecen, vangestDecen = solution[num_node:], solution[:num_node]
    logging.debug("vangestDecen")
    logging.debug(vangestDecen)
    logging.debug("vmagestDecen")
    logging.debug(vmagestDecen)
    vangestDecen = vangestDecen - vangestDecen[slack_index]
    return vmagestDecen * base_voltages, vangestDecen


class StateEstimatorFederate:
    "State estimator federate. Wraps state_estimation with pubs and subs"

    def __init__(
        self,
        federate_name,
        algorithm_parameters: AlgorithmParameters,
        input_mapping,
        broker_config: BrokerConfig,
    ):
        "Initializes federate with name and remaps input into subscriptions"
        deltat = 0.1

        self.algorithm_parameters = algorithm_parameters

        # Create Federate Info object that describes the federate properties #
        fedinfo = h.helicsCreateFederateInfo()

        h.helicsFederateInfoSetBroker(fedinfo, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(fedinfo, broker_config.broker_port)

        fedinfo.core_name = federate_name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, deltat
        )

        self.vfed = h.helicsCreateValueFederate(federate_name, fedinfo)
        logger.info("Value federate created")

        # Register the publication #
        self.sub_voltages_magnitude = self.vfed.register_subscription(
            input_mapping["voltages_magnitude"], "V"
        )
        self.sub_power_P = self.vfed.register_subscription(
            input_mapping["powers_real"], "W"
        )
        self.sub_power_Q = self.vfed.register_subscription(
            input_mapping["powers_imaginary"], "W"
        )
        self.sub_topology = self.vfed.register_subscription(
            input_mapping["topology"], ""
        )
        self.pub_voltage_mag = self.vfed.register_publication(
            "voltage_mag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_voltage_angle = self.vfed.register_publication(
            "voltage_angle", h.HELICS_DATA_TYPE_STRING, ""
        )

    def run(self):
        "Enter execution and exchange data"
        # Enter execution mode #
        self.vfed.enter_executing_mode()
        logger.info("Entering execution mode")

        granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

        self.initial_ang = None
        self.initial_V = None
        topology = Topology.parse_obj(self.sub_topology.json)
        ids = topology.base_voltage_magnitudes.ids
        logger.info("Topology has been read")
        slack_index = None
        if not isinstance(topology.admittance, AdmittanceMatrix) and not isinstance(
            topology.admittance, AdmittanceSparse
        ):
            raise "Weighted Least Squares algorithm expects AdmittanceMatrix/Sparse as input"

        for i in range(len(ids)):
            if ids[i] == topology.slack_bus[0]:
                slack_index = i

        while granted_time < h.HELICS_TIME_MAXTIME:
            if not self.sub_voltages_magnitude.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.vfed, h.HELICS_TIME_MAXTIME
                )
                continue

            logger.info("start time: " + str(datetime.now()))

            voltages = VoltagesMagnitude.parse_obj(self.sub_voltages_magnitude.json)
            power_P = PowersReal.parse_obj(self.sub_power_P.json)
            power_Q = PowersImaginary.parse_obj(self.sub_power_Q.json)
            knownV = get_indices(topology, voltages)

            if self.initial_V is None:
                voltage_inv_map = {v: i for i, v in enumerate(voltages.ids)}
                topology_inv_map = {
                    v: i for i, v in enumerate(topology.base_voltage_magnitudes.ids)
                }
                if all(map(lambda x: x in voltage_inv_map, topology.slack_bus)):
                    self.initial_V = np.mean(
                        [
                            voltages.values[voltage_inv_map[slack_bus]]
                            / topology.base_voltage_magnitudes.values[
                                topology_inv_map[slack_bus]
                            ]
                            for slack_bus in topology.slack_bus
                        ]
                    )
                else:
                    self.initial_V = np.mean(
                        np.array(voltages.values)
                        / np.array(topology.base_voltage_magnitudes.values)[knownV]
                    )
            if self.initial_ang is None:
                self.initial_ang = np.array(topology.base_voltage_angles.values)

            voltage_magnitudes, voltage_angles = state_estimator(
                self.algorithm_parameters,
                topology,
                power_P,
                power_Q,
                voltages,
                initial_V=self.initial_V,
                initial_ang=self.initial_ang,
                slack_index=slack_index,
            )
            # self.initial_V = voltage_magnitudes
            # self.initial_ang = voltage_angles
            self.pub_voltage_mag.publish(
                VoltagesMagnitude(
                    values=list(voltage_magnitudes), ids=ids, time=voltages.time
                ).json()
            )
            self.pub_voltage_angle.publish(
                VoltagesAngle(
                    values=list(voltage_angles), ids=ids, time=voltages.time
                ).json()
            )
            logger.info("end time: " + str(datetime.now()))

        self.destroy()

    def destroy(self):
        "Finalize and destroy the federates"
        h.helicsFederateDisconnect(self.vfed)
        logger.info("Federate disconnected")

        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()


def run_simulator(broker_config: BrokerConfig):
    with open("static_inputs.json") as f:
        config = json.load(f)
        federate_name = config["name"]
        if "algorithm_parameters" in config:
            parameters = AlgorithmParameters.parse_obj(config["algorithm_parameters"])
        else:
            parameters = AlgorithmParameters.parse_obj({})

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = StateEstimatorFederate(
        federate_name, parameters, input_mapping, broker_config
    )
    sfed.run()


if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))
