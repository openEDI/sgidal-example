import logging
import helics as h
import numpy as np
from pydantic import BaseModel
import json
from datetime import datetime

from oedisi.types.data_types import MeasurementArray, EquipmentNodeArray
from oedisi.types.common import BrokerConfig

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class MeasurementConfig(BaseModel):
    name: str
    additive_noise_stddev: float = 0.0
    multiplicative_noise_stddev: float = 0.0
    measurement_file: str
    run_freq_time_step: float = 1.0


def get_indices(labelled_array, indices):
    "Get list of indices in the topology for each index of the labelled array"
    inv_map = {v: i for i, v in enumerate(indices)}
    return [inv_map[v] for v in labelled_array.ids]


def reindex(measurement_array: MeasurementArray, indices):
    inv_map = {v: i for i, v in enumerate(measurement_array.ids)}
    if isinstance(measurement_array, EquipmentNodeArray):
        return measurement_array.__class__(
            values=[measurement_array.values[inv_map[i]] for i in indices],
            ids=indices,
            units=measurement_array.units,
            equipment_ids=[
                measurement_array.equipment_ids[inv_map[i]] for i in indices
            ],
            time=measurement_array.time,
        )
    else:
        return measurement_array.__class__(
            values=[measurement_array.values[inv_map[i]] for i in indices],
            ids=indices,
            units=measurement_array.units,
            time=measurement_array.time,
        )


def apply(f, measurement_array: MeasurementArray):
    if isinstance(measurement_array, EquipmentNodeArray):
        return measurement_array.__class__(
            values=list(map(f, measurement_array.values)),
            ids=measurement_array.ids,
            units=measurement_array.units,
            equipment_ids=measurement_array.equipment_ids,
            time=measurement_array.time,
        )
    else:
        return measurement_array.__class__(
            values=list(map(f, measurement_array.values)),
            ids=measurement_array.ids,
            units=measurement_array.units,
            time=measurement_array.time,
        )


class MeasurementRelay:
    def __init__(
        self, config: MeasurementConfig, input_mapping, broker_config: BrokerConfig
    ):
        self.rng = np.random.default_rng(12345)
        # deltat = 60.

        # Create Federate Info object that describes the federate properties #
        fedinfo = h.helicsCreateFederateInfo()
        fedinfo.core_name = config.name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        h.helicsFederateInfoSetBroker(fedinfo, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(fedinfo, broker_config.broker_port)

        logger.debug(config.name)

        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_period, config.run_freq_time_step
        )
        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_offset, 0.03
        )

        self.vfed = h.helicsCreateValueFederate(config.name, fedinfo)
        logger.info("Value federate created")

        # Register the publication #
        self.sub_measurement = self.vfed.register_subscription(
            input_mapping["subscription"], ""
        )

        # TODO: find better way to determine what the name of this federate instance is than looking at the subscription
        self.pub_measurement = self.vfed.register_publication(
            "publication", h.HELICS_DATA_TYPE_STRING, ""
        )

        self.additive_noise_stddev = config.additive_noise_stddev
        self.multiplicative_noise_stddev = config.multiplicative_noise_stddev
        self.measurement_file = config.measurement_file

    def transform(self, measurement_array: MeasurementArray, unique_ids):
        new_array = reindex(measurement_array, unique_ids)
        return apply(
            lambda x: (1 + self.rng.normal(scale=self.multiplicative_noise_stddev)) * x
            + self.rng.normal(scale=self.additive_noise_stddev),
            new_array,
        )

    def run(self):
        # Enter execution mode #
        self.vfed.enter_executing_mode()
        logger.info("Entering execution mode")

        granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)
        while granted_time < h.HELICS_TIME_MAXTIME:
            logger.info("start time: " + str(datetime.now()))
            json_data = self.sub_measurement.json
            if "equipment_ids" in json_data:
                measurement = EquipmentNodeArray.parse_obj(json_data)
            else:
                measurement = MeasurementArray.parse_obj(json_data)

            with open(self.measurement_file, "r") as fp:
                self.measurement = json.load(fp)
            measurement_transformed = self.transform(measurement, self.measurement)
            logger.debug("measured transformed")
            logger.debug(measurement_transformed)

            self.pub_measurement.publish(measurement_transformed.json())

            granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)
            logger.info("end time: " + str(datetime.now()))

        self.destroy()

    def destroy(self):
        h.helicsFederateDisconnect(self.vfed)
        logger.info("Federate disconnected")
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()


def run_simulator(broker_config: BrokerConfig):
    with open("static_inputs.json") as f:
        config = MeasurementConfig(**json.load(f))

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = MeasurementRelay(config, input_mapping, broker_config)
    sfed.run()


if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))
