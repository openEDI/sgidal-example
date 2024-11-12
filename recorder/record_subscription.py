import csv
import json
import logging
from datetime import datetime
import time

import helics as h
import numpy as np
import pandas as pd
import pyarrow as pa
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import MeasurementArray

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class Recorder:
    def __init__(
        self,
        name,
        feather_filename,
        csv_filename,
        input_mapping,
        broker_config: BrokerConfig,
    ):
        self.rng = np.random.default_rng(12345)
        deltat = 0.01
        # deltat = 60.

        # Create Federate Info object that describes the federate properties #
        fedinfo = h.helicsCreateFederateInfo()

        h.helicsFederateInfoSetBroker(fedinfo, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(fedinfo, broker_config.broker_port)

        fedinfo.core_name = name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        logger.debug(name)

        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, deltat
        )

        self.vfed = h.helicsCreateValueFederate(name, fedinfo)
        logger.info("Value federate created")

        # Register the publication #
        self.sub = self.vfed.register_subscription(input_mapping["subscription"], "")
        self.feather_filename = feather_filename
        self.csv_filename = csv_filename

    def run(self):
        # Enter execution mode #
        self.vfed.enter_initializing_mode()
        self.vfed.enter_executing_mode()
        logger.info("Entering execution mode")

        start = True
        granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

        with pa.OSFile(self.feather_filename, "wb") as sink, pa.OSFile(
            self.feather_filename + ".stream", "wb"
        ) as streamsink:
            writer = None
            streamwriter = None
            while granted_time < h.HELICS_TIME_MAXTIME:
                logger.info("start time: " + str(datetime.now()))
                logger.debug(granted_time)
                # Check that the data is a MeasurementArray type
                measurement = MeasurementArray(**self.sub.json)
                measurement_dict = {
                    key: value
                    for key, value in zip(measurement.ids, measurement.values)
                }
                measurement_dict["time"] = measurement.time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                logger.debug(measurement.time)

                if start:
                    schema_elements = [(key, pa.float64()) for key in measurement.ids]
                    schema_elements.append(("time", pa.string()))
                    schema = pa.schema(schema_elements)
                    writer = pa.ipc.new_file(sink, schema)
                    streamwriter = pa.ipc.new_stream(streamsink, schema)
                    start = False

                record_batch = pa.RecordBatch.from_pylist([measurement_dict])
                writer.write_batch(record_batch)
                streamwriter.write_batch(record_batch)

                granted_time = h.helicsFederateRequestTime(
                    self.vfed, h.HELICS_TIME_MAXTIME
                )
                logger.info("end time: " + str(datetime.now()))

            if writer is not None:
                writer.close()
                streamwriter.close()
        data = pd.read_feather(self.feather_filename)
        data.to_csv(self.csv_filename, header=True, index=False)
        self.destroy()

    def destroy(self):
        h.helicsFederateDisconnect(self.vfed)
        logger.info("Federate disconnected")
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()


def run_simulator(broker_config: BrokerConfig):
    with open("static_inputs.json") as f:
        config = json.load(f)
        name = config["name"]
        feather_path = config["feather_filename"]
        csv_path = config["csv_filename"]

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = Recorder(name, feather_path, csv_path, input_mapping, broker_config)
    sfed.run()


if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))
