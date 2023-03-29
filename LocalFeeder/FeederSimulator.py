# -*- coding: utf-8 -*-
# import helics as h
from typing import Any, List
import opendssdirect as dss
import numpy as np
import time
from time import strptime
from scipy.sparse import coo_matrix
import os
import random
import math
import logging
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from pydantic import BaseModel

from dss_functions import (
    snapshot_run,
    parse_Ymatrix,
    get_loads,
    get_pvSystems,
    get_Generator,
    get_capacitors,
    get_voltages,
    get_y_matrix_file,
    get_vnom,
    get_vnom2,
)
import dss_functions
import xarray as xr

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def permutation(from_list, to_list):
    """
    Create permutation representing change in from_list to to_list

    Specifically, if `permute = permutation(from_list, to_list)`,
    then `permute[i] = j` means that `from_list[i] = to_list[j]`.

    This also means that `to_list[permute] == from_list`, so you
    can convert from indices under to_list to indices under from_list.

    You may view the permutation as a function from `from_list` to `to_list`.
    """
    # return [to_list.find(v) for v in enumerate(from_list)]
    index_map = {v: i for i, v in enumerate(to_list)}
    return [index_map[v] for v in from_list]


def check_node_order(l1, l2):
    logger.debug("check order " + str(l1 == l2))


class FeederConfig(BaseModel):
    name: str
    use_smartds: bool = False
    profile_location: str
    opendss_location: str
    sensor_location: str = ""
    start_date: str
    number_of_timesteps: float
    run_freq_sec: float = 15 * 60
    start_time_index: int = 0
    topology_output: str = "topology.json"
    use_sparse_admittance = False


class Command(BaseModel):
    obj_name: str
    obj_property: str
    val: Any


class CommandList(BaseModel):
    __root__: List[Command]


class FeederSimulator(object):
    """ A simple class that handles publishing the solar forecast
    """

    def __init__(self, config: FeederConfig):
        """ Create a ``FeederSimulator`` object

        """
        self._feeder_file = None
        self._simulation_time_step = None
        self._opendss_location = config.opendss_location
        self._profile_location = config.profile_location
        self._sensor_location = config.sensor_location
        self._use_smartds = config.use_smartds

        self._circuit = None
        self._AllNodeNames = None
        self._source_indexes = None
        # self._capacitors=[]
        self._capNames = []
        self._regNames = []

        # timegm(strptime('2019-07-23 14:50:00 GMT', '%Y-%m-%d %H:%M:%S %Z'))
        self._start_time = int(
            time.mktime(strptime(config.start_date, "%Y-%m-%d %H:%M:%S"))
        )
        self._run_freq_sec = config.run_freq_sec
        self._simulation_step = config.start_time_index
        self._number_of_timesteps = config.number_of_timesteps
        self._vmult = 0.001

        self._nodes_index = []
        self._name_index_dict = {}

        self._simulation_time_step = "15m"
        if self._use_smartds:
            self._feeder_file = os.path.join("opendss", "Master.dss")
            if not os.path.isfile(os.path.join("opendss", "Master.dss")):
                self.download_data("oedi-data-lake", "Master.dss", True)
            self.load_feeder()
            self.create_measurement_lists()
        else:
            self._feeder_file = os.path.join("opendss", "master.dss")
            if not os.path.isfile(os.path.join("opendss", "master.dss")):
                self.download_data("gadal", "master.dss")
            self.load_feeder()

    def download_data(self, bucket_name, master_name, update_loadshape_location=False):
        logging.info(f"Downloading from bucket {bucket_name}")
        # Equivalent to --no-sign-request
        s3_resource = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
        bucket = s3_resource.Bucket(bucket_name)
        opendss_location = self._opendss_location
        profile_location = self._profile_location
        sensor_location = self._sensor_location

        for obj in bucket.objects.filter(Prefix=opendss_location):
            output_location = os.path.join(
                "opendss", obj.key.replace(opendss_location, "").strip("/")
            )
            os.makedirs(os.path.dirname(output_location), exist_ok=True)
            bucket.download_file(obj.key, output_location)

        modified_loadshapes = ""
        os.makedirs(os.path.join("profiles"), exist_ok=True)
        if update_loadshape_location:
            all_profiles = set()
            with open(os.path.join("opendss", "LoadShapes.dss"), "r") as fp_loadshapes:
                for row in fp_loadshapes.readlines():
                    new_row = row.replace("../", "")
                    new_row = new_row.replace("file=", "file=../")
                    for token in new_row.split(" "):
                        if token.startswith("(file="):
                            location = (
                                token.split("=../profiles/")[1].strip().strip(")")
                            )
                            all_profiles.add(location)
                    modified_loadshapes = modified_loadshapes + new_row
            with open(os.path.join("opendss", "LoadShapes.dss"), "w") as fp_loadshapes:
                fp_loadshapes.write(modified_loadshapes)
            for profile in all_profiles:
                s3_location = f"{profile_location}/{profile}"
                bucket.download_file(s3_location, os.path.join("profiles", profile))

        else:
            for obj in bucket.objects.filter(Prefix=profile_location):
                output_location = os.path.join(
                    "profiles", obj.key.replace(profile_location, "").strip("/")
                )
                os.makedirs(os.path.dirname(output_location), exist_ok=True)
                bucket.download_file(obj.key, output_location)

        if sensor_location != "":
            output_location = os.path.join("sensors", os.path.basename(sensor_location))
            if not os.path.exists(os.path.dirname(output_location)):
                os.makedirs(os.path.dirname(output_location))
            bucket.download_file(sensor_location, output_location)

    def create_measurement_lists(
        self,
        percent_voltage=75,
        percent_real=75,
        percent_reactive=75,
        voltage_seed=1,
        real_seed=2,
        reactive_seed=3,
    ):

        random.seed(voltage_seed)
        os.makedirs("sensors", exist_ok=True)
        voltage_subset = random.sample(
            self._AllNodeNames,
            math.floor(len(self._AllNodeNames) * float(percent_voltage) / 100),
        )
        with open(os.path.join("sensors", "voltage_ids.json"), "w") as fp:
            json.dump(voltage_subset, fp, indent=4)

        random.seed(real_seed)
        real_subset = random.sample(
            self._AllNodeNames,
            math.floor(len(self._AllNodeNames) * float(percent_real) / 100),
        )
        with open(os.path.join("sensors", "real_ids.json"), "w") as fp:
            json.dump(real_subset, fp, indent=4)

        random.seed(reactive_seed)
        reactive_subset = random.sample(
            self._AllNodeNames,
            math.floor(len(self._AllNodeNames) * float(percent_voltage) / 100),
        )
        with open(os.path.join("sensors", "reactive_ids.json"), "w") as fp:
            json.dump(reactive_subset, fp, indent=4)

    def snapshot_run(self):
        snapshot_run(dss)

    def get_circuit_name(self):
        return self._circuit.Name()

    def get_source_indices(self):
        return self._source_indexes

    def get_node_names(self):
        return self._AllNodeNames

    def load_feeder(self):
        dss.Basic.LegacyModels(True)
        dss.run_command("clear")
        result = dss.run_command("redirect " + self._feeder_file)
        if not result == "":
            raise ValueError("Feeder not loaded: " + result)
        self._circuit = dss.Circuit
        self._AllNodeNames = self._circuit.YNodeOrder()
        self._node_number = len(self._AllNodeNames)
        self._nodes_index = [self._AllNodeNames.index(ii) for ii in self._AllNodeNames]
        self._name_index_dict = {
            ii: self._AllNodeNames.index(ii) for ii in self._AllNodeNames
        }

        self._source_indexes = []
        for Source in dss.Vsources.AllNames():
            self._circuit.SetActiveElement("Vsource." + Source)
            Bus = dss.CktElement.BusNames()[0].upper()
            for phase in range(1, dss.CktElement.NumPhases() + 1):
                self._source_indexes.append(
                    self._AllNodeNames.index(Bus.upper() + "." + str(phase))
                )

        self.setup_vbase()

    def get_y_matrix(self):
        get_y_matrix_file(dss)
        Ymatrix = parse_Ymatrix("base_ysparse.csv", self._node_number)
        new_order = self._circuit.YNodeOrder()
        permute = np.array(permutation(new_order, self._AllNodeNames))
        # inv_permute = np.array(permutation(self._AllNodeNames, new_order))
        return coo_matrix(
            (Ymatrix.data, (permute[Ymatrix.row], permute[Ymatrix.col])),
            shape=Ymatrix.shape,
        )

    def setup_vbase(self):
        self._Vbase_allnode = np.zeros((self._node_number), dtype=np.complex_)
        self._Vbase_allnode_dict = {}
        for ii, node in enumerate(self._AllNodeNames):
            self._circuit.SetActiveBus(node)
            self._Vbase_allnode[ii] = dss.Bus.kVBase() * 1000
            self._Vbase_allnode_dict[node] = self._Vbase_allnode[ii]

    def get_G_H(self, Y11_inv):
        Vnom = self.get_vnom()
        # ys=Y11
        # R = np.linalg.inv(ys).real
        # X = np.linalg.inv(ys).imag
        R = Y11_inv.real
        X = Y11_inv.imag
        G = (
            R * np.diag(np.cos(np.angle(Vnom)) / abs(Vnom))
            - X * np.diag(np.sin(np.angle(Vnom)) / Vnom)
        ).real
        H = (
            X * np.diag(np.cos(np.angle(Vnom)) / abs(Vnom))
            - R * np.diag(np.sin(np.angle(Vnom)) / Vnom)
        ).real
        return Vnom, G, H

    def get_vnom2(self):
        _Vnom, Vnom_dict = get_vnom2(dss)
        Vnom = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        for voltage_name in Vnom_dict.keys():
            Vnom[self._name_index_dict[voltage_name]] = Vnom_dict[voltage_name]
        # Vnom(1: 3) = [];
        Vnom = np.concatenate(
            (Vnom[: self._source_indexes[0]], Vnom[self._source_indexes[-1] + 1 :])
        )
        return Vnom

    def get_vnom(self):
        _Vnom, Vnom_dict = get_vnom(dss)
        Vnom = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        # print([name_voltage_dict.keys()][:5])
        for voltage_name in Vnom_dict.keys():
            Vnom[self._name_index_dict[voltage_name]] = Vnom_dict[voltage_name]
        # Vnom(1: 3) = [];
        logger.debug(Vnom[self._source_indexes[0] : self._source_indexes[-1]])
        Vnom = np.concatenate(
            (Vnom[: self._source_indexes[0]], Vnom[self._source_indexes[-1] + 1 :])
        )
        Vnom = np.abs(Vnom)
        return Vnom

    def get_PQs_load(self, static=False):
        num_nodes = len(self._name_index_dict.keys())

        PQ_names = self._AllNodeNames
        PQ_load = np.zeros((num_nodes), dtype=np.complex_)
        for ld in get_loads(dss, self._circuit):
            self._circuit.SetActiveElement("Load." + ld["name"])
            for ii in range(len(ld["phases"])):
                name = ld["bus1"].upper() + "." + ld["phases"][ii]
                index = self._name_index_dict[name]
                if static:
                    power = complex(ld["kW"], ld["kVar"])
                    PQ_load[index] += power / len(ld["phases"])
                else:
                    power = dss.CktElement.Powers()
                    PQ_load[index] += complex(power[2 * ii], power[2 * ii + 1])
                assert PQ_names[index] == name

        return xr.DataArray(PQ_load, {"bus": PQ_names})


    def get_PQs_pv(self, static=False):
        num_nodes = len(self._name_index_dict.keys())

        PQ_names = self._AllNodeNames
        PQ_PV = np.zeros((num_nodes), dtype=np.complex_)
        for PV in get_pvSystems(dss):
            bus = PV["bus"].split(".")
            if len(bus) == 1:
                bus = bus + ["1", "2", "3"]
            self._circuit.SetActiveElement("PVSystem." + PV["name"])
            for ii in range(len(bus) - 1):
                name = bus[0].upper() + "." + bus[ii + 1]
                index = self._name_index_dict[name]
                if static:
                    power = complex(
                        -1 * PV["kW"], -1 * PV["kVar"]
                    )  # -1 because injecting
                    PQ_PV[index] += power / (len(bus) - 1)
                else:
                    power = dss.CktElement.Powers()
                    PQ_PV[index] += complex(power[2 * ii], power[2 * ii + 1])
                assert PQ_names[index] == name
        return xr.DataArray(PQ_PV, {"bus": PQ_names})

    def get_PQs_gen(self, static=False):
        num_nodes = len(self._name_index_dict.keys())

        PQ_names = self._AllNodeNames
        PQ_gen = np.zeros((num_nodes), dtype=np.complex_)
        for PV in get_Generator(dss):
            bus = PV["bus"]
            self._circuit.SetActiveElement("Generator." + PV["name"])
            for ii in range(len(bus) - 1):
                name = bus[0].upper() + "." + bus[ii + 1]
                index = self._name_index_dict[name]
                if static:
                    power = complex(
                        -1 * PV["kW"], -1 * PV["kVar"]
                    )  # -1 because injecting
                    PQ_gen[index] += power / (len(bus) - 1)
                else:
                    power = dss.CktElement.Powers()
                    PQ_gen[index] += complex(power[2 * ii], power[2 * ii + 1])
                assert PQ_names[index] == name
        return xr.DataArray(PQ_gen, {"bus": PQ_names})

    def get_PQs_cap(self, static=False):
        num_nodes = len(self._name_index_dict.keys())

        PQ_names = self._AllNodeNames
        PQ_cap = np.zeros((num_nodes), dtype=np.complex_)
        for cap in get_capacitors(dss):
            for ii in range(cap["numPhases"]):
                name = cap["busname"].upper() + "." + cap["busphase"][ii]
                index = self._name_index_dict[name]
                if static:
                    power = complex(
                        0, -1 * cap["kVar"]
                    )  # -1 because it's injected into the grid
                    PQ_cap[index] += power / cap["numPhases"]
                else:
                    PQ_cap[index] = complex(0, cap["power"][2 * ii + 1])
                assert PQ_names[index] == name

        return xr.DataArray(PQ_cap, {"bus": PQ_names})

    def get_loads(self):
        loads = get_loads(dss, self._circuit)
        self._load_power = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        load_names = []
        load_powers = []
        load = loads[0]
        for load in loads:
            for phase in load["phases"]:
                self._load_power[
                    self._name_index_dict[load["bus1"].upper() + "." + phase]
                ] = complex(load["power"][0], load["power"][1])
                load_names.append(load["bus1"].upper() + "." + phase)
                load_powers.append(complex(load["power"][0], load["power"][1]))
        return self._load_power, load_names, load_powers

    def get_voltages_actual(self):
        """

        :return voltages in actual values:
        """
        _, name_voltage_dict = get_voltages(self._circuit)
        res_feeder_voltages = np.zeros((len(self._AllNodeNames)), dtype=np.complex_)
        for voltage_name in name_voltage_dict.keys():
            res_feeder_voltages[
                self._name_index_dict[voltage_name]
            ] = name_voltage_dict[voltage_name]

        return xr.DataArray(res_feeder_voltages, {"bus": list(name_voltage_dict.keys())})

    def change_obj(self, change_commands: CommandList):
        """set/get an object property.

        Input: objData should be a list of lists of the format,

        objName,objProp,objVal,flg],...]
        objName -- name of the object.
        objProp -- name of the property.
        objVal -- val of the property. If flg is set as 'get', then objVal is not used.
        flg -- Can be 'set' or 'get'

        P.S. In the case of 'get' use a value of 'None' for objVal. The same object i.e.
        objData that was passed in as input will have the result i.e. objVal will be
        updated from 'None' to the actual value.
        Sample call: self._changeObj([['PVsystem.pv1','kVAr',25,'set']])
        self._changeObj([['PVsystem.pv1','kVAr','None','get']])
        """

        for entry in change_commands.__root__:
            dss.Circuit.SetActiveElement(
                entry.obj_name
            )  # make the required element as active element
            dss.CktElement.Properties(entry.obj_property).Val = entry.val

    def run_command(self, cmd):
        dss.run_command(cmd)

    def solve(self, hour, second):
        dss.run_command(
            f"set mode=yearly loadmult=1 number=1 hour={hour} sec={second} stepsize={self._simulation_time_step} "
        )
        dss.run_command("solve")
