"""OpenDSS functions. Mutates global state, originally from GO-Solar project."""
import math


def get_loads(dss, circuit):
    """Get list of load dicts from OpenDSS circuit."""
    data = []
    load_flag = dss.Loads.First()
    dss.Circuit.SetActiveClass("Load")
    while load_flag:
        load = dss.Loads
        datum = {
            "name": load.Name(),
            "kV": load.kV(),
            "kW": load.kW(),
            "PF": load.PF(),
            "Delta_conn": load.IsDelta(),
        }
        _ = circuit.SetActiveElement("Load.%s" % datum["name"])
        cktElement = dss.CktElement
        bus = cktElement.BusNames()[0].split(".")
        datum["kVar"] = (
            float(datum["kW"])
            / float(datum["PF"])
            * math.sqrt(1 - float(datum["PF"]) * float(datum["PF"]))
        )
        datum["bus1"] = bus[0]
        datum["numPhases"] = len(bus[1:])
        datum["phases"] = bus[1:]
        if not datum["numPhases"]:
            datum["numPhases"] = 3
            datum["phases"] = ["1", "2", "3"]
        datum["voltageMag"] = cktElement.VoltagesMagAng()[0]
        datum["voltageAng"] = cktElement.VoltagesMagAng()[1]
        datum["power"] = dss.CktElement.Powers()[:2]

        data.append(datum)
        load_flag = dss.Loads.Next()

    return data


def get_pvsystems(dss):
    """Get list of PVSystem dicts from OpenDSS circuit."""
    data = []
    PV_flag = dss.PVsystems.First()
    while PV_flag:
        datum = {}
        # PVname = dss.CktElement.Name()
        PVname = dss.PVsystems.Name()
        PVpmpp = dss.PVsystems.Pmpp()
        PVkW = dss.PVsystems.kW()
        PVpf = dss.PVsystems.pf()
        PVkVARated = dss.PVsystems.kVARated()
        PVkvar = dss.PVsystems.kvar()

        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]
        # PVkV = dss.run_command('? ' + PVname + '.kV')
        # Not included in PVsystems commands for some reason

        datum["name"] = PVname
        datum["bus"] = bus
        datum["phases"] = bus[1:]
        datum["Pmpp"] = PVpmpp
        datum["pf"] = PVpf
        datum["kW"] = PVkW
        datum["kVar"] = PVkvar
        datum["kVARated"] = PVkVARated
        datum["numPhase"] = NumPhase
        datum["numPhases"] = NumPhase
        datum["power"] = dss.CktElement.Powers()[: 2 * NumPhase]

        data.append(datum)
        PV_flag = dss.PVsystems.Next()
    return data


def get_generators(dss):
    """Get list of Generator dicts from OpenDSS circuit."""
    data = []
    gen_flag = dss.Generators.First()
    dss.Circuit.SetActiveClass("Generator")
    while gen_flag:
        GENname = dss.Generators.Name()
        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]
        GENkW = dss.Generators.kW()
        GENpf = dss.Generators.PF()
        GENkV = dss.Generators.kV()
        bus = bus.split(".")
        if len(bus) == 1:
            bus = bus + ["1", "2", "3"]
        datum = {
            "name": GENname,
            "bus": bus,
            "phases": bus[1:],
            "name_bus": GENname + "." + bus[0],
            "kW": GENkW,
            "kVar": dss.Generators.kvar(),
            "pf": GENpf,
            "kV": GENkV,
            "numPhase": NumPhase,
            "numPhases": NumPhase,
        }
        data.append(datum)
        gen_flag = dss.Generators.Next()
    return data


def get_capacitors(dss):
    """Get list of Capacitor dicts from OpenDSS circuit."""
    data = []
    cap_flag = dss.Capacitors.First()
    dss.Circuit.SetActiveClass("Capacitor")
    while cap_flag:
        datum = {}
        capname = dss.CktElement.Name()
        NumPhase = dss.CktElement.NumPhases()
        bus = dss.CktElement.BusNames()[0]
        kvar = dss.Capacitors.kvar()
        datum["name"] = capname
        temp = bus.split(".")
        datum["busname"] = temp[0]
        datum["busphase"] = temp[1:]
        if not datum["busphase"]:
            datum["busphase"] = ["1", "2", "3"]
        datum["kVar"] = kvar
        datum["numPhases"] = NumPhase
        datum["power"] = dss.CktElement.Powers()[: 2 * NumPhase]
        data.append(datum)
        cap_flag = dss.Capacitors.Next()
    return data


def get_voltages(circuit):
    """Get dict of names to voltages from OpenDSS circuit."""
    temp_Vbus = circuit.YNodeVArray()
    AllNodeNames = circuit.YNodeOrder()
    node_number = len(AllNodeNames)
    name_voltage_dict = {
        AllNodeNames[ii]: complex(temp_Vbus[ii * 2], temp_Vbus[ii * 2 + 1])
        for ii in range(node_number)
    }
    return name_voltage_dict
