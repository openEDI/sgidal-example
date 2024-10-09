from fastapi import FastAPI, BackgroundTasks, HTTPException
from measuring_federate import run_simulator
from fastapi.responses import JSONResponse
import traceback
import requests
import uvicorn
import socket
import logging
import sys
import json
import os

from oedisi.componentframework.system_configuration import ComponentStruct
from oedisi.types.common import ServerReply, HeathCheck, DefaultFileNames
from oedisi.types.common import BrokerConfig

sensor_logger = logging.getLogger("measuring_federate")
logger = logging.getLogger('uvicorn.error')
logger.addHandler(*sensor_logger.handlers)
logger.setLevel(logging.DEBUG)

app = FastAPI()

is_kubernetes_env = os.environ['SERVICE_NAME'] if 'SERVICE_NAME' in os.environ else None

def build_url(host:str, port:int, enpoint:list):
    if is_kubernetes_env:
        SERVICE_NAME = os.environ['SERVICE_NAME']
        url = f"http://{host}.{SERVICE_NAME}:{port}/"
    else:
        url = f"http://{host}:{port}/"
    url = url + "/".join(enpoint)
    return url 

@app.get("/")
async def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    response = HeathCheck(
        hostname = hostname,
        host_ip = host_ip
    ).dict()
    return JSONResponse(response, 200)
    
@app.post("/run")
async def run_model(broker_config:BrokerConfig, background_tasks: BackgroundTasks):
    logger.info(f"{broker_config=}")
    feeder_host = broker_config.feeder_host
    feeder_port = broker_config.feeder_port
    url = build_url(feeder_host, feeder_port, ['sensor']) 
    logger.info(url)
    try:   
        logger.info("Requesting sensor information. This might take a while")
        reply = requests.get(url)
        sensor_dict = reply.json()
        if not sensor_dict:
            msg = "empty sensor list"
            raise HTTPException(404, msg)
        logger.info(f"Sensors types available {list(sensor_dict.keys())}", )
        
        for sensor_type, sensorlist in sensor_dict.items():    
            with open(f"{sensor_type}.json", "w") as outfile:
                json.dump(sensorlist, outfile)

        background_tasks.add_task(run_simulator, broker_config)
        response = ServerReply(
            detail = f"Task sucessfully added."
        ).dict() 
        return JSONResponse(response, 200)
    except Exception as e:
        err = traceback.format_exc()
        raise HTTPException(500,str(err))

@app.post("/configure")
async def configure(component_struct:ComponentStruct): 
    component = component_struct.component
    params = component.parameters
    params["name"] = component.name
    links = {}
    for link in component_struct.links:
        links[link.target_port] = f"{link.source}/{link.source_port}"
    json.dump(links , open(DefaultFileNames.INPUT_MAPPING.value, "w"))
    json.dump(params , open(DefaultFileNames.STATIC_INPUTS.value, "w"))
    response = ServerReply(
            detail = f"Sucessfully updated configuration files."
        ).dict() 
    return JSONResponse(response, 200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ['PORT']))
