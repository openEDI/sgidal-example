from fastapi import FastAPI, BackgroundTasks, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import HTTPException
from pathlib import Path
import helics as h
import grequests
import traceback
import requests
import zipfile
import uvicorn
import logging
import socket
import shutil
import time
import yaml
import json
import sys
import os

from oedisi.componentframework.system_configuration import WiringDiagram, ComponentStruct
from oedisi.types.common import ServerReply, HeathCheck, DefaultFileNames

app = FastAPI()

is_kubernetes_env = os.environ['SERVICE_NAME'] if 'SERVICE_NAME' in os.environ else None


def build_url(host:str, port:int, enpoint:list):
    if is_kubernetes_env:
        SERVICE_NAME = os.environ['SERVICE_NAME']
        url = f"http://{host}.{SERVICE_NAME}:{port}/"
    else:
        url = f"http://{host}:{port}/"
    url = url + "/".join(enpoint) + "/" 
    return url 
    
def find_filenames(path_to_dir=os.getcwd(), suffix=".feather"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def read_settings():
    component_map = {}
    with open("docker-compose.yml", "r") as stream:
        config = yaml.safe_load(stream)
    services = config["services"]
    print(services)
    broker = services.pop("oedisi_broker")
    broker_host = broker["hostname"]

    broker_ip = socket.gethostbyname(broker_host) 
    api_port = int(broker["ports"][0].split(":")[0])

    for service in services:
        host = services[service]["hostname"]
        port = int(services[service]["ports"][0].split(":")[0])
        component_map[host] = port

    return services, component_map, broker_ip, api_port


@app.get("/")
def read_root():
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)

    response = HeathCheck(hostname=hostname, host_ip=host_ip).dict()

    return JSONResponse(response, 200)


@app.post("/profiles")
async def upload_profiles(file: UploadFile):
    try:
        services, _, _, _ = read_settings()
        for service in services:
            if "feeder" in service.lower():
                ip = services[service]["host"]
                port = int(services[service]["ports"][0].split(":")[0])
                data = file.file.read()
                if not file.filename.endswith(".zip"):
                    HTTPException(
                        400, "Invalid file type. Only zip files are accepted."
                    )
                with open(file.filename, "wb") as f:
                    f.write(data)
                    
                url = build_url(ip, port, ["profiles"])    
                logging.info(f"making a request to url - {url}")
                
                files = {"file": open(file.filename, "rb")}
                r = requests.post(url, files=files)
                response = ServerReply(detail=r.text).dict()
                return JSONResponse(response, r.status_code)
        raise HTTPException(status_code=404, detail="Unable to upload profiles")
    except Exception as e:
        err = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/model")
async def upload_model(file: UploadFile):
    try:
        services, _, _, _ = read_settings()
        for service in services:
            if "feeder" in service.lower():
                ip = services[service]["host"]
                port = int(services[service]["ports"][0].split(":")[0])
                data = file.file.read()
                if not file.filename.endswith(".zip"):
                    HTTPException(
                        400, "Invalid file type. Only zip files are accepted."
                    )
                with open(file.filename, "wb") as f:
                    f.write(data)
                    
                url = build_url(ip, port, ["model"])    
                logging.info(f"making a request to url - {url}")
       
                files = {"file": open(file.filename, "rb")}
                r = requests.post(url, files=files)
                response = ServerReply(detail=r.text).dict()
                return JSONResponse(response, r.status_code)
        raise HTTPException(status_code=404, detail="Unable to upload model")
    except Exception as e:
        err = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(err))

@app.get("/results")
def download_results():
    services, _, _, _ = read_settings()
    for service in services:
        if "recorder" in service.lower():
            host = services[service]["hostname"]
            port = int(services[service]["ports"][0].split(":")[0])
            
            url = build_url(host, port, ["download"])    
            logging.info(f"making a request to url - {url}")
            
            response = requests.get(url)
            with open(f"{service}.feather", "wb") as out_file:
                shutil.copyfileobj(response.raw, out_file)
                time.sleep(2)

    file_path = "results.zip"
    with zipfile.ZipFile(file_path, "w") as zipMe:
        for feather_file in find_filenames():
            zipMe.write(feather_file, compress_type=zipfile.ZIP_DEFLATED)

    try:
        return FileResponse(path=file_path, filename=file_path, media_type="zip")
    except Exception as e:
        raise HTTPException(status_code=404, detail="Failed download")


@app.get("/terminate")
def terminate_simulation():
    try:
        h.helicsCloseLibrary()
        return JSONResponse({"detail": "Helics broker sucessfully closed"}, 200)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Failed download ")


def run_simulation():
    services, component_map, broker_ip, api_port = read_settings()
    logging.info(f"{broker_ip}, {api_port}")
    initstring = f"-f {len(component_map)} --name=mainbroker --loglevel=trace --local_interface={broker_ip} --localport=23404"
    logging.info(f"Broker initaialization string: {initstring}")
    broker = h.helicsCreateBroker("zmq", "", initstring)
    logging.info(broker)
    isconnected = h.helicsBrokerIsConnected(broker)
    logging.info(f"Broker connected: {isconnected}")
    logging.info(str(component_map))
    replies = []
    for service_ip, service_port in component_map.items():
        
        url = build_url(service_ip, service_port, ["run"])    
        logging.info(f"making a request to url - {url}")
        
        myobj = {
            "broker_port": 23404,
            "broker_ip": broker_ip,
            "api_port": api_port,
            "services": services,
        }
        replies.append(grequests.post(url, json=myobj))
    grequests.map(replies)
    while h.helicsBrokerIsConnected(broker):
        time.sleep(1)
    h.helicsCloseLibrary()

    return


@app.post("/run")
async def run_feeder(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_simulation)
        response = ServerReply(detail="Task sucessfully added.").dict()
        return JSONResponse({"detail": response}, 200)
    except Exception as e:
        err = traceback.format_exc()
        raise HTTPException(status_code=404, detail=str(err))

    
@app.post("/configure")
async def configure(wiring_diagram:WiringDiagram): 
    json.dump(wiring_diagram.dict(), open(DefaultFileNames.WIRING_DIAGRAM.value, "w"))
    for component in wiring_diagram.components:
        component_model  = ComponentStruct(
            component = component,
            links = []
        )
        for link in wiring_diagram.links:
            if link.target == component.name:
                component_model.links.append(link)
        
        url = build_url(component.host, component.container_port, ["configure"])    
        logging.info(f"making a request to url - {url}")
                
        r = requests.post(url, json=component_model.dict())
        assert r.status_code==200, f"POST request to update configuration failed for url - {url}"
    return JSONResponse(ServerReply(detail="Sucessfully updated config files for all containers").dict(), 200)
        
if __name__ == "__main__":
    port = int(sys.argv[2])
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ['PORT']))
