import numpy as np
from PIL import Image
import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
import base64
from typing import List
import sys
import time
import multiprocessing
import onnxruntime as ort
from ftplib import FTP
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = multiprocessing.cpu_count()
try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
    DEVICE: str = "GPU"
except:
    print('No Nvidia GPU in system!')
    DEVICE: str = "CPU"
# DEVICE: str = ort.get_device()
model_dict_infer = {}
app = FastAPI()
CURRENT_PATH: str = os.getcwd()
FAIL_COUNT_MAX = 10
TIME_REPEAT_SEC = 60


###############################################################################################################################################################################################
def interupt_timer_ftp_download():
    path_model_ftp = os.getcwd() + "/model"
    try:
        all_model_local = os.listdir(path_model_ftp)
        print(f"all_model_in_local {all_model_local}", flush=True)
        with FTP("pca571.nseb.co.th", timeout=20) as ftp:
            ftp.encoding = "utf-8"
            ftp.login()
            ftp.cwd("AIModelTest")
            for file_name in ftp.nlst():
                path_write_file = path_model_ftp + "/" + file_name
                if file_name in all_model_local:
                    remote_datetime = ftp.voidcmd("MDTM " + file_name)[4:].strip()
                    remote_timestamp = time.mktime(
                        time.strptime(remote_datetime, "%Y%m%d%H%M%S")
                    )
                    local_timestamp = os.path.getmtime(path_write_file)
                    if int(remote_timestamp) > int(local_timestamp) or os.stat(
                        path_write_file
                    ).st_size != ftp.size(f"{file_name}"):
                        os.remove(path_write_file)
                        download_file_size = 0
                        for _ in range(FAIL_COUNT_MAX):
                            with open(path_write_file, "wb") as file:
                                ftp.retrbinary(f"RETR {file_name}", file.write)
                            download_file_size = os.stat(path_write_file).st_size
                            if download_file_size != ftp.size(f"{file_name}"):
                                os.remove(path_write_file)
                            else:
                                print(f"Download {file_name} Success", flush=True)
                                break

                elif file_name not in all_model_local:
                    download_file_size = 0
                    for _ in range(FAIL_COUNT_MAX):
                        with open(path_write_file, "wb") as file:
                            ftp.retrbinary(f"RETR {file_name}", file.write)
                        download_file_size = os.stat(path_write_file).st_size
                        if download_file_size != ftp.size(f"{file_name}"):
                            os.remove(path_write_file)
                        else:
                            print(f"Download {file_name} Success", flush=True)
                            break
    except Exception as e:
        print(e, flush=True)
    threading.Timer(TIME_REPEAT_SEC, interupt_timer_ftp_download).start()

###############################################################################################################################################################################################


class data_input_predict(BaseModel):
    image: bytes = None
    model_name: str = None


class data_input_preload_model(BaseModel):
    preload: List[str] = None


def nomalize_image(image: np.ndarray, mean: tuple, std: tuple) -> np.ndarray:
    image /= 255.0
    image -= mean
    image /= std
    return image


def decode_base64(image_base64: bytes = None) -> Image:
    return Image.open(BytesIO(base64.b64decode(image_base64)))


@app.get("/")
async def hello():
    return "Hi server is working"


@app.post("/preload_model")
def preload_model(model_name_all: data_input_preload_model) -> str:
    global model_dict_infer
    model_dict_infer.clear()
    model_dict_infer = {}
    for model_name in model_name_all.preload:
        try:
            IMAGE_SIZE: int = 0
            OUTPUT_SHAPE: List[int] = []
            MODEL_CLASS: List[str] = []
            PATH_CLASS: str = "{}/model/{}.txt".format(CURRENT_PATH, model_name)
            with open(PATH_CLASS, "r") as class_name:
                for i, label_class in enumerate(class_name.readlines()):
                    if i == 0:
                        IMAGE_SIZE: int = int(label_class.strip())
                    elif i == 1:
                        OUTPUT_SHAPE: tuple = tuple(
                            [int(data) for data in label_class.strip().split(" ")]
                        )
                    elif i == 2:
                        MEAN: tuple = tuple(
                            [float(data) for data in label_class.strip().split(" ")]
                        )
                    elif i == 3:
                        STD: tuple = tuple(
                            [float(data) for data in label_class.strip().split(" ")]
                        )
                    elif i == 4:
                        CATAGORY: str = label_class.strip()
                    else:
                        MODEL_CLASS.append(label_class.strip())

            if DEVICE.lower() == "cpu":
                providers: List[str] = ["CPUExecutionProvider"]
                PATH_MODEL: str = "{}/model/{}.onnx".format(CURRENT_PATH, model_name)
                model_dict_infer[model_name] = {
                    "MODEL": ort.InferenceSession(PATH_MODEL, providers=providers),
                    "IMAGE_SIZE": IMAGE_SIZE,
                    "OUTPUT_SHAPE": OUTPUT_SHAPE,
                    "MEAN": MEAN,
                    "STD": STD,
                    "CATAGORY": CATAGORY,
                    "MODEL_CLASS": MODEL_CLASS,
                }

            elif DEVICE.lower() == "gpu":
                providers: List[tuple] = [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                            "arena_extend_strategy": "kNextPowerOfTwo",
                            "gpu_mem_limit": 1 * 1024 * 1024 * 1024,
                            "cudnn_conv_algo_search": "EXHAUSTIVE",
                            "do_copy_in_default_stream": True,
                        },
                    )
                ]
                PATH_MODEL: str = "{}/model/{}.onnx".format(CURRENT_PATH, model_name)
                model_dict_infer[model_name] = {
                    "MODEL": ort.InferenceSession(PATH_MODEL, providers=providers),
                    "IMAGE_SIZE": IMAGE_SIZE,
                    "OUTPUT_SHAPE": OUTPUT_SHAPE,
                    "MEAN": MEAN,
                    "STD": STD,
                    "CATAGORY": CATAGORY,
                    "MODEL_CLASS": MODEL_CLASS,
                }
        except:
            return f"Fail to load {model_name}"
    return "Success"


@app.post("/predict")
async def predict(request: data_input_predict):
    global model_dict_infer
    start: float = time.time()
    try:
        image: Image = decode_base64(request.image).convert("RGB")
    except:
        return "Fail to decode_base64 image"
    image_size: tuple = image.size
    inputs: np.ndarray = np.expand_dims(
        nomalize_image(
            np.array(
                image.resize(
                    (
                        model_dict_infer[request.model_name]["IMAGE_SIZE"],
                        model_dict_infer[request.model_name]["IMAGE_SIZE"],
                    )
                )
            ).astype(np.float32),
            model_dict_infer[request.model_name]["MEAN"],
            model_dict_infer[request.model_name]["STD"],
        ).transpose([2, 0, 1]),
        axis=0,
    )
    try:
        if DEVICE.lower() == "gpu":
            x_ortvalue = ort.OrtValue.ortvalue_from_numpy(inputs, "cuda", 0)
            outputs = np.empty(
                model_dict_infer[request.model_name]["OUTPUT_SHAPE"], dtype=np.float32
            )
            y_ortvalue = ort.OrtValue.ortvalue_from_numpy(outputs, "cuda", 0)
            input_name = model_dict_infer[request.model_name]["MODEL"].get_inputs()[0].name
            output_name = (
                model_dict_infer[request.model_name]["MODEL"].get_outputs()[0].name
            )
            io_binding = model_dict_infer[request.model_name]["MODEL"].io_binding()
            io_binding.bind_ortvalue_input(input_name, x_ortvalue)
            io_binding.bind_ortvalue_output(output_name, y_ortvalue)
            model_dict_infer[request.model_name]["MODEL"].run_with_iobinding(io_binding)
            outputs: np.ndarray = y_ortvalue.numpy()

        elif DEVICE.lower() == "cpu":
            input_name: str = (
                model_dict_infer[request.model_name]["MODEL"].get_inputs()[0].name
            )
            outputs: np.ndarray = model_dict_infer[request.model_name]["MODEL"].run(
                None, {input_name: inputs}
            )[0]
    except Exception as e:
        print(e, flush=True)
        return e
    
    pred: np.ndarray = np.squeeze(np.argmax(outputs, axis=1)).astype(np.int8)
    if model_dict_infer[request.model_name]["CATAGORY"] == "segmentation":
        pil_image: Image = (
            Image.fromarray(pred)
            .resize(image_size, resample=Image.Resampling.NEAREST)
            .convert("L")
        )
        buff = BytesIO()
        pil_image.save(buff, format="JPEG")
        image_base64 = base64.b64encode(buff.getvalue())
        image_return: dict = {
            "result": image_base64.decode("utf-8"),
            "class": model_dict_infer[request.model_name]["MODEL_CLASS"],
            "catagory": model_dict_infer[request.model_name]["CATAGORY"],
        }
    else:
        image_return: dict = {
            "result": pred,
            "class": model_dict_infer[request.model_name]["MODEL_CLASS"][pred],
            "catagory": model_dict_infer[request.model_name]["CATAGORY"],
        }
    end: float = time.time()
    print(f"time inference: {end-start} sec on {DEVICE}", flush=True)
    return image_return


@app.get("/clear")
def clear() -> str:
    global model_dict_infer
    print(f"Before clear {sys.getsizeof(model_dict_infer)}", flush=True)
    model_dict_infer.clear()
    print(f"After clear {sys.getsizeof(model_dict_infer)}", flush=True)
    return "cleared all model"


if __name__ == "__main__":
    print(f"DEVICE: {DEVICE}", flush=True)
    thread = threading.Thread(target = interupt_timer_ftp_download)
    thread.start()
    uvicorn.run(app, host="0.0.0.0", port=5555)
