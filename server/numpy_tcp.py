import numpy as np
import json
from json import JSONEncoder
import requests
# from bottle import run, post, request, response

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def np2json(numpyArrayOne):
    # Serialization
    numpyData = {"array": numpyArrayOne}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
    return encodedNumpyData

def json2np(encodedNumpyData):
    decodedArrays = json.loads(encodedNumpyData)
    finalNumpyArray = np.asarray(decodedArrays["array"])
    return finalNumpyArray

Base_url="192.168.1.13"
port = "1123"
url = 'http://'+Base_url+':'+port

def real_img(img, url=url+'/real'):
    obj = np2json(img)
    r = requests.post(url, json=obj)
    res = r.text
    res = json2np(res)
    return res

def blur_img(img, url=url+'/dillusion'):
    obj = np2json(img)
    r = requests.post(url, json=obj)
    res = r.text
    res = json2np(res)
    return res

def depth_img(img, url=url+'/depth'):
    obj = np2json(img)
    r = requests.post(url, json=obj)
    res = r.text
    res = json2np(res)
    return res

if __name__ == "__main__":
    import cv2
    temp_path = "cycleGAN/datasets/epMine_ver4/trainA/step_0.png"
    img = cv2.imread(temp_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    obj = np2json(img)
    r = requests.post(url, json=obj)
    res = r.text
    res = json2np(res)
    cv2.imwrite("tcp.png", res)
