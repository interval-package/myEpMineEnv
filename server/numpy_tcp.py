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

def bulr_img(img, url='http://localhost:1123/dillusion'):
    obj = np2json(img)
    r = requests.post(url, json=obj)
    res = r.text
    res = json2np(res)
    return res

if __name__ == "__main__":

    import requests
    url = 'http://localhost:1123/process'
    myobj = {'somekey': 'somevalue'}
    r = requests.post(url, json={"key": "value"})
    res = r.json()
