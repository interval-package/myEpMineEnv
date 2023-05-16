import cv2
import numpy as np
import requests
from server.numpy_tcp import np2json, json2np
url = 'http://localhost:1123/dillusion'
# myobj = {'somekey': 'somevalue'}
# r = requests.post(url, json=myobj)
# res = r.json()


if __name__ == "__main__":
    temp_path = "cycleGAN/datasets/epMine_ver4/trainA/step_0.png"
    img = cv2.imread(temp_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    obj = np2json(img)
    r = requests.post(url, json=obj)
    res = r.text
    res = json2np(res)
    cv2.imwrite("tcp.png", res)
    pass