import torch
import cv2
from PIL import Image
import numpy as np
import argparse
import torchvision
import torchvision.transforms as transforms
from cycleGAN.options.test_options import TestOptions
from cycleGAN.models.test_model import TestModel
from cycleGAN.data.aligned_dataset import AlignedDataset, get_transform, get_params
# from cycleGAN.models.retina_gan_model import retinaganmodel

Generator_path = "./cycleGAN/checkpoints/frozen_net/latest_net_G_A.pth"

def true_opt():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.gpu_ids = [2]
    return opt

def make_model(opt):
    # make model, using the test model
    model = TestModel(opt)
    # caution: unset the scrict mode
    load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
    model.load_networks(load_suffix)
    # model.load_networks("latest")
    return model

# from cycleGAN.data.base_dataset import __make_power_2

opt = true_opt()
model = make_model(opt)
# dset = AlignedDataset(opt)

# prepare transform
transform_list = []

transform_list.append(transforms.ToTensor())
trans_rent = get_transform(opt, grayscale = False)

trans_to = transforms.Compose(transform_list)
trans_back = transforms.Resize([128, 128], transforms.InterpolationMode.BICUBIC)

from cycleGAN.util.util import tensor2im

def view_dillusion(img:np.ndarray):
    # the img should be [1,3,256,256]
    # in rgb format
    # img = trans_to(img).float()

    img = Image.fromarray(img)
    img = trans_rent(img)
    model.set_input({"A": img, 'A_paths': None})
    model.test()

    visuals = model.get_current_visuals()

    res = model.fake.view(1,3,opt.load_size,opt.load_size)
    # res = trans_back(res)
    res = res.detach().clone()
    res = tensor2im(res)
    return res

def server_main():
    import json
    from json import JSONEncoder
    from bottle import run, post, request, response
    from server.numpy_tcp import np2json, json2np

    @post('/dillusion')
    def dillusion_action():
        req_obj = json.loads(request.body.read())
        img = json2np(req_obj).astype('uint8')
        img = view_dillusion(img)
        img = np2json(img)
        return img

    run(host='localhost', port=1123, debug=True)


if __name__ == "__main__":
    server_main()

    # temp_path = "cycleGAN/datasets/epMine_ver4/trainA/step_0.png"
    # img = cv2.imread(temp_path).astype('uint8')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # res = view_dillusion(img)
    # cv2.imwrite("./res.png", res)
    pass