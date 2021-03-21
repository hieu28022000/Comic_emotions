import torch
from collections import OrderedDict
from libs.CRAFT.predict import test_net
import os
import cv2
from libs.CRAFT import file_utils

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def craft_text_detect(image, config, net):    
    print('Loading weights CRAFT from checkpoint (' + config.TRAINED_MODEL + ')')
    if config.CUDA:
        net.load_state_dict(copyStateDict(torch.load(config.TRAINED_MODEL)))
    else:
        net.load_state_dict(copyStateDict(torch.load(config.TRAINED_MODEL, map_location='cpu')))

    if config.CUDA:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if config.REFINE:
        from libs.CRAFT.refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + config.refiner_model + ')')
        if config.CUDA:
            refine_net.load_state_dict(copyStateDict(torch.load(config.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(config.refiner_model, map_location='cpu')))

        refine_net.eval()
        config.POLY = True

    bboxes, polys, score_text = test_net(net, image, config.TEXT_THRESHOLD, config.LINK_THRESHOLD, config.LOW_TEST, config.CUDA, config.POLY, refine_net, config)
    
    # result_folder = './result'
    # image_path = './images/123.jpg'
    # # save score text
    # filename, file_ext = os.path.splitext(os.path.basename(image_path))
    # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    # cv2.imwrite(mask_file, score_text)

    # file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    return  bboxes, polys, score_text

