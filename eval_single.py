import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from pretrainedmodels import inceptionv4
import pretrainedmodels.utils as utils
import NLUtils
import numpy as np
import subprocess
import glob
import shutil

def extract_image_feats(video_path):
    print('Mengekstrak fitur dari images...')
    model = inceptionv4(pretrained='imagenet')
    model = model.cuda()
    model.last_linear = utils.Identity()
    model.eval()
    C, H, W = 3, 299, 299
    load_image_fn = utils.LoadTransformImage(model)
    dst = os.path.join(video_path.split('\\')[0], 'info')
    if os.path.exists(dst):
            print(" Menghapus Direktori: " + dst + "\\")
            shutil.rmtree(dst)
    os.makedirs(dst)
    with open(os.devnull, "w") as ffmpeg_log:
        command = 'ffmpeg -i ' + video_path + ' -vf scale=400:300 ' + '-qscale:v 2 '+ '{0}/%06d.jpg'.format(dst)
        subprocess.call(command, shell=True, stdout=ffmpeg_log, stderr=ffmpeg_log)
    list_image = sorted(glob.glob(os.path.join(dst, '*.jpg')))
    samples = np.round(np.linspace(0, len(list_image) - 1, 80))
    list_image = [list_image[int(sample)] for sample in samples]
    images = torch.zeros((len(list_image), C, H, W))
    for i in range(len(list_image)):
        img = load_image_fn(list_image[i])
        images[i] = img
    with torch.no_grad():
        image_feats = model(images.cuda().squeeze())
    image_feats = image_feats.cpu().numpy()
    for file in os.listdir(dst):
        if file.endswith('.jpg'):
            os.remove(os.path.join(dst, file))

    return image_feats


def main(opt):
    video_path = opt["video_path"]

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    image_feats = extract_image_feats(video_path)
    image_feats = torch.from_numpy(image_feats).type(torch.FloatTensor).unsqueeze(0)

    encoder = EncoderRNN(opt["dim_vid"], opt["dim_hidden"], bidirectional=bool(opt["bidirectional"]),
                             input_dropout_p=opt["input_dropout_p"], rnn_dropout_p=opt["rnn_dropout_p"])
    decoder = DecoderRNN(16860, opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                            input_dropout_p=opt["input_dropout_p"],
                            rnn_dropout_p=opt["rnn_dropout_p"], bidirectional=bool(opt["bidirectional"]))
    model = S2VTAttModel(encoder, decoder).cuda()

    model.load_state_dict(torch.load(opt["saved_model"]))
    model.eval()
    opt = dict()
    opt['child_sum'] = True
    opt['temporal_attention'] = True
    opt['multimodel_attention'] = True
    with torch.no_grad():
        _, seq_preds = model(image_feats.cuda(), mode='inference', opt=opt)
    vocab = json.load(open('data/info.json'))['ix_to_word']
    sent = NLUtils.decode_sequence(vocab, seq_preds)
    print(sent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recover_opt', type=str, default='data/save/opt_info.json',
                        help='Lokasi opt_info.json berada')
    parser.add_argument('--saved_model', type=str, default='data/save/model_500.pth',
                        help='Lokasi model berada')
    parser.add_argument('--video_path', type=str, default='tester/vidGue.mkv',required=False,help='Lokasi video yang akan di prediksi')

    args = parser.parse_args()
    args = vars((args))
    opt = json.load(open(args["recover_opt"]))
    for k, v in args.items():
        opt[k] = v

    main(opt)
