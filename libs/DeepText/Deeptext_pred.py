import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from libs.DeepText.utils import CTCLabelConverter, AttnLabelConverter
from libs.DeepText.dataset import RawDataset, AlignCollate
from libs.DeepText.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Deeptext_predict(model_path, image_path):
    sensitive = False
    Prediction = 'Attn'
    character = '0123456789abcdefghijklmnopqrstuvwxyz'
    image_folder = image_path
    saved_model = model_path
    if sensitive:
        character = string.printable[:-6]
    """ model configuration """
    if 'CTC' in Prediction:
        converter = CTCLabelConverter(character)
    else:
        converter = AttnLabelConverter(character)
        num_class = len(converter.character)

    model = Model(num_class)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    model.load_state_dict(torch.load(saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=32, imgW=100, keep_ratio_with_pad=False)
    demo_data = RawDataset(root=image_folder)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=192,
        shuffle=False,
        num_workers=4,
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([25] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, 25 + 1).fill_(0).to(device)

            if 'CTC' in Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
    return pred

if __name__ == '__main__':
    image_path = './demo_image/illusion.png'
    saved_model = './TPS-ResNet-BiLSTM-Attn.pth'
    print(Deeptext_predict(saved_model, image_path))
