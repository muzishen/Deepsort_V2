import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net
from .baseline import CosineBaseline
from .mobilednetv3_stu import ft_net
class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        # state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net = ft_net(class_num=751)
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        #print(self.device)
        #state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

        model_dict = self.net.state_dict()
        pretrained_dict = {k: v for k, v in model_dict.items() if k in state_dict}
        model_dict.update(pretrained_dict)

        # self.net.load_state_dict(model_dict)
        # self.net = CosineBaseline(num_classes=11103,last_stride=1,model_path=model_path)
        # state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        # self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (32, 64)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.net.eval()


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
#             if im.shape[0] == 0 or im.shape[1] == 0:
#                 return np.zeros([size[1], size[0], 3], dtype=np.float32)
            return cv2.resize(im.astype(np.float32)/255., size)
#         print(im_crops)
#         print(self.size)
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        self.net.eval()
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()
    # def __call__(self, im_crops):
    #     #self.net.eval()
    #     inputs = self._preprocess(im_crops)
    #     #print(inputs.shape)
    #     features = torch.FloatTensor()
    #     with torch.no_grad():
    #         input_img = inputs.to(self.device)
    #         outputs = self.net(input_img)
    #         #print(outputs.shape)
    #         f1 = outputs.data.cpu()
    #         #print(f1.shape)
    #         # flip
    #         inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
    #         input_img = inputs.to('cuda')
    #         outputs = self.net(input_img)
    #         f2 = outputs.data.cpu()
    #         ff = f1 + f2
    #         #print(ff.shape)
    #         #feat = feat.div(feat.norm(p=2, dim=1, keepdim=True))
    #         #fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    #         ff = ff.div(ff.norm(p=2, dim=1, keepdim=True))
    #         features = torch.cat((features, ff), 0)
    #     return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

