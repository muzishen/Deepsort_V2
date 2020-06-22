import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
import glob


def read_file(txt):
    # print(txt)
    # file = open(txt)
    # a  =file.read()
    data = np.loadtxt(txt, delimiter=',')
    # print(data)
    # print('===')
    #length = data.shape[0]
    array1 = (data[:,0:5])
    #print(array1)
    # array2 = data[:,4]
    # array3 = np.zeros(length)
    return array1

def zxywh_to_xywh(boxes_xyxy):
    if isinstance(boxes_xyxy, torch.Tensor):
        boxes_xywh = boxes_xyxy.clone()
    elif isinstance(boxes_xyxy, np.ndarray):
        boxes_xywh = boxes_xyxy.copy()
    # print(boxes_xywh[:,0])
    boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 0]+boxes_xyxy[:, 2]) / 2.
    boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 1] +boxes_xyxy[:, 3])  / 2.
    boxes_xywh[:, 2] =  boxes_xyxy[:, 2]
    boxes_xywh[:, 3] = boxes_xyxy[:, 3]

    return boxes_xywh

class detect(object):
    def __init__(self,  is_xywh=True, use_cuda=True):

        # constants
        #self.size = self.net.width, self.net.height
        #self.score_thresh = score_thresh
        #self.conf_thresh = conf_thresh

        self.use_cuda = use_cuda
        self.is_xywh = is_xywh

        #self.num_classes = self.net.num_classes
        #self.class_names = self.load_class_names(namesfile)

    def read_txt(self, height, wight ,txt_path,score_thresh ):
        #im = cv2.imread(image_path)

        #print(txt_path)

        boxes= read_file(txt_path)
        # print(boxes)
        # print(len(boxes))
        # print(txt_boxes)
        # boxes = post_process(boxes, 0, self.conf_thresh, self.nms_thresh)[0].cpu()
        boxes = boxes[boxes[:, -1] > score_thresh, :]  # bbox xmin ymin xmax ymax
        boxes = boxes[boxes[:, -2]/boxes[:, -3] < 4, :]
        boxes = boxes[boxes[:, -3]/boxes[:, -2] < 1, :]

        boxes = boxes[boxes[:, -2]< (height*0.6), :]
        boxes = boxes[boxes[:, -3]< (wight*0.5), :]

        #print('detection',len(boxes[boxes[:, -1] > 0.5, :]))
        if len(boxes)==0:
            #print('no person in this')
            bbox = torch.FloatTensor([]).reshape([0,4])
            cls_conf = torch.FloatTensor([])
            cls_ids = torch.LongTensor([])
        else:
            bbox = boxes
            if self.is_xywh:
                # bbox x y w h
                bbox = zxywh_to_xywh(bbox)
                #print(bbox)
        new_bbox = []
        new_cls_conf = []
        new_cls_ids = []
        for i in range(bbox.shape[0]):
            # print(bbox[i].shape)
            # if bbox[i][0] < 1:
            #     bbox[i][0] = 1
            # if bbox[i][1] < 1:
            #     bbox[i][1] = 1
            # if bbox[i][2] <= 1 or bbox[i][3] <= 1:
            #     continue
            # print(bbox[i])
            #print(bbox[i][:4].tolist())
            # print(bbox[i])

            new_bbox.append(bbox[i][:4].tolist())
            new_cls_conf.append(float(bbox[i][-1]))
            new_cls_ids.append(0)
        #print(np.array(new_bbox))
        return np.array(new_bbox), np.array(new_cls_conf), np.array(new_cls_ids)


class ImageTracker(object):
    def __init__(self, cfg, args, image_path, save_filename, im_width, im_height):
        self.cfg = cfg
        self.args = args
        self.image_path = image_path

        self.logger = get_logger("root")
        self.save_filename = save_filename
        self.im_width = im_width
        self.im_height = im_height
        
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        
        
        # if args.display:
        #     cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow("test", args.display_width, args.display_height)

#         if args.cam != -1:
#             print("Using webcam " + str(args.cam))
#             self.vdo = cv2.VideoCapture(args.cam)
#         else:
#             self.vdo = cv2.VideoCapture()
        self.detector = detect(is_xywh=True, use_cuda=True)
        #print(self.detector)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
#         self.class_names = self.detector.class_names


    def __enter__(self):
#         if self.args.cam != -1:
#             ret, frame = self.vdo.read()
#             assert ret, "Error: Camera error"
#             self.im_width = frame.shape[0]
#             self.im_height = frame.shape[1]

#         else:
#             assert os.path.isfile(self.video_path), "Path error"
#             self.vdo.open(self.video_path)
#             self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
#             self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             assert self.vdo.isOpened()
#         self.images = glob.glob(self.image_path+'/*.jpg')
        image_format = ['.jpg', '.jpeg', '.png', '.tif']
        self.images = sorted(glob.glob('%s/*.*' % self.image_path))
        self.images = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.images))
        #print(self.images)
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, self.save_filename+".avi")
            self.save_results_path = os.path.join(self.args.save_path, self.save_filename+'.txt')
            #print(self.save_results_path)
            # create video writer
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width,self.im_height))
            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)


    def run(self):
        results = []
        idx_frame = 0
        for img_file in self.images:
            # print(img_file)
            txt_file = img_file.split('.')[0]+ '.txt'
            # print(txt_file)
            idx_frame += 1
#             if idx_frame % self.args.frame_interval:
#                 continue

            start = time.time()
#             print(img_file)
            ori_im = cv2.imread(img_file)
            im = ori_im
            height,weight,_ = im.shape

            # print(self.detector(im))

            bbox_xywh, cls_conf, cls_ids = self.detector.read_txt(  height, weight, txt_file, args.conf_scores)
            #print(bbox_xywh)
            #print(len(bbox_xywh))
            if len(bbox_xywh) == 0:
                continue
            else:
                # select person class
                mask = cls_ids==0
    #             print(bbox_xywh)
                bbox_xywh = bbox_xywh[mask]
               # print(len(bbox_xywh))
                # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                #bbox_xywh[:,3:] *= 1.05
                cls_conf = cls_conf[mask]
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]
                    #ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                    results.append((idx_frame-2, bbox_tlwh, identities))

                # end = time.time()

                # if self.args.display:
                #     cv2.imshow("test", ori_im)
                #     cv2.waitKey(1)

                # if self.args.save_path:
                #     self.writer.write(ori_im)
                # if self.args.display:
                #     cv2.imshow("test", ori_im)
                #     cv2.waitKey(1)
                # save results
                write_results(self.save_results_path, results, 'mot')

                # logging
                # self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                #                 .format(end-start, 1/(end-start), bbox_xywh.shape[0], len(outputs)))

            

def parse_args():
    parser = argparse.ArgumentParser()
#     parser.add_argument("VIDEO_PATH", type=str)
    #parser.add_argument("--config_detection", type=str, default="./configs/rcnn.yaml")
    parser.add_argument("--data_path", type=str, default="./dataset/data")
    parser.add_argument("--track_name", type=str, default="Track")

    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./outputs0.0/")
    parser.add_argument("--conf_scores", type=float, default=0.5)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__=="__main__":
    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = parse_args()
    cfg = get_config()
    #cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    with ImageTracker(cfg, args, args.track_data, args.track_name, args.display_width, args.display_height) as img_trk:
        img_trk.run()
    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_pytorch-master/detector/Pedestron/datasets/zhongxing/test/Track1/img1', 'Track1', 1550, 734) as img_trk:
    #     img_trk.run()

    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_pytorch-master/detector/Pedestron/datasets/zhongxing/test/Track4/img1', 'Track4', 1920, 980) as img_trk:
    #    img_trk.run()

    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_pytorch-master/detector/Pedestron/datasets/zhongxing/test/Track5/img1', 'Track5', 1400, 559) as img_trk:
    #      img_trk.run()
    #
    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_pytorch-master/detector/Pedestron/datasets/zhongxing/test/Track9/img1', 'Track9', 1116, 874) as img_trk:
    #     img_trk.run()
    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_pytorch-master/detector/Pedestron/datasets/zhongxing/test/Track10/img1', 'Track10', 615, 593) as img_trk:
    #     img_trk.run()


    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_new/detector/Pedestron/datasets/zhongxing/testB/Track2/new_img1', 'Track2', 1550, 734) as img_trk:
    #     img_trk.run()
    # #
    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_new/detector/Pedestron/datasets/zhongxing/testB/Track3/new_img1', 'Track3', 1116, 874) as img_trk:
    #     img_trk.run()
    #
    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_new/detector/Pedestron/datasets/zhongxing/testB/Track6/new_img1', 'Track6', 1400, 559) as img_trk:
    #     img_trk.run()
    # #
    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_new/detector/Pedestron/datasets/zhongxing/testB/Track8/new_img1', 'Track8', 928, 620) as img_trk:
    #     img_trk.run()
    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_new/detector/Pedestron/datasets/zhongxing/testB/Track11/new_img1', 'Track11', 615, 593) as img_trk:
    #     img_trk.run()
    # # #

    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_new/detector/Pedestron/datasets/zhongxing/testB/Track12/new_img1', 'Track12', 1728, 824) as img_trk:
    #     img_trk.run()
    # end = time.time() - start_time
    # print(end)