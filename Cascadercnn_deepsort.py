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

class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda, detector_name='EfficientDet')
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
#         self.class_names = self.detector.class_names


    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

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
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#             print(im)
            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            
#             print(bbox_xywh, cls_conf)

            # select person class
            mask = cls_ids==0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:,3:] *= 1.2 
            cls_conf = cls_conf[mask]
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:,:4]
                identities = outputs[:,-1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame-1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                            .format(end-start, 1/(end-start), bbox_xywh.shape[0], len(outputs)))


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
        
        
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

#         if args.cam != -1:
#             print("Using webcam " + str(args.cam))
#             self.vdo = cv2.VideoCapture(args.cam)
#         else:
#             self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda, detector_name='Rcnn')
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
            print(img_file)
            idx_frame += 1
#             if idx_frame % self.args.frame_interval:
#                 continue

            start = time.time()
#             print(img_file)
            ori_im = cv2.imread(img_file)
            im = ori_im
#             im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
#             im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#             print(im)
            # do detection
            #print(self.detector(im))
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            #print(bbox_xywh)

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
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame-2, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)
            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)
            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                            .format(end-start, 1/(end-start), bbox_xywh.shape[0], len(outputs)))
            
            

def parse_args():
    parser = argparse.ArgumentParser()
#     parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/rcnn.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./outputs/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    
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


    with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_new/detector/Pedestron/datasets/zhongxing/testB/Track2/new_img1', 'Track2', 1550, 734) as img_trk:
        img_trk.run()
    #
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
    # #
    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/deep_sort_new/detector/Pedestron/datasets/zhongxing/testB/Track12/new_img1', 'Track12', 1728, 824) as img_trk:
    #     img_trk.run()
    # with ImageTracker(cfg, args, r'/media/shenfei/shensdd/10millon_data/test/11_Train_Station_Square', '11_Train_Station_Square', 26583, 14957) as img_trk:
    #     img_trk.run()