# from .YOLOv3 import YOLOv3
#from .EfficientDet import EfficientDet
from .Pedestron.detector import Rcnn


__all__ = ['build_detector']

def build_detector(cfg, use_cuda, detector_name='Rcnn'):
    if detector_name == 'Rcnn':
        #print(cfg)
        return Rcnn(cfg.Rcnn.CONFIG, cfg.Rcnn.WEIGHT,score_thresh=cfg.Rcnn.SCORE_THRESH,
                    is_xywh=True, use_cuda=use_cuda)
    # elif detector_name == 'YOLOv3':
    #     return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES,
    #                 score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH,
    #                 is_xywh=True, use_cuda=use_cuda)
