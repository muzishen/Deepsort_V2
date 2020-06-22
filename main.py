from prepare import prepare_data
from utils.parser import get_config
from post_process import post_process
import time
import cv2

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
#     parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--data_path", type=str, default="./dataset/data")
    parser.add_argument("--result_path", type=str, default="./post_process")
    parser.add_argument("--config_detection", type=str, default="./configs/mmdetection.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()
    
if __name__ == "__main__":
    print('start mot')
    
    start_time = time.time()
    args = parse_args()

    prepare_data(args.data_path)
    tracks = os.listdir(r'./dataset/test-c')
    for track in tracks:
        track_data_folder = os.path.join(r'./dataset/test-c', track, 'img1')
        im = cv2.imread(os.path.join(track_data_folder, '00000.jpg'))
        i_h, i_w,_ = im.shape
        os.system('cgexec -g memory:myGroup python vis_zhongxing.py --data_path '+track_data_folder+'--track_name'+track)
        post_process(args.result_path, './output/'+track+'.txt')


    
        
    
    