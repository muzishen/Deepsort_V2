
import os

def post_process(save_path, track_file):
    track_res = {}
    with open(track_file, 'r') as f:
        for line in f.readlines():
            data = line.split(',')
            seq = int(data[0])
            track_id = int(data[1])
            bl = float(data[2])
            bt = float(data[3])
            bw = float(data[4])
            bh = float(data[5])
            conf = int(data[6])
            c_type = int(data[7])
            if bl  < 0 :
                bl = 0
            if bt < 0:
                bt = 0
            if track_id not in track_res:
                track_res[track_id] = [[seq, bl, bt, bw, bh, conf, c_type]]
            else:
                if int(seq) - int(track_res[track_id][-1][0]) > 1 and int(seq) - int(track_res[track_id][-1][0]) < 10:
                    print(seq, bl, bt, bw, bh)
                    print(track_res[track_id][-1])
                    det_x1 = (bl - track_res[track_id][-1][1]) / (seq - track_res[track_id][-1][0])
                    det_y1 = (bt - track_res[track_id][-1][2]) / (seq - track_res[track_id][-1][0])
                    det_x2 = (bl+bw - (track_res[track_id][-1][1]+track_res[track_id][-1][3])) / (seq - track_res[track_id][-1][0])
                    det_y2 = (bt+bh - (track_res[track_id][-1][2]+track_res[track_id][-1][4])) / (seq - track_res[track_id][-1][0])
                    det_conf = (conf - track_res[track_id][-1][5]) / (seq - track_res[track_id][-1][0])
                while int(seq) - int(track_res[track_id][-1][0]) > 1 and int(seq) - int(track_res[track_id][-1][0]) < 10:
                    x1 = track_res[track_id][-1][1]+det_x1
                    y1 = track_res[track_id][-1][2]+det_y1
                    x2 = track_res[track_id][-1][1]+track_res[track_id][-1][3]+det_x2
                    y2 = track_res[track_id][-1][2]+track_res[track_id][-1][4]+det_y2
                    print('add', [track_res[track_id][-1][0]+1, x1, y1, x2-x1, y2-y1, track_res[track_id][-1][4]+det_conf, c_type])
                    track_res[track_id].append([track_res[track_id][-1][0]+1, x1, y1, x2-x1, y2-y1, track_res[track_id][-1][4]+det_conf, c_type])
                    
                track_res[track_id].append([seq, bl, bt, bw, bh, conf, c_type])
#     print(track_res[1])
    final_res = {}
    for track_id, items in track_res.items():
        for item in items:
            if item[0] not in final_res:
                final_res[item[0]] = [[item[0], track_id, item[1], item[2], item[3], item[4], item[5], item[6]]]
            else:
                final_res[item[0]].append([item[0], track_id, item[1], item[2], item[3], item[4], item[5], item[6]])
#     print(final_res)
    with open(os.path.join(save_path, track_file.split('/')[-1]), 'w') as f:
            save_format = '{frame},{id},{x1},{y1},{w},{h},1,0\n'
            for k, v in final_res.items():
                for _v in v:
                    line = save_format.format(frame=_v[0], id=_v[1], x1=_v[2], y1=_v[3], w=_v[4], h=_v[5])
                    f.write(line)
                              
if __name__ == "__main__":
    post_process('/media/shenfei/shensdd/deep_sort_new/post_final', '/media/shenfei/shensdd/deep_sort_new/default_epoch2/Track2.txt')
    post_process('/media/shenfei/shensdd/deep_sort_new/post_final', '/media/shenfei/shensdd/deep_sort_new/default_epoch2/Track3.txt')
    post_process('/media/shenfei/shensdd/deep_sort_new/post_final', '/media/shenfei/shensdd/deep_sort_new/default_epoch2/Track6.txt')
    post_process('/media/shenfei/shensdd/deep_sort_new/post_final', '/media/shenfei/shensdd/deep_sort_new/default_epoch2/Track8.txt')
    post_process('/media/shenfei/shensdd/deep_sort_new/post_final', '/media/shenfei/shensdd/deep_sort_new/default_epoch2/Track11.txt')
    post_process('/media/shenfei/shensdd/deep_sort_new/post_final', '/media/shenfei/shensdd/deep_sort_new/default_epoch2/Track12.txt')

                
        
        
                
            
            