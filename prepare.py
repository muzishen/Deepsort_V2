import os
import shutil

def prepare_data(test_root = r'./dataset/data', save_path = r'./dataset/test-c'):
  tracks = os.listdir(test_root)
  for track in tracks:
      os.mkdir(os.path.join(save_path, track))
      os.mkdir(os.path.join(save_path, track, 'img1'))
      imgs = os.listdir(os.path.join(test_root, track))
      
      for img in imgs:
          if img.endswith('txt'):
              continue
          img_name = img.split('.')[0]
          new_img_name = '%05d' % int(img_name)
          print(new_img_name)
          if new_img_name == '00001':
              shutil.copy(os.path.join(test_root, track, img), os.path.join(save_path, track, 'img1',  '000000.jpg'))
              shutil.copy(os.path.join(test_root, track, img.replace('jpg', 'txt')), os.path.join(save_path, track, 'img1',  '000000.txt'))
              shutil.copy(os.path.join(test_root, track, img), os.path.join(save_path, track, 'img1',  '00000.jpg'))
              shutil.copy(os.path.join(test_root, track, img.replace('jpg', 'txt')), os.path.join(save_path, track, 'img1',  '00000.txt'))
              shutil.copy(os.path.join(test_root, track, img), os.path.join(save_path, track,'img1',  '00001.jpg'))
              shutil.copy(os.path.join(test_root, track, img.replace('jpg', 'txt')), os.path.join(save_path, track, 'img1',  '00001.txt'))
          else:
              shutil.copy(os.path.join(test_root, track, img), os.path.join(save_path, track,'img1',  new_img_name+'.jpg'))
              shutil.copy(os.path.join(test_root, track, img.replace('jpg', 'txt')), os.path.join(save_path, track,'img1',  new_img_name+'.txt'))




# test_root = r'/storage/dataset/test-a'
# tracks = os.listdir(test_root)
# for track in tracks:
#     shutil.copy(os.path.join(test_root, track, 'img1',  '00000.jpg'), os.path.join(test_root, track, 'img1',  '000000.jpg'))