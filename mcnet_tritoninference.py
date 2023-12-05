import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import modules.generator as GEN
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
from collections import OrderedDict
import pdb
if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

class MCNET_tritoninference:
    
    def __init__(self,relative,cpu,find_best_frame,best_frame,adapt_scale):
        '''
        
        
        
        
        '''
        
        self.relative=relative
        self.cpu=cpu
        # self.find_best_frame=find_best_frame
        self.best_frame=best_frame
        # self.result_video=result_video
        self.adapt_scale=adapt_scale
        
   
    def make_animation(self,source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
        sources = []
        drivings = []
        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not cpu:
                source = source.cuda()
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

            kp_source = kp_detector(source)
            if not cpu:
                kp_driving_initial = kp_detector(driving[:, :, 0].cuda())
            else:
                kp_driving_initial = kp_detector(driving[:, :, 0])
            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                if not cpu:
                    driving_frame = driving_frame.cuda()
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                    use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                drivings.append(np.transpose(driving_frame.data.cpu().numpy(), [0, 2, 3, 1])[0])
                sources.append(np.transpose(source.data.cpu().numpy(), [0, 2, 3, 1])[0])
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return sources, drivings, predictions
    
    def find_best_frame(self,source, driving, cpu=False):
        import face_alignment

        def normalize_kp(kp):
            kp = kp - kp.mean(axis=0, keepdims=True)
            area = ConvexHull(kp[:, :2]).volume
            area = np.sqrt(area)
            kp[:, :2] = kp[:, :2] / area
            return kp

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True,
                                        device='cpu' if cpu else 'cuda')
        kp_source = fa.get_landmarks(255 * source)[0]
        kp_source = normalize_kp(kp_source)
        norm  = float('inf')
        frame_num = 0
        for i, image in tqdm(enumerate(driving)):
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        return frame_num
    
    
    def infer(self,source_image,driving_video,generator,kp_detector,relative,cpu,find_best_frame,best_frame,result_video,adapt_scale):
        
        source_image = imageio.imread(source_image)
        reader = imageio.get_reader(driving_video)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        
        reader.close()
        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        if find_best_frame or best_frame is not None:
            i = best_frame if best_frame is not None else self.find_best_frame(source_image, driving_video)
            print ("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i+1)][::-1]
            sources_forward, drivings_forward, predictions_forward = self.make_animation(source_image, driving_forward, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale)
            sources_backward, drivings_backward, predictions_backward = self.make_animation(source_image, driving_backward, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
            sources = sources_backward[::-1] + sources_forward[1:]
            drivings = drivings_backward[::-1] + drivings_forward[1:]
        else:
            sources, drivings, predictions = self.make_animation(source_image, driving_video, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale)
        
        # imageio.mimsave(result_video, [np.concatenate((img_as_ubyte(s),img_as_ubyte(d),img_as_ubyte(p)),1) for (s,d,p) in zip(sources, drivings, predictions)], fps=fps)
        imageio.mimsave(result_video, [img_as_ubyte(p) for p in  predictions], fps=fps)
        return result_video
        
    
    
    
    