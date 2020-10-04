from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from random import choice
import torch
import PIL 


class DataLoader(Dataset):
    def __init__(self, img_dir, tri_dir, alpha_dir):
        self.img_path = img_dir
        self.tri_path = tri_dir
        self.alpha_path = alpha_dir
        self.imgs = list(os.listdir(img_dir))
        self.tris = list(os.listdir(tri_dir))

    def __len__(self):
        return len(self.imgs)
    
    # Load the image
    # 1. Crop it in 320x320 (some images are less in size than 640)
    # 2. Similarly for Trimap
    # 3. Get Foreground and Background using alpha
    # 4. Mark image as done
    # 5. yeild image

    def _crop_image(self,sample, size = 320):
        image, trimap, fg, bg,al = sample['image'], sample['trimap'], sample['fg'], sample['bg'], sample['alpha']
        h,w,c = image.shape
        dim = min(h,w)
        pos = np.random.randint(0, dim-size , 1)[0]
        cropped_img = image[pos:pos+size, pos:pos+size]
        cropped_tri = trimap[pos:pos+size, pos:pos+size]
        cropped_fg = fg[pos:pos+size, pos:pos+size]
        cropped_bg = bg[pos:pos+size, pos:pos+size]
        cropped_al = al[pos:pos+size, pos:pos+size]
        cropped_al = np.expand_dims(cropped_al, 2)
        output = np.concatenate((cropped_al, cropped_fg, cropped_bg), axis=2)
        output = np.transpose(output, (2,1,0))
        return  {"image" : cropped_img, "trimap" : cropped_tri, "output" : output}

    def _get_fg(self, img, alpha):
        h, w, c = img.shape
        # After loading - >
        # First get FG from Image
        fg = np.zeros((h,w,c), dtype=np.uint8)
        bg = np.zeros((h,w,c), dtype=np.uint8)
        for x in range(h):
            for y in range(w):
                al = alpha[x][y]
                if al>=1:
                    fg[x][y][0] = int(img[x][y][0]*255.0)
                    fg[x][y][1] = int(img[x][y][1]*255.0)
                    fg[x][y][2] = int(img[x][y][2]*255.0)
                elif al>0:
                    fg[x][y][0] = int(al * img[x][y][0]*255.0)
                    fg[x][y][1] = int(al * img[x][y][1]*255.0)
                    fg[x][y][2] = int(al * img[x][y][2]*255.0)
                    bg[x][y][0] = int((1-al) * img[x][y][0]*255.0)
                    bg[x][y][1] = int((1-al) * img[x][y][1]*255.0)
                    bg[x][y][2] = int((1-al) * img[x][y][2]*255.0)
                else:
                    bg[x][y][0] = int(img[x][y][0]*255.0)
                    bg[x][y][1] = int(img[x][y][1]*255.0)
                    bg[x][y][2] = int(img[x][y][2]*255.0)
        return fg,bg


    def __getitem__(self, idx):
        random_img = choice(self.imgs)
        self.imgs.remove(random_img)
        image = (cv2.imread(os.path.join(self.img_path, random_img))/255.0)[:, :, ::-1]
        trimap_im = cv2.imread(os.path.join(self.tri_path, random_img),0) / 255.0
        alpha = cv2.imread(os.path.join(self.alpha_path, random_img),0) / 255.0
        h,w = trimap_im.shape
        trimap = np.zeros((h, w, 2))
        trimap[trimap_im == 1, 1] = 1
        trimap[trimap_im == 0, 0] = 1
        fg, bg = self._get_fg(image, alpha)
        sample = {"image" : image, "trimap" : trimap, "fg" : fg, "bg": bg, "alpha" : alpha}
        # Make function to get outputs also
        return self._crop_image(sample)
        

