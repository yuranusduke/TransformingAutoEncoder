"""
Utilities functions

Created by Kunhong Yu
Date: 2021/07/01
"""
import torch as t
from cv2 import warpAffine
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

#################
#     Config    #
#################
class Config(object):
    """
    Args :
        --cap_dim: capsule dimension, default is 120
        --out_dim: output dimension, default is 300
        --num_caps: number of capsules, default is 7
        --epochs: training epochs, default is 20
        --batch_size: default is 100
    """
    cap_dim = 120
    out_dim = 300
    num_caps = 7

    epochs = 20
    batch_size = 100

    device = 'cuda' if t.cuda.is_available() else 'cpu'

    def parse(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                print(k + ' does not exist, will be added!')

            setattr(self, k, v)


##################
#    Transform   #
##################
def shift(img : np.ndarray, dx : int, dy : int) -> np.ndarray:
    """Shift one image
    Args :
        --img: input numpy image
        --dx: shift in x
        --dy: shift in y
    return :
        --rimg: rotated image
    """
    img = np.transpose(img, (1, 2, 0))
    r, c, _ = img.shape
    trans = np.array([[1, 0, dx], [0, 1, dy]]).astype(np.float32)
    wimg = warpAffine(img, trans, (r, c)).reshape(r, c, 1)
    rimg = np.transpose(wimg, (2, 0, 1))

    return rimg

def BatchShift(imbatch : np.ndarray, dxdy = [-4, 4]):
    """Shift one batch of images
    Args :
        --imbatch: batch images
        --dxdy: shift in both x and y, default is [-4, 4]
    return :
        --imbatch: transformed image batch
        --R: random chosen residual
    """
    dim = imbatch.shape

    R = np.random.randint(low = dxdy[0], high = dxdy[1], size = (dim[0], 2))
    for i in range(dim[0]):
        imbatch[i : i + 1] = shift(imbatch[i : i + 1], R[i][0], R[i][1])

    return imbatch, R


##################
#    Visualize   #
##################
def show_batch(batch, index = 0):
    """Show batch of images
    Args :
        --battch: batch image in numpy form
        --index: index of batch, default is 0
    """
    f, ax = plt.subplots(len(batch), 1, figsize = (5, 20))
    f.suptitle(f'Batch {index + 1} \n Input || Recon || Target', fontsize = 30)
    for i in range(len(batch)):
        ax[i].imshow(batch[i], cmap = 'gray')
        ax[i].axis('off')

    plt.savefig(f'./results/re_imgs/re_img_{index + 1}.png')
    plt.close()

def show_weights(weights):
    """Show weights of one layer
    Args :
        --weights: weights of one layer
    """
    plt.figure(figsize = (20, 5))
    plt.imshow(weights, cmap = 'gray_r')
    plt.axis('off')
    plt.savefig(f'./results/weights_of_generation.png')
    plt.close()

def generate_animnated():
    """
    Generate one animated image
    """
    path = os.path.join('./results/re_imgs')
    files = os.listdir(path)

    plt.ion()

    for file in files[:30]:
        plt.cla()
        img = Image.open(os.path.join(path, file))
        img = np.asarray(img)
        plt.imshow(img, cmap = 'gray')
        plt.axis('off')
        plt.pause(0.6)

    plt.ioff()
    plt.close()


if __name__ == '__main__':
    generate_animnated()