import numpy as np
from PIL import Image

from configs import *
    
class Obs_Processor():
    def __init__(self, extra_channels: int = 0) -> None:
        self.downsample_rate = 4
        self.rgb_channels = 3
        self.n_channels = self.rgb_channels + int(extra_channels)

    def _resize(self, img, resample):
        resize_shape = (
            img.shape[0] // self.downsample_rate,
            img.shape[1] // self.downsample_rate,
        )
        return np.asarray(Image.fromarray(img).resize(resize_shape, resample))

    def process_img(self, img):
        processed_img = self.change_bg_color(img)
        processed_img = self._resize(processed_img, Image.Resampling.BILINEAR)
        processed_img = processed_img/255.0

        return processed_img

    def process_mask(self, mask):
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim != 2:
            raise ValueError("Expected a 2D binary mask")
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        resized_mask = self._resize(mask, Image.Resampling.NEAREST)
        resized_mask = (resized_mask > 0).astype(np.float32)
        return resized_mask[..., None]

    def change_bg_color(self, img):
        processed_img = img.copy()
        bg_pos = img==BG_COLOR[:3]
        bg_pos = (np.sum(bg_pos,axis=-1) == 3)
        processed_img[bg_pos] = (0,0,0)
        return processed_img
    
