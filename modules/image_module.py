import cv2
import tifffile
import numpy as np
import pandas as pd
import time

from PIL import Image, ImageFilter

class ImageModule:
    
    data_dirs=[]
    regions=[]
    input_img_size=256
    random_distortion=True
    total_df =pd.DataFrame()
    gt_key=""
    img_key=""
    n_resize=1
    strategy_name=""
    custom_prefix=""
    training_time=0.0
    pre_process = False

    def __init__(self, _sm):
        self.data_dirs = _sm.data_dirs
        self.regions = _sm.regions
        self.input_img_size = _sm.input_img_size
        self.random_distortion = _sm.random_distortion
        self.total_df = _sm.total_df
        self.gt_key = _sm.gt_key
        self.img_key = _sm.img_key
        self.n_resize = _sm.n_resize
        self.strategy_name = _sm.strategy_name
        self.custom_prefix = _sm.custom_prefix
        
    def filterUnsharp(self, img, unsharp_radius=800, unsharp_percent=150):
        
        img_array = Image.fromarray(img.astype('uint8'))
        unsharp = img_array.filter(ImageFilter.UnsharpMask(radius=unsharp_radius, percent=unsharp_percent))
        
        res_cv2 = np.array(unsharp)
        res = res_cv2
        
        return res
    
    def filterBlur(self, img, kernel=11):
        
        res = cv2.blur(img, ksize = (kernel, kernel))
        
        return res
    
    def filterDilate(self, img, kernel=2, iters=3):
        
        kernel_dilate = np.ones((kernel, kernel), np.uint8)
        res = cv2.dilate(img, kernel_dilate, iterations = iters)
        
        return res
    
    def filterMeanDenoise(self, img):
        
        res = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        
        return res
        
    def preProcessing(self, img, unsharp_radius=800, unsharp_percent=150):
        
        res = self.filterUnsharp(img, unsharp_radius, unsharp_percent)

        return res

    def showInputImageSize(self):
        print("Input Image Size :",self.input_img_size)
        
    def loadFrame(self, nID):
        
        img_id = self.total_df["ID"][nID]
        region_id = self.total_df["Region"][nID]
        
        img_filepath = self.data_dirs[region_id]+'/{}'.format(img_id)+'.png'
        img = cv2.imread(img_filepath)
        
        return img

    def loadSingleIMG(self, nID):
        
        frame = self.loadFrame(nID)
        img = frame[:,0:600,:]
    
        return img

    def loadSingleGT(self, nID):
        
        frame = self.loadFrame(nID)
        mask = frame[:,600:1200,:]
        
        blue = mask[:,:,2]
        blue = blue + 1
        blue[blue>1]=0
        mask = blue

        return mask

    def loadImageSet(self, nID):
        
        frame = self.loadFrame(nID)
        
        ## Image
        img = frame[:,0:600,:]
        
        ## GT
        mask = frame[:,600:1200,:]
        blue = mask[:,:,2]
        blue = blue + 1
        blue[blue>1]=0
        mask = blue

        assert img.shape[:2] == mask.shape, 'The image and GT have different image sizes'

        return img, mask

    def resizeImage(self, input_img, width, height):

        dim = (width, height)
        img = cv2.resize(input_img, dim)

        return img

    def cropSingle(self, input_img, x, y, window_size):

        crop_img = input_img[y:y+window_size, x:x+window_size].copy()

        return crop_img

    def cropImageSet(self, img, mask, x, y, window_size):

        crop_img  =  self.cropSingle(img,  x, y, window_size)
        crop_mask =  self.cropSingle(mask, x, y, window_size)

        return crop_img, crop_mask

    def randomHueSaturationValue(self, image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
        if np.random.random() < u:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image

    def randomShiftScaleRotate(self, image, mask,
                               shift_limit=(-0.0625, 0.0625),
                               scale_limit=(-0.1, 0.1),
                               rotate_limit=(-45, 45), aspect_limit=(0, 0),
                               borderMode=cv2.BORDER_CONSTANT, u=0.5):
        if np.random.random() < u:
            height, width, channel = image.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(
                                           0, 0,
                                           0,))

        return image, mask


    def randomHorizontalFlip(self, image, mask, u=0.5):
        if np.random.random() < u:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        return image, mask

    def randomDistortion(self, img, mask):
        img = self.randomHueSaturationValue(img,
                                       hue_shift_limit=(-50, 50),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))
        img, mask = self.randomShiftScaleRotate(img, mask,
                                           shift_limit=(-0.0625, 0.0625),
                                           scale_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))
        img, mask = self.randomHorizontalFlip(img, mask)

        return img, mask

    def chooseCroppedRegions(self, nID, coord_x, coord_y, width, height):
    
        ## Load Images
        img, mask = self.loadImageSet(nID)

        ## Resize Images
        img  = self.resizeImage(img, width, height)
        mask = self.resizeImage(mask, width, height)

        ## Crop Image
        crop_img, crop_mask= self.cropImageSet(img, mask, coord_x, coord_y, self.input_img_size)

        ## Edge Enhancement
        if (self.pre_process):
            crop_img = self.preProcessing(crop_img)

        
        ## Random distortion
        if self.random_distortion :
            crop_img, crop_mask = self.randomDistortion(crop_img, crop_mask)

        ## Make a channel match the dimension
        crop_mask = np.expand_dims(crop_mask, axis=2)
    
        return crop_img, crop_mask
    
    def searchXYfromResizedIMG(self, crop_id, n_height, window_size):
        
        i = crop_id % n_height
        j = crop_id // n_height
        x = i * window_size
        y = j * window_size
    
        return x,y

    def searchXYfromRawIMG(self, crop_id, n_height, window_size, width, height):
    
        x,y = searchXYResized(crop_id, n_height+1, window_size)
    
        if ( width - x <= window_size + 1 ):
            x = width - window_size - 1

        if ( height - y <= window_size +1 ):
            y = height - window_size -1
    
        return x,y
    
    def getImageSize(self, mask, window_size):

        height, width = mask.shape
        n_height = int(np.floor(height/window_size))
        n_width = int(np.floor(width/window_size))

        return n_height, n_width, height, width
    
    def predictSingleImage(self, model, nID, window_size, doResize=True, verbose=False):
        
        start_time = time.time()
        
        img, mask = self.loadImageSet(nID)
        
        n_height, n_width, height, width = self.getImageSize(mask, window_size)
        
        n_windows=1
        y_pred=[]
        
        if (doResize):
            d = self.n_resize*self.input_img_size
            img = self.resizeImage(img, d, d)
            n_windows = self.n_resize * self.n_resize
            y_pred = np.zeros(shape=[d, d], dtype=np.uint8)
        else:
            n_windows = (n_height+1) * (n_width + 1)
            y_pred = np.zeros(shape=[height, width], dtype=np.uint8)
            
        ########### Cropping
        x=y=0
        
        for crop_id in range(n_windows):

            ## Find the coordinate to start cropping the image
            if (doResize):
                x,y = self.searchXYfromResizedIMG(crop_id, self.n_resize, window_size)
            else:
                x,y = self.searchXYfromRawIMG(crop_id, n_height, window_size, width, height)
                
            ## Crop the selected region
            crop_img = self.cropSingle(img, x, y, window_size)

            ## Fit into the input size
            crop_img = self.resizeImage(crop_img, self.input_img_size, self.input_img_size)

        ## Predict
        x_pred_crop = np.array(crop_img, np.float32) / 255
        x_pred_crop = np.expand_dims(x_pred_crop,axis=0)
        y_pred_crop = model.predict(x_pred_crop)

        pred_mask_crop = np.array(255*y_pred_crop[0], dtype=np.uint8)
        rgb_pred_mask_crop = cv2.cvtColor(pred_mask_crop,cv2.COLOR_GRAY2RGB)
        rgb_pred_mask_crop = cv2.cvtColor(rgb_pred_mask_crop,cv2.COLOR_RGB2GRAY)
        thresh, y_pred_crop = cv2.threshold(rgb_pred_mask_crop, 225, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        y_pred_crop[y_pred_crop>0] = 1
    
        ## Fill a patch into the original size of a mask
        y_pred[y:y+window_size, x:x+window_size]=self.resizeImage(y_pred_crop, window_size, window_size)
    
        ## Zoom into the original size
        if (doResize):
            y_pred = self.resizeImage(y_pred, width, height)

        y_true = mask
        
        if (verbose):
            print("Predict:" , self.total_df["ID"][nID], " / elapsed time: " , round(time.time()-start_time,1), "[sec]")

        return y_pred, y_true
        
        

