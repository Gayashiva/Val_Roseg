from PIL import Image
import numpy as np
import pandas as pd
import os, os.path
import glob
def get_date_taken(imageDir):
    image_path_list = []
    time=[]
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]

    #create a list all files in directory and
    #append files with a vaild extention to image_path_list
    image_list=np.array([])
    images=pd.DataFrame(columns=['time','file'])
    for file in os.listdir(imageDir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        time.append(Image.open(os.path.join(imageDir, file))._getexif()[36867])
        image_path_list.append(file)
    images['time']=time
    images['file']=image_path_list
    return images
if __name__=='__main__':
    path = '/Users/B.Suryanarayanan/Documents/Git/Ice_Stupa_Analysis/analysis/data/raw/Val_Roseg_Timelapse'
    images=get_date_taken(path)
    images.to_csv('/Users/B.Suryanarayanan/Documents/Git/Ice_Stupa_Analysis/analysis/data/processed/image_dates.csv')
