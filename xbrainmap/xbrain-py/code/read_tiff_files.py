import numpy as np
import glob
from PIL import Image

def read_tiff_files(files_location):
    files = glob.glob(files_location + '/*.tif')
    files.sort()
    input_data = []
    file_count = 0
    
    for file in files:
        input_data.append(np.asarray(Image.open(file)))
        file_count += 1
        
    input_data = np.asarray(input_data)
    # input data axes is z,y,x - change it to x,y,z                                                     
    input_data = np.transpose(input_data, (2,1,0))
    return input_data


