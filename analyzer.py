from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
try:
    import cPickle as p
except:
    import Pickle as p


import logging
import numpy as np
import caltech101.file_reader as fread


ACT='act'
SAMPLE_W='sample_width'
IMG='img'
ROW_IDX='row_idx'
COL_IDX='col_idx'


def get_img_categories(best_dict,fr):
    '''
    Method to get the category of the images present in the best dictionary
    
    Args:
    ----
    best_dict: The best dictionary
    fr: An instance of file reader
    '''
    img=best_dict[IMG]
    cat=fr.getCategories()
    
    
    # find the category beginning idx to discretize
    np_dis_idx=np.zeros(len(cat)-1,dtype=np.uint32)
    for i,ct in enumerate(cat[1:]):
        np_dis_idx[i]=fr.get_cat_id_range(ct)[0]
    return np.digitize(img,np_dis_idx)

def get_act_category_count(best_dict,fr):
    '''
    Method to get a n_hid_unit x n_cat matrix where each position i,j
    refers to number of images of category j present in hidden unit i's top 
    activations
    
    Args:
    best_dict: The best dictionary
    fr: An instance of file reader
    '''
    cat_idx=get_img_categories(best_dict,fr)
    cat=fr.getCategories()
    cat_counts=np.zeros((cat_idx.shape[0],len(cat)),dtype=np.uint32)
    row_idx=np.arange(0,cat_idx.shape[0])

    for i in np.arange(0,cat_idx.shape[1]):
        cat_counts[row_idx,cat_idx[:,i]]+=1
    return cat_counts

def get_img_patch(img, x, y):
    '''
    
    '''

    # receptive field arithmetic : https://medium.com/@nikasa1889/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    recept_size=404
    jump=32
    start=0.5
    
    
    center_x, center_y=start+x*jump,start+y*jump
    beg_x,end_x=center_x-(recept_size/2),center_x+(recept_size/2)
    beg_y,end_y=center_y-(recept_size/2),center_y+(recept_size/2)
    beg_x = 0 if beg_x<0 else beg_x
    beg_y = 0 if beg_y<0 else beg_y
    end_x = img.size[1] if end_x+1>img.size[1] else end_x+1
    end_y = img.size[1] if end_y+1>img.size[1] else end_y+1
    return img[beg_x:end_x,beg_y:end_y]



# resize_by_factor and center_crop have been copied from  autocolorize to prevent the loading of caffe for this
# which needs a couple of seconds and installation of caffe is also required then
def resize_by_factor(img, factor):
    if factor != 1:
        return rescale(img, factor, mode='constant', cval=0)
    else:
        return img

def center_crop(img, size, value=0.0):
    """Center crop with padding (using `value`) if necessary"""
    new_img = np.full(size + img.shape[2:], value, dtype=img.dtype)

    dest = [0, 0]
    source = [0, 0]
    ss = [0, 0]
    for i in range(2):
        if img.shape[i] < size[i]:
            diff = size[i] - img.shape[i]
            dest[i] = diff // 2
            source[i] = 0
            ss[i] = img.shape[i]
        else:
            diff = img.shape[i] - size[i]
            source[i] = diff // 2
            ss[i] = size[i]

    new_img[dest[0]:dest[0]+ss[0], dest[1]:dest[1]+ss[1]] = \
            img[source[0]:source[0]+ss[0], source[1]:source[1]+ss[1]]

    return new_img



def get_padded_image(grayscale,input_size):
    min_side=input_size//2
    max_side=input_size-12
    shorter_side = np.min(grayscale.shape[:2])
    longer_side = np.max(grayscale.shape[:2])
    scale = min(min_side / shorter_side, 1)
    if longer_side * scale >= max_side:
        scale = max_side / longer_side
    if scale != 1:
        grayscale = resize_by_factor(grayscale, scale)

    grayscale = center_crop(grayscale, (size, size))
    return grayscale


def main():
    dict_file='dump/best_dict_cat.p'
    fr_file='file_reader.p'
    best_dict=dict()
    with open(dict_file,'rb') as f:
        best_dict=p.load(f)
    

    fr=None
    with open(fr_file,'rb') as f:
        fr=p.load(f)

        
    print(fr.getCategories())

    catgs=fr.getCategories()
    # The following are the categories
    # ['BACKGROUND_Google', 'Faces', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang', 'cougar_body']
    
    cat_count_matrix=get_act_category_count(best_dict,fr)

    for idx,cat in enumerate(catgs):
        fig=plt.figure()
        freq=cat_count_matrix[:,idx]
        x_axis=range(0,len(freq))
        plt.bar(x_axis,freq)
        size=fr.get_category_size(cat)
        plt.title(cat+' '+str(size))
        plt.savefig('fig/'+cat+'.png')
        plt.close()


if __name__=='__main__':
    main()
