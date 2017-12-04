from __future__ import print_function
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


def main():
    dict_file='dump/best_dict_cat.p'
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
        plt.figure()
        freq=cat_count_matrix[:,idx]
        x_axis=range(0,len(freq))
        plt.bar(x_axis,freq)
        size=fr.get_category_size(cat)
        plt.title(cat+' '+str(size))
        plt.savefig('fig/'+cat+'.png')


if __name__=='__main__':
    main()
