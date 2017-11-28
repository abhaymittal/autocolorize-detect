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
        
    fr=fread.FileReader()

if __name__=='__main__':
    main()
