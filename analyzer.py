from __future__ import print_function
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
import os
import plot 
try:
    import cPickle as p
except:
    import Pickle as p


import logging
import numpy as np
import caltech101.file_reader as fread
from scipy.misc import imsave
from skimage.transform import rescale,resize

ACT='act'
SAMPLE_W='sample_width'
IMG='img'
ROW_IDX='row_idx'
COL_IDX='col_idx'

# This file contains functions to analyze the hidden units


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

def get_img_patch(img, row, col):
    '''
    Method to get the patch which is in the receptive field of the
    fc7 neuron at row,col 

    Args:
    img: The original image
    row: the row idx of the neuron
    col: The col idx of the neuron

    Return:
    img: image patch in the receptive field of the neuron
    '''

    # receptive field arithmetic : https://medium.com/@nikasa1889/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    recept_size=404
    jump=32
    start=0.5
    
    
    center_x, center_y=start+col*jump,start+row*jump
    beg_x,end_x=center_x-(recept_size/2),center_x+(recept_size/2)
    beg_y,end_y=center_y-(recept_size/2),center_y+(recept_size/2)
    beg_x=int(np.floor(beg_x))
    beg_y=int(np.floor(beg_y))
    end_x=int(np.ceil(end_x))
    end_y=int(np.ceil(end_y))
    beg_x = 0 if beg_x<0 else beg_x
    beg_y = 0 if beg_y<0 else beg_y
    end_x = img.shape[1] if end_x+1>img.shape[1] else end_x+1
    end_y = img.shape[0] if end_y+1>img.shape[0] else end_y+1
    
    return img[beg_y:end_y,beg_x:end_x]



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



def get_padded_image(img,input_size):
    '''
    Method to get the padded image that is fed to the net
    
    Args:
    img: The original image
    input_size: The input size to the neural net, Integer value
    '''
    min_side=float(input_size//2)
    max_side=input_size-12
    shorter_side = np.min(img.shape[:2])
    longer_side = np.max(img.shape[:2])
    scale = min(min_side / shorter_side, 1)
    if longer_side * scale >= max_side:
        scale = max_side / longer_side
    if scale != 1:
        img = resize_by_factor(img, scale)

    img = center_crop(img, (input_size, input_size))
    return img

def plot_category_histograms(best_dict,fr):
    '''
    Method to plot the frequency histograms for categories which show that for each hidden unit in the best_dict, how many
    images of the category occured 
    
    Args:
    best_dict: The best dictionary so far
    fr: File reader instance
    '''
    print(fr.getCategories())
    catgs=fr.getCategories()
    # The following are the categories
    # ['BACKGROUND_Google', 'Faces', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang', 'cougar_body']
    
    cat_count_matrix=get_act_category_count(best_dict,fr)
    print('Plotting category histograms')
    for idx,cat in enumerate(catgs):
        fig=plt.figure(figsize=(8,6))
        freq=cat_count_matrix[:,idx]
        max_freq=np.max(freq)
        x_axis=range(0,len(freq))
        plt.bar(x_axis,freq)
        size=fr.get_category_size(cat)
        plt.title('Histogram for '+cat+' with '+str(size)+' images')
        plt.xlabel('Hidden unit index')
        plt.ylabel('Count of '+cat+' in top 1000')
        plt.savefig('fig/'+cat+'.png')
        plt.close()

def generate_patches_neuron(fr,best_dict,neuron_idx, input_size=576, n_imgs=None):
    '''
    Method to get the receptive field for all the images in the best_dict for a neuron. The patches are 
    saved in patch/ directory
    Args:
    fr: File reader 
    best_dict: The best activation dictionary
    neuron_idx: The index of the hidden unit whose patches are to be generated
    input_size: The neural net input size
    n_imgs: The number of patches to generate
    '''
    if n_imgs is None:
        n_imgs=best_dict[IMG].shape[1]
    img_ids=best_dict[IMG][neuron_idx,:n_imgs]
    rows=best_dict[ROW_IDX][neuron_idx,:n_imgs]
    cols=best_dict[COL_IDX][neuron_idx,:n_imgs]
    for i,idx in enumerate(img_ids):
        print('Processed image idx: ',idx)
        img=fr.get_image_by_id(idx)
        img=get_padded_image(img,input_size)
        patch=get_img_patch(img, rows[i], cols[i])
        patch_name='patch/'+str(idx)+' '+str(rows[i])+' '+str(cols[i])+'.jpg'
        imsave(patch_name,patch)
    return



    

def get_best_neurons(fr,cat_name,best_dict,n=1):
    '''
    Method to get the best neurons for a category
    
    Args:
    fr: File reader instance
    cat_name: The category name
    best_dict: best activation dictionary
    n: The number of neurons needed
    '''
    size=fr.get_category_size(cat_name)
    cat_count_matrix=get_act_category_count(best_dict,fr)
    cats=fr.getCategories()
    cat_idx=cats.index(cat_name)
    counts=cat_count_matrix[:,cat_idx]
    best_idx=np.argsort(-1*counts)
    return best_idx[:n]

def get_neuron_statistics(neuron_idx,best_dict,fr,ntop=1000,n_cat=3):
    '''
    Method to print the top categories detected by a neuron
    Args:
    neuron_idx: The index of the neuron being considered
    best_dict: the best activation dictionary
    fr: File reader instance
    ntop: The number of activations to consider
    n_cat: Print details about top n_cat categories
    '''
    print('Among top '+str(ntop))
    img=best_dict[IMG][:,:ntop]
    best_dictn=dict()
    best_dictn[IMG]=img
    cat_count_matrix=get_act_category_count(best_dictn,fr)
    cats=fr.getCategories()
    cat_idx=np.argsort(-1*cat_count_matrix[neuron_idx,:])
    cat_idx=cat_idx[:n_cat]
    counts=cat_count_matrix[neuron_idx,:]
    for cat_id in cat_idx:
        print('Category = '+cats[cat_id]+' size = ',fr.get_category_size(cats[cat_id]),' count = ',counts[cat_id])
    return

def compute_precision_recall(cat_name, neurons,fr,best_dict,verbose=True):
    '''
    Method to compute the precision and recall for a set of hidden units
    
    Args:
    cat_name: The category for which precision and recall are to be computed
    neurons: A numpy array of hidden units
    fr: File reader instance
    best_dict: The best activations dictionary
    
    Return:
    precision,recall
    '''
    num_samples=best_dict[IMG].shape[1]
    if verbose:
        print('Computing precision and recall for ',cat_name,' @',num_samples)
    cats=fr.getCategories()
    cat_size=np.minimum(fr.get_category_size(cat_name),num_samples)
    cat_count_matrix=get_act_category_count(best_dict,fr)
    precision=np.zeros(neurons.shape,dtype=np.float64)
    recall=np.zeros_like(precision)
    for i,neuron in enumerate(neurons):
        count=cat_count_matrix[neuron,cats.index(cat_name)]
        tp=count
        
        # False positive and false negative should be same here because we are taking num_samples best activations and the 
        # images for cat_name that we missed were replaced by some other category images (FN)
        fp=cat_size-count
        fn=fp

        precision[i]=float(tp)/(tp+fp)
        recall[i]=float(tp)/(tp+fn)
        if verbose:
            print('Neuron',neuron,' detected ',count,' out of ',cat_size,'. Precision = ',precision[i])

    return precision,recall


def generate_top_patches_img(fr,best_dict,neuron_idx, n_cols=8, n_imgs=None, input_size=576):
    if n_imgs is None:
        n_imgs=best_dict[IMG].shape[1]
    img_ids=best_dict[IMG][neuron_idx,:n_imgs]
    recept_rows=best_dict[ROW_IDX][neuron_idx,:n_imgs]
    recept_cols=best_dict[COL_IDX][neuron_idx,:n_imgs]
    n_rows=int(np.ceil(n_imgs*1.0/n_cols))
    img_idx=0
    ind_img_size=(100,100)
    output_img=np.zeros((n_rows*ind_img_size[0],n_cols*ind_img_size[1]))
    break_flag=False
    for r in np.arange(n_rows):
        for c in np.arange(n_cols):
            img=fr.get_image_by_id(img_ids[img_idx])
            img=get_padded_image(img,input_size)
            patch=get_img_patch(img, recept_rows[img_idx], recept_cols[img_idx])
            patch=resize(patch,ind_img_size)
            beg_row_idx=r*ind_img_size[0]
            beg_col_idx=c*ind_img_size[1]
            if(len(patch.shape)==3):
                patch=patch[...,:3].mean(-1)
            output_img[beg_row_idx:beg_row_idx+ind_img_size[0],beg_col_idx:beg_col_idx+ind_img_size[1]]=patch
            img_idx+=1
            if img_idx>=n_imgs:
                break_flag=True
                break
        if break_flag:
            break
    return output_img

    

def main():
    dict_file='dump/best_max_dict.p'
    fr_file='dump/file_reader.p'
    best_dict=dict()
    with open(dict_file,'rb') as f:
        best_dict=p.load(f)
    fr=None
    with open(fr_file,'rb') as f:
        fr=p.load(f)
    # plot_category_histograms(best_dict,fr)
    # print('Generating patches')
    # generate_patches_neuron(fr,best_dict,10,n_imgs=10)
    
    cat_name='Faces_easy'

    # The following is the code to print statistics category wise, like the best neurons for category and the top categories 
    # among those neurons
    # neurons=get_best_neurons(fr,cat_name,best_dict,5)
    # cat_count_matrix=get_act_category_count(best_dict,fr)
    # cats=fr.getCategories()
    # cat_idx=cats.index(cat_name)
    # print('Neurons for '+cat_name+' are = ',neurons)
    # print('Their counts are =',cat_count_matrix[neurons,cat_idx])
    # for neuron in neurons:
    #     print ('Neuron ',neuron)
    #     get_neuron_statistics(neuron,best_dict,fr,ntop=1000,n_cat=3)
    #     get_neuron_statistics(neuron,best_dict,fr,ntop=500,n_cat=3)
    #     get_neuron_statistics(neuron,best_dict,fr,ntop=400,n_cat=3)
    #     get_neuron_statistics(neuron,best_dict,fr,ntop=300,n_cat=3)
    #     get_neuron_statistics(neuron,best_dict,fr,ntop=200,n_cat=3)
    #     get_neuron_statistics(neuron,best_dict,fr,ntop=100,n_cat=3)
    #     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    # plot_category_histograms(best_dict,fr)


    cat_name='Faces_easy'
    num_neurons=5
    num_samples=1000
    best_dictn=dict()
    best_dictn[IMG]=best_dict[IMG][:,:num_samples]
    neurons=get_best_neurons(fr,cat_name,best_dictn,num_neurons)
    act_img=generate_top_patches_img(fr,best_dict,neurons[0], n_cols=20, n_imgs=500)
    imsave('act.png',act_img)
    # precision=np.zeros([num_neurons,11])
    # n_samples=[1,10,50,100,200,300,400,500,600,700,800]
    # # n_samples=[1,10,50,100,200,300,400,500,800]
    # for i,num_samples in enumerate(n_samples):
    #     best_dictn=dict()
    #     best_dictn[IMG]=best_dict[IMG][:,:num_samples]
    #     pr,_=compute_precision_recall(cat_name, neurons, fr,best_dictn,verbose=False)
    #     precision[:,i]=pr
    # neuron_names=['unit '+str(x) for x in neurons]
    # plot.group_line_plot(n_samples,precision,neuron_names,'Number of top activations','Precision','Precision vs number of top activations for airplanes','prec_air.png')


if __name__=='__main__':
    main()
