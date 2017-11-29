from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
os.environ["GLOG_minloglevel"] = "2"
import caffe
import autocolorize as ac
import logging
import argparse
import faces_in_the_wild_dataset.download_data as dd
import numpy as np
import config
from caltech101.file_reader import FileReader
import matplotlib.pyplot as plt

try:
    import cPickle as p
except:
    import Pickle as p

ACT='act'
SAMPLE_W='sample_width'
IMG='img'
ROW_IDX='row_idx'
COL_IDX='col_idx'

def configure_logging(log_level=logging.INFO):
    '''
    Method to configure the logger
    '''
    # Rewrite log
    logging.basicConfig(filename='ac_max.log',filemode='w',level=log_level) 
    # logging.basicConfig(level=log_level)

def parse_args():
    '''
    Method to get the parsed command line arguments
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument('-s','--save_img', 
                        help='True to store colorized images, else false',action='store_true')
    parser.add_argument('-d','--debug',help='Flag to enable debugging',
                        action='store_true')
    parser.add_argument('-l','--load_best_cat_dict',help='load the previously saved dictionary after finishing category',
                        action='store_true')
    return parser.parse_args();

def get_fc7_activations(net):
    '''
    Method to get the activations of the fc7 layer
    '''
    fc7=net.blobs['fc7']
    return fc7.data

def get_fc7_max_activation(net):
    '''
    Method to get the max activation for each filter in the fc7 layer
    '''
    act=get_fc7_activations(net)
    return np.max(act,(0,2,3))

def initialize_dict(act_dict,n_samples,n_hid_units):
    '''
    Method to initialize the activation dictionaries
    
    ACT variable defines the key to a numpy array storing the 
    activaton values. For every hidden unit 
    n_samples*act_width*act_height activations are stored
    
    
    Args:
    -----
    act_dict: The dictionary to initialize
    n_samples: the number of samples to store in the dict. 
    n_hid_units: The number of hidden units
    '''
    act_dict[ACT]=np.ndarray((n_hid_units,n_samples),dtype=np.float64)
    act_dict[ROW_IDX]=np.zeros((n_hid_units,n_samples),dtype=np.uint8)
    act_dict[COL_IDX]=np.zeros((n_hid_units,n_samples),dtype=np.uint8)
    return

def initialize_best_act_dict(act_dict,n_hid_units,n_top):
    '''
    Method to initialize the activation dictionary which stores the best activations
    
    ACT variable defines the key to a numpy array storing the 
    activaton values. For every hidden unit  n_top activations
    are stored
    
    IMG variable defines the key to a numpy array storing the images
    corresponding to each activation index. Its size is same as ACT array
    
    Args:
    -----
    act_dict: The dictionary to initialize
    n_samples: the number of samples to store in the dict. 
         Must be < 255
    n_hid_units: The number of hidden units
    act_width, act_height: The width and height of the activation volume
    '''
    logging.info('Initializing best act dictionary for '+str(n_top)+' activations')
    act_dict[ACT]=np.zeros((n_hid_units,n_top),dtype=np.float64)
    act_dict[IMG]=np.zeros((n_hid_units,n_top),dtype=np.uint32)
    act_dict[ROW_IDX]=np.zeros((n_hid_units,n_top),dtype=np.uint8)
    act_dict[COL_IDX]=np.zeros((n_hid_units,n_top),dtype=np.uint8)

def append_to_dict(act_dict,current_act,idx):
    '''
    Method to apend activations to the activation dict

    Args:
    ----
    act_dict: The activation dictionary
    current_act:current activation volume (1, n_hidden, height , width)
    idx: The idx of the image in batch, IMG variable is set to this value
    '''
    activations=act_dict[ACT]
    act_width=current_act.shape[3]

    # reshape the best act to 2d
    current_act=current_act.reshape((current_act.shape[1],-1))
    
    max_act=np.amax(current_act,1)
    max_idx=np.argmax(current_act,1)
    row_idx=max_idx/act_width
    col_idx=max_idx%act_width
    
    # append the values
    act_dict[ACT][:,idx] = max_act
    act_dict[ROW_IDX][:,idx]=row_idx
    act_dict[COL_IDX][:,idx]=col_idx

    return

def update_best_dict(best_dict,act_dict, img_idx):
    '''
    Method to update the best dictionary usi>ng the current activation
    dictionary.
    Args:
    best_dict: The best dict to update
    img_idx: An array of image ids
    '''
    cur_act=act_dict[ACT]
    best_act=best_dict[ACT]
    x=np.argsort(-cur_act,axis=1)
    # print(x)
    n_best=best_act.shape[1]
    x=x[:,:n_best]
    img_idx=np.array(img_idx)
    idx=img_idx[x]

    act=np.ndarray((cur_act.shape[0],n_best),dtype=np.float64)
    row_idx=np.arange(0,cur_act.shape[0]).reshape((cur_act.shape[0],1))
    act=cur_act[row_idx,x]
    act_row_idx=act_dict[ROW_IDX][row_idx,x]
    act_col_idx=act_dict[COL_IDX][row_idx,x]
    
    c=np.concatenate((best_act,act),axis=1)
    c_img=np.concatenate((best_dict[IMG],idx),axis=1)
    c_act_row_idx=np.concatenate((best_dict[ROW_IDX],act_row_idx),axis=1)
    c_act_col_idx=np.concatenate((best_dict[COL_IDX],act_col_idx),axis=1)


    # print(c_img.shape, c.shape)
    x=np.argsort(-c,kind='mergesort',axis=1)
    x=x[:,:n_best]
    best_act=c[row_idx,x]
    best_dict[IMG]=c_img[row_idx,x]
    best_dict[ACT]=best_act
    best_dict[ROW_IDX]=c_act_row_idx[row_idx,x]
    best_dict[COL_IDX]=c_act_col_idx[row_idx,x]
    # print (best_act)
    # print(best_dict[IMG])
    return


def main():
    IMG_SIZE=576
    args=parse_args()
    print(args)

    # configure logging
    if args.debug:
        configure_logging(level=logging.DEBUG)
    else:
        configure_logging()

    # load the network and get the images
    net=ac.load_default_classifier(input_size=IMG_SIZE)
    logging.info('Loaded network')
    # image_dict=dd.get_images();
    fr=FileReader()
    # run on an image here to get the height and width
    act_dict=dict()
    batch_size=100
    n_hid_units=4096
    wid=IMG_SIZE/32 # 5 pooling layers before fc7 with kernel size 2
    hgh=IMG_SIZE/32
    # initialize_dict(act_dict,batch_size,n_hid_units,wid,hgh)

    # best dict
    best_dict=dict()
    if(not args.load_best_cat_dict):
        logging.info('Initializing best dictionary from scratch')
        n_top=50
        initialize_best_act_dict(best_dict,n_hid_units,n_top)
    else:
        logging.info('Loading previously saved cat dict file')
        with open(config.BEST_CAT_DICT_FILE,'rb') as f:
            best_dict=p.load(f)
        print(best_dict.keys())
        print(best_dict[ACT])
        print(best_dict[IMG])
        

    cat=fr.getCategories()
    logging.info(cat)
    limit=1000
    for ct in cat[2:]:
        if ct==cat[3]: # After faces, do only max 100 images of others
            limit=1
        logging.info('Category: '+ct+" Size = "+str(fr.get_category_size(ct)))
        l=0
        while fr.has_more(ct) and l<limit:
            imgs,idx=fr.getNextFiles(ct,batch_size)
            initialize_dict(act_dict,len(idx),n_hid_units)
            j=0
            for img in imgs:
                logging.info("Run for img "+str(idx[j])+" shape = ")
                logging.info(img.shape)
                rgb,info=ac.colorize(img,classifier=net,return_info=True)
                activation=get_fc7_activations(net)
                # print("Append to dict")
                append_to_dict(act_dict,activation,j)
                # print("Save")
                if args.save_img:
                    plt.subplot(1,2,1)
                    plt.imshow(np.stack((img,img,img),-1))
                    plt.subplot(1,2,2)
                    plt.imshow(rgb)
                    plt.savefig(config.SAVE_IMG_DIR+str(idx[j])+'.png')
                    # ac.image.save(config.SAVE_IMG_DIR+str(idx[j])+'.jpg',img)
                    # ac.image.save(config.SAVE_IMG_DIR+'color_'+str(idx[j])+'.jpg',rgb)
                # print("Saved")
                j+=1
            logging.info('Batch done, update best')
            update_best_dict(best_dict,act_dict,idx)
            l+=1
        logging.info('Move to next category')
        with open(config.BEST_MAX_CAT_DICT_FILE,'wb') as f:
            p.dump(best_dict,f)
        
    logging.info('All images done')
    
    with open(config.BEST_MAX_DICT_FILE,'wb') as f:
        p.dump(best_dict,f)
if __name__=='__main__':
    main()
