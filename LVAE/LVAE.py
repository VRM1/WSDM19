import numpy as np
from tqdm import tqdm
import json
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from DSAE import DSAEB
import gensim
import sys
np.random.seed(7)
''' This DSAE model is for using the NON - embedding layer version of the model'''
    
    
# doe sone hot encoding of labels
def encodeLabels(labels):
    
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_Y = label_encoder.transform(labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    labels = np_utils.to_categorical(encoded_Y)
    return labels

# method to train the model with product reviews.
def GetReviews(graph,doc2vec):
    
    itm_rev_a = []
    itm_rev_b = []
    pairs = []
    labels = []
    for ids in graph:
        ids = ids.split(',')
        itm_rev_a.append(list(doc2vec[ids[0]]))
        itm_rev_b.append(list(doc2vec[ids[1]]))
        pairs.append(ids)
        labels.append(ids[2])
    itm_rev_a = np.array(itm_rev_a)
    itm_rev_b = np.array(itm_rev_b)
    return (itm_rev_a,itm_rev_b,pairs,labels)

# method that return item-item relationship graph (i.e. substitute, complements etc)
def GetItmGraphsBinary(name):
    

    end = '_graph_2class_filtered.json'
    with open('dataset/'+name+'/'+name+end,'r') as fp:
        itm_pairs = json.load(fp)
    # we have two cases now (a) substitute and (b) complement
    subs,compl,=[],[]
    for itms in tqdm(itm_pairs):
        tmp_pairs = [i.strip() for i in itms.strip('(|)').split(',')]
        if itm_pairs[itms] == 1:
            subs.append(','.join(tmp_pairs)+',1')
        else:
            compl.append(','.join(tmp_pairs)+',0')
    subs = subs[:500]
    compl = compl[:500]
    data = subs + compl
   # print '% of substitutes:{:.2f}'.format(len(subs)/float(len(data))*100)
    print('Percent of substitutes:%'%(len(subs)/float(len(data))*100))
   # print '% of complements:{:.2f}'.format(len(compl)/float(len(data))*100)
    print('Percent of substitutes:%'%(len(compl)/float(len(data))*100))
    return data

'''
The categorical link is based on the following rules described in the paper
1."Users who viewed x also viewed y"
2."Users who viewed x eventually bought y"
3."Users who bought x also bought y"
4."Users frequently bought x and y together"
'''
def GetItmGraphsCateg(name):
    

    end = '_graph_4class_filtered.json'
    with open('dataset/'+name+'/'+name+end,'r') as fp:
        itm_pairs = json.load(fp)
    # we have two cases now (a) substitute and (b) complement
    vu_x_also_y,vu_x_even_y,=[],[]
    bt_x_also_y,bt_x_n_y,=[],[]
    for itms in tqdm(itm_pairs):
        tmp_pairs = [i.strip() for i in itms.strip('(|)').split(',')]
        if itm_pairs[itms] == 1:
            vu_x_also_y.append(','.join(tmp_pairs)+',1')
        elif itm_pairs[itms] == 2:
            vu_x_even_y.append(','.join(tmp_pairs)+',2')
        elif itm_pairs[itms] == 3:
            bt_x_also_y.append(','.join(tmp_pairs)+',3')
        else:
            bt_x_n_y.append(','.join(tmp_pairs)+',4')
#     subs = subs[:2500000]
#     compl = compl[:5000000]
    data = vu_x_also_y + vu_x_even_y + bt_x_also_y + bt_x_n_y
    print('1. %% Users who viewed x also viewed y: %.2f'%(len(vu_x_also_y)/float(len(data))*100))
    print ('2. %% Users who viewed x eventually bought y: %.2f'%(len(vu_x_also_y)/float(len(data))*100))
    print ('3. %% Users who bought x also bought y: %.2f'%(len(bt_x_also_y)/float(len(data))*100))
    print ('4. %% Users frequently bought x and y together: %.2f'%(len(bt_x_n_y)/float(len(data))*100))
    # print '1. % Users who viewed x also viewed y: {:.2f}'.format(len(vu_x_also_y)/float(len(data))*100)
    # print '2. % Users who viewed x eventually bought y:{:.2f}'.format(len(vu_x_also_y)/float(len(data))*100)
    # print '3. % Users who bought x also bought y:{:.2f}'.format(len(bt_x_also_y)/float(len(data))*100)

    return data
    
def TrainAmazon(name,batch_size,z_dim,epochs,ld_weight,typ):
    
    # load the d2v trained model, which will serve as input to VAE
    d2v_model = gensim.models.doc2vec.Doc2Vec.load('dataset/'+name+'/'+name+'_reviews'+'.d2v')
#     main_data = [list(d2v_model.docvecs[i]) for i in xrange(len(d2v_model.docvecs))]
    print("Model loaded")
    #  maximum length of each document review
    inpt_dim = len(d2v_model[0])
    # total number of unique words
    inter_dim = 256
    if typ == 'binary':
        graph_data = GetItmGraphsBinary(name)
    else:
        graph_data = GetItmGraphsCateg(name)

    print ('total item pairs: %d, total item with reviews: %d'%(len(graph_data),len(d2v_model.docvecs)))
    # print 'total item pairs:{}, total item with reviews:{}'\
    #         .format(len(graph_data),len(d2v_model.docvecs))
    # 3K samples for train and 1k for test
    train,test = train_test_split(graph_data,train_size=0.75)
    train,validation = train_test_split(train,train_size=0.80)
    print('train size: %d, validation size: %d, test size: %d'\
          %(len(train),len(validation),len(test)))

    # print 'train size:{}, validation size:{}, test size:{}'.format(len(train),len(validation),len(test))
    ''' get the reviews corresponding to the selected 
    item pairs and their labels (substitutes and complements)'''
    train_a,train_b,pairs_train,labels = GetReviews(train,d2v_model)
    valid_a,valid_b,pairs_valid,valid_labels = GetReviews(validation,d2v_model)
    test_a,test_b,pairs_test,test_labels = GetReviews(test,d2v_model)
#     print '# test dataset:{} items'.format(len(test))
#     encoded_revs = [train[itm] for itm in sorted(train.keys())]
    labels = encodeLabels(labels)
    valid_labels = encodeLabels(valid_labels)
    labels_test = encodeLabels(test_labels)
    # get the models
    dsae = DSAEB(name,inter_dim, z_dim, inpt_dim,batch_size,typ)
    DSAE,LinkPredictor,cp,LinkGenerator = dsae.getModels()
    if ld_weight == 'yes':
        weights_path = 'output/DSAE_'+name+'.hdf5'
        DSAE.load_weights(weights_path)
        print('loaded pre-trained weights')
    else:
        # train the model
        DSAE.fit([np.array(train_a),np.array(train_b)],[np.array(train_a),np.array(train_b),labels], \
                 shuffle=True, nb_epoch=epochs, batch_size=batch_size, \
                 validation_data=[[np.array(valid_a),np.array(valid_b)],\
                                  [np.array(valid_a),np.array(valid_b),valid_labels]], \
                 callbacks=cp)
    
    results = LinkPredictor.evaluate(x=[np.array(test_a),np.array(test_b)],y=[labels_test])
    print(results)
     

if __name__ == '__main__':
    
    ''' max length of each review is 200 after the filtration.
    So, we use this for padding.'''
    data_typ = ['Musical_Instrument','Electronics','Movies_and_TV','Books']
    batch_size=512
    epochs = 70
    z_dim=60
    # the type of class label (1-> subs, 2-> complement, or the 4 type of classes with direction)
    typ=['binary','categorical']
    # chose the data type here by index. For instance, 0 indicates 'Musical_Instrument', 1 indicates 'Electronics'..
    name = data_typ[0]
    ld_weight = 'no'
    TrainAmazon(name,batch_size,z_dim,epochs,ld_weight,typ[1])
