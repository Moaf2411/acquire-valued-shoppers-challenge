import numpy as np
import pandas as pd

#----------------------------------------<< train a gbdt on every dataset >>---------------------------------------------


def gety(x):
    if x == 'f':
        return 0
    return 1
'''

from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
for i in range(21):
    df = pd.read_csv('trainsets//'+str(i)+'.csv')
    df['repeater'] = np.vectorize(gety)(df['repeater'])
    Y = df['repeater']
    X = df.drop('repeater',axis=1)
    X = X.drop('repeattrips',axis=1)
    
    model = GradientBoostingClassifier(learning_rate=1,n_estimators=100,max_depth=4,validation_fraction=0.05,verbose=1)
    model.fit(X,Y)
    dump(model,'models//model_'+str(i)+'.joblib')
'''


'''
transformed = {4:0,5:0,7:0,8:0,11:0,12:0,14:0,15:0,19:0,20:0,22:0,23:0,26:0,27:0,29:0,30:0}

from joblib import load
models = []
for i in range(21):
    models.append(load('models//model_'+str(i)+'.joblib'))
    


    
for z in range(21):
    print(z)
    print('-----------')
    df = pd.read_csv('trainsets//'+str(z)+'.csv')
    X = df.drop(['repeater','repeattrips'],axis=1)
    feats = []
    for i in models:
        feats.append(i.apply(X))
    
    final_feats = []
    
    for i in range(len(df)):
        ff = transformed.copy()
        for j in range(len(models)):
            full_tree = []
            for k in range(len(models[j].estimators_)):
                if len( models[j].estimators_[k][0].tree_.__getstate__()['nodes'] ) == 31: # if it was a full tree, save the index 
                    full_tree.append(k)
            
            for k in full_tree:
                ff[int(feats[j][i][k])] += 1  #feats[j][i][k] => for model number j, get the prediction of k'th tree (which is a full tree) for the i'th row
        final_feats.append( np.array(list(ff.values()))/np.array(list(ff.values())).max() )
        del ff
        del full_tree
    
    ff = pd.DataFrame(final_feats)
    ff.to_csv('trainsets//'+str(z)+'_transformed.csv',index=False)
    del ff
    del df
    del X
    del feats
    del final_feats

'''    






# find out index of each node returned by apply()
'''
ind = 0
for i in models[0].estimators_:
    if len( i[0].tree_.__getstate__()['nodes'] ) == 31:
        for j in i[0].tree_.__getstate__()['nodes']:
            print('node number '+str(ind)+' : ',end='')
            print(j)
            ind += 1
        break         
'''   
        
# node indecies from left of the tree to right:
# 4 , 5 , 7 , 8 , 11 , 12 , 14 , 15 , 19 , 20 , 22 , 23 , 26 , 27 , 29 , 30






'''
for i in range(21):
    df = pd.read_csv('trainsets//'+str(i)+'.csv')
    x = df['repeater']
    
    x = pd.DataFrame(x)
    x['y'] = np.vectorize(gety)(x['repeater'])
    x.to_csv('trainsets//'+str(i)+'_y.csv',index=False)
    del df
    del x
'''

from sklearn.ensemble import GradientBoostingClassifier
model1 = GradientBoostingClassifier(learning_rate=0.3,max_depth=10,n_estimators=100,verbose=1)
X = pd.read_csv('trainsets//0_transformed.csv')
Y = pd.read_csv('trainsets//0_y.csv')['y']
model1.fit(X,Y)




transformed = {4:0,5:0,7:0,8:0,11:0,12:0,14:0,15:0,19:0,20:0,22:0,23:0,26:0,27:0,29:0,30:0}
from joblib import load
models = []
for i in range(21):
    models.append(load('models//model_'+str(i)+'.joblib'))
    
    

df = pd.read_csv('trainsets//5.csv')
ytrue = df['repeater']
ytrue = pd.DataFrame(ytrue)
ytrue['y'] = np.vectorize(gety)(ytrue['repeater'])
res = model1.predict(pd.read_csv('trainsets//5_transformed.csv'))


    
wrong = 0
for i in range(len(res)):
    if res[i] != ytrue.iloc[i]['y']:
        wrong += 1
print(1 - wrong/10000)
    
            
            
            
            
            
            
