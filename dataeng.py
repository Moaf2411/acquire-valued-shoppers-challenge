import pandas as pd
from datetime import datetime
import numpy as np
import os

#--------------------<< merge trainHistory and testHistory with offers to get offer information >>-----------------------------

'''
offers = pd.read_csv('offers.csv')
train = pd.read_csv('trainHistory.csv')

df = train.join(offers.set_index('offer'),on='offer',how='left',lsuffix='test',rsuffix='_offers')


df = df.drop('offer',axis=1)

df.to_csv('train.csv',index=False)


offers = pd.read_csv('offers.csv')
test = pd.read_csv('testHistory.csv')

df = test.join(offers.set_index('offer'),on='offer',how='left',lsuffix='test',rsuffix='_offers')


df = df.drop('offer',axis=1)

df.to_csv('test.csv',index=False)

'''

#####################################################################################################


train = pd.read_csv('train.csv')








###############################################################################################################
#-------------------------------------<< getting unique values for categorical features >>---------------------------

'''
files = os.listdir('data')
cats2 = set()
dept = set()
chain = set()
company = set()
brand = set()
measure = set()



for i in files:
    d = pd.read_csv('data//'+i)
    for j in d['category'].unique():
        cats2.add(j)
    for j in d['dept'].unique():
        dept.add(j)
    for j in d['chain'].unique():
        chain.add(j)
   # for j in d['company'].unique():
    #    company.add(j)
    #for j in d['brand'].unique():
    #    brand.add(j)
    for j in d['productmeasure'].unique():
        measure.add(j)
    del d
    print(len(cats2))
    print(len(dept))
    print(len(chain))
    print(len(measure))

categs = pd.DataFrame(cats2,columns=['category'])
categs.to_csv('categories.csv',index=False)    

depts = pd.DataFrame(dept,columns=['dept'])
depts.to_csv('depts.csv',index=False) 

chains = pd.DataFrame(chain,columns=['chain'])
chains.to_csv('chains.csv',index=False) 

companies = pd.DataFrame(company,columns=['company'])
companies.to_csv('companies.csv',index=False) 

brands = pd.DataFrame(brand,columns=['brand'])
brands.to_csv('brands.csv',index=False) 

measures = pd.DataFrame(measure,columns=['measure'])
measures.to_csv('measures.csv',index=False) 

'''








#------------------------------- one hot encoder for category / first try -------------------------------

cat = pd.read_csv('categories.csv')
catt = set()
for i in cat['category']:
    if len(str(i)) == 3:
        catt.add(str(i)[0:1])
    else:
        catt.add(str(i)[0:2])



cat = cat.sort_values(by='category')
cat = np.array(cat['category'])
cats = dict.fromkeys(catt,0)
category_keys = []
for i in range(len(cats)):
    category_keys.append('cat_'+str(i))


def getcat(x,cat):
    if len(str(x)) <= 3:
        c = cat.copy()
        c[str(x)[0:1]] = 1
        ret = np.array(list(c.values()))
        del c
        ret = np.reshape(ret,(1,83))
        ret = pd.DataFrame(ret,columns=category_keys)
        return ret
    else:
        c = cat.copy()
        c[str(x)[0:2]] = 1
        ret = np.array(list(c.values()))
        del c
        ret = np.reshape(ret,(1,83))
        ret = pd.DataFrame(ret,columns=category_keys)
        return ret




#----------------------------------------------- one hot encoder for chain / first try ---------------------------------------------------------------

chain = pd.read_csv('chains.csv')
chainn = set()
for i in chain['chain']:
    chainn.add(i)
    


chains = dict.fromkeys(chainn,0)
chain_keys = []
for i in range(len(chains)):
    chain_keys.append('chain_'+str(i))


def getchain(x,chains):
    c = chains.copy()
    c[x] = 1
    ret = np.array(list(c.values()))
    del c
    ret = np.reshape(ret,(1,134))
    ret = pd.DataFrame(ret,columns=chain_keys)
    return ret
    
  
    
#-----------------------------------<< get days between offer and transaction >>---------------------------

def getdate(d):
    return datetime.strptime(d,'%Y-%m-%d').date()

def getdist(d1,d2):
    d = (getdate(d1) - getdate(d2)).days
    if d < 30:
        return 2
    else:
        return 1
    



#read in data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')










#--------------------------------------- company one hot encoder / first try -----------------------------------------------------------------------------------

comp = set()
for i in train['company'].unique():
    comp.add(i)
for i in test['company'].unique():
    comp.add(i)

comp = dict.fromkeys(comp,0)
comp[0] = 0
company_keys = []
for i in range(len(comp)):
    company_keys.append('comp_'+str(i))



def getcomp(x,comp):
    c = comp.copy()
    keys = c.keys()
    if x in list(keys):
        c[x] = 1
    else:
        c[0] = 1
    del keys
    del x
    ret = np.array(list(c.values()))
    ret = np.reshape(ret,(1,19))
    ret = pd.DataFrame(ret,columns=company_keys)
    del c
    return ret
    
 
 




#-------------------------------------------------- one hot encoder for brand / first try -------------------------------------------------------------------

brand = set()
for i in train['brand'].unique():
    brand.add(i)
for i in test['brand'].unique():
    brand.add(i)

brand = dict.fromkeys(brand,0)
brand[0] = 0
brand_keys = []
for i in range(len(brand)):
    brand_keys.append('brand_'+str(i))



def getbrand(x,brand):
    c = brand.copy()
    keys = c.keys()
    if x in list(keys):
        c[x] = 1
    else:
        c[0] = 1
    del keys
    del x
    ret = np.array(list(c.values()))
    ret = np.reshape(ret,(1,20))
    ret = pd.DataFrame(ret,columns=brand_keys)
    del c
    return ret   
    

    
    


market = set()

for i in train['market'].unique():
    market.add(str(i))
for i in test['market'].unique():
    market.add(str(i))
    

'''
#-----------------------------------------<< user transaction summary / first try >>---------------------------------

for ind in range(3914,len(train)):
    print('file number ',end='')
    print(ind)
    count = 0
    with open('data//'+str(train.iloc[ind]['id'])+'.csv') as f:
        
        w = 0
        o = f.readline()
        del o
        x = f.readline()
        x = x.split('\n')[0].split(',')
        chain = getchain(int(x[1]),chains)
        category = getcat(x[3], cats)
        company = getcomp(x[4],comp)
        brands = getbrand(x[5], brand)
        w1 = getdist(train.iloc[ind]['offerdate'], x[6])
        money = 0
        if x[9] != '0':
            money = np.float16(x[10]) / int(x[9])
        category = category * w1
        w += w1
        
        while(True):
            count += 1
            money1 = 0
            x = f.readline()
            if x == '':
                break
            x = x.split('\n')[0].split(',')
            chain1 = getchain(int(x[1]),chains)
            category1 = getcat(x[3], cats)
            company1 = getcomp(x[4],comp)
            brand1 = getbrand(int(x[5]), brand)
            w1 = getdist(train.iloc[ind]['offerdate'], x[6])
            if x[9] != '0':
                money1 = np.float16(x[10]) / int(x[9])
            category1 = category1 * w1
            chain += chain1
            category = category + category1
            company += company1
            brands += brand1
            money += money1
            w += w1
            del chain1
            del x
            del category1
            del company1
            del brand1
            del w1
            del money1
            
        print('# rows:',end = ' ')
        print(count)
        print('-------------')
        #save row
        category = category/np.array(category).max()
        company = company/np.array(company).max()
        chain2 = np.zeros((1,13))
        n = 0
        c = 0
        tot = 0
        for z in chain.iloc[0]:
            n+=1
            tot += z
            if n == 10 and c != 12:
                chain2[0,c] = tot
                tot = 0
                n = 0
                c += 1
            if n == 14 and c == 12:
                chain[0,c] = tot
        
        chain2=pd.DataFrame(chain2/np.array(chain2).max(),columns=['chain_1','chain_2','chain_3','chain_4','chain_5','chain_6','chain_7','chain_8','chain_9','chain_10','chain_11','chain_12','chain_13'])
        brands = pd.DataFrame(brands/np.array(brands).max())
        row = category.join(company)
        row = row.join(chain2)
        row = row.join(brands)
        
        
        offerchain = getchain(train.iloc[ind]['chain'], chains)
        offerchain2 = np.zeros((1,13),dtype='byte')
        n = 0
        c = 0
        tot = 0
        for z in offerchain.iloc[0]:
            n+=1
            tot += z
            if n == 10 and c != 12:
                offerchain2[0,c] = tot
                tot = 0
                n = 0
                c += 1
            if n == 14 and c == 12:
                offerchain2[0,c] = tot
        offerchain2=pd.DataFrame(offerchain2,columns=['offerchain_1','offerchain_2','offerchain_3','offerchain_4','offerchain_5','offerchain_6','offerchain_7','offerchain_8','offerchain_9','offerchain_10','offerchain_11','offerchain_12','offerchain_13'])
        row = row.join(offerchain2)
        
        
        offercat = getcat(train.iloc[ind]['category'], cats)
        offercat2 = np.zeros((1,8),dtype='byte')
        n = 0
        c = 0
        tot = 0
        for z in offercat.iloc[0]:
            n+=1
            tot += z
            if n == 10 and c != 7:
                offercat2[0,c] = tot
                tot = 0
                n = 0
                c += 1
            if n == 13 and c == 7:
                offercat2[0,c] = tot
        offercat2=pd.DataFrame(offercat2,columns=['offercat_1','offercat_2','offercat_3','offercat_4','offercat_5','offercat_6','offercat_7','offercat_8'])
    
        row = row.join(offercat2)
        
        
        
        offercompany = getcomp(train.iloc[ind]['company'],comp)
        offercompany.astype('byte')
        offercompany.columns = ['offercomp_0', 'offercomp_1', 'offercomp_2', 'offercomp_3', 'offercomp_4', 'offercomp_5', 'offercomp_6',
               'offercomp_7', 'offercomp_8', 'offercomp_9', 'offercomp_10', 'offercomp_11', 'offercomp_12',
               'offercomp_13', 'offercomp_14', 'offercomp_15', 'offercomp_16', 'offercomp_17', 'offercomp_18']
        offerbrand = getbrand(train.iloc[ind]['brand'], brand)
        offerbrand.astype('byte')
        offerbrand.columns = ['offerbrand_0', 'offerbrand_1', 'offerbrand_2', 'offerbrand_3', 'offerbrand_4', 'offerbrand_5',
               'offerbrand_6', 'offerbrand_7', 'offerbrand_8', 'offerbrand_9', 'offerbrand_10', 'offerbrand_11',
               'offerbrand_12', 'offerbrand_13', 'offerbrand_14', 'offerbrand_15', 'offerbrand_16', 'offerbrand_17',
               'offerbrand_18', 'offerbrand_19']
        row = row.join(offerbrand)
        row = row.join(offercompany)
        row['repeattrips'] = train.iloc[ind]['repeattrips']
        row['offervalue'] = train.iloc[ind]['offervalue']
        row['repeater'] = train.iloc[ind]['repeater']
        row['money'] = money/w
        
        row.to_csv('train//'+str(ind)+'.csv',index=False)
        del row
        del company
        del category
        del chain
        del chain2
        del brands
        del offerchain
        del offerchain2
        del offercat
        del offercat2
        del offercompany
        del offerbrand
    del f
    
 
'''


#-------------------------------------------<< user transaction summary / second try >>---------------------------------------------------------

   
'''
curr = pd.read_csv('data//'+str(train.iloc[0]['id'])+'.csv')  
w = 0
chain = getchain(curr.iloc[0]['chain'],chains)
category = getcat(curr.iloc[0]['category'], cats)
company = getcomp(curr.iloc[0]['company'],comp)
brands = getbrand(curr.iloc[0]['brand'], brand)
w1 = getdist(train.iloc[0]['offerdate'], curr.iloc[0]['date'])
money = 0
if curr.iloc[0]['purchasequantity'] != 0:
   money = curr.iloc[0]['purchaseamount'] / curr.iloc[0]['purchasequantity']
category = category * w1
w += w1
for i1 in range(1,len(curr)):      
    money1 = 0
    chain1 = getchain(curr.iloc[i1]['chain'],chains)
    category1 = getcat(curr.iloc[i1]['category'], cats)
    company1 = getcomp(curr.iloc[i1]['company'],comp)
    brand1 = getbrand(curr.iloc[i1]['brand'], brand)
    w1 = getdist(train.iloc[0]['offerdate'], curr.iloc[i1]['date'])
    if curr.iloc[i1]['purchasequantity'] != 0:
       money1 = curr.iloc[i1]['purchaseamount'] / curr.iloc[i1]['purchasequantity']
    category1 = category1 * w1
    chain += chain1
    category = category + category1
    company += company1
    brands += brand1
    money += money1
    w += w1
    del chain1
    del category1
    del company1
    del brand1
    del w1
    del money1    

    
#save row
category = category/np.array(category).max()
company = company/np.array(company).max()
chain = chain
chain2 = np.zeros((1,13))
n = 0
c = 0
tot = 0
for z in chain.iloc[0]:
    n+=1
    tot += z
    if n == 10 and c != 12:
        chain2[0,c] = tot
        tot = 0
        n = 0
        c += 1
        if n == 14 and c == 12:
            chain[0,c] = tot
        
chain2=pd.DataFrame(chain2/np.array(chain2).max(),columns=['chain_1','chain_2','chain_3','chain_4','chain_5','chain_6','chain_7','chain_8','chain_9','chain_10','chain_11','chain_12','chain_13'])
brands = brands/np.array(brands).max()
row = category.join(company)
row = row.join(chain2)
row = row.join(brands)
        
        
offerchain = getchain(train.iloc[0]['chain'], chains)
offerchain2 = np.zeros((1,13),dtype='byte')
n = 0
c = 0
tot = 0
for z in offerchain.iloc[0]:
    n+=1
    tot += z
    if n == 10 and c != 12:
        offerchain2[0,c] = tot
        tot = 0
        n = 0
        c += 1
    if n == 14 and c == 12:
        offerchain2[0,c] = tot
offerchain2=pd.DataFrame(offerchain2,columns=['offerchain_1','offerchain_2','offerchain_3','offerchain_4','offerchain_5','offerchain_6','offerchain_7','offerchain_8','offerchain_9','offerchain_10','offerchain_11','offerchain_12','offerchain_13'])
row = row.join(offerchain2)
        
        
offercat = getcat(train.iloc[0]['category'], cats)
offercat2 = np.zeros((1,8),dtype='byte')
n = 0
c = 0
tot = 0
for z in offercat.iloc[0]:
    n+=1
    tot += z
    if n == 10 and c != 7:
        offercat2[0,c] = tot
        tot = 0
        n = 0
        c += 1
    if n == 13 and c == 7:
        offercat2[0,c] = tot
offercat2=pd.DataFrame(offercat2,columns=['offercat_1','offercat_2','offercat_3','offercat_4','offercat_5','offercat_6','offercat_7','offercat_8'])
    
row = row.join(offercat2)
        
        
        
offercompany = getcomp(train.iloc[0]['company'],comp)
offercompany.astype('byte')
offercompany.columns = ['offercomp_0', 'offercomp_1', 'offercomp_2', 'offercomp_3', 'offercomp_4', 'offercomp_5', 'offercomp_6',
           'offercomp_7', 'offercomp_8', 'offercomp_9', 'offercomp_10', 'offercomp_11', 'offercomp_12',
               'offercomp_13', 'offercomp_14', 'offercomp_15', 'offercomp_16', 'offercomp_17', 'offercomp_18']
offerbrand = getbrand(train.iloc[0]['brand'], brand)
offerbrand.astype('byte')
offerbrand.columns = ['offerbrand_0', 'offerbrand_1', 'offerbrand_2', 'offerbrand_3', 'offerbrand_4', 'offerbrand_5',
               'offerbrand_6', 'offerbrand_7', 'offerbrand_8', 'offerbrand_9', 'offerbrand_10', 'offerbrand_11',
               'offerbrand_12', 'offerbrand_13', 'offerbrand_14', 'offerbrand_15', 'offerbrand_16', 'offerbrand_17',
               'offerbrand_18', 'offerbrand_19']
row = row.join(offerbrand)
row = row.join(offercompany)
row['repeattrips'] = train.iloc[0]['repeattrips']
row['offervalue'] = train.iloc[0]['offervalue']
row['repeater'] = train.iloc[0]['repeater']
row['money'] = money/w
        
row.to_csv(str(0)+'.csv',index=False)
del row
del company
del category
del chain
del chain2
del brands
del offerchain
del offerchain2
del offercat
del offercat2
del offercompany 
del offerbrand
del curr
'''  
















#------------------------------<< user transaction summary / third try / optimized one >>-------------------------------- 

def cat2dept(x):
    if len(str(x)) <= 3:
        return str(x)[0:1]
    else:
        return str(x)[0:2]






def getbrand2(x,brand):
    c = brand.copy()
    b = x.value_counts()
    for k in c.keys():
        if k in b.index:
            c[k] = b[k]
    c[0] = (len(x)-np.array(list(c.values())).sum())
    del x
    ret = np.array(list(c.values()))
    div = max(c[0],b.max())
    ret = ret / div
    ret = np.reshape(ret,(1,20))
    ret = pd.DataFrame(ret,columns=brand_keys)
    del c
    return ret   
    
    
  
    
  
    
def getcomp2(x,comp):
    c = comp.copy()
    b = x.value_counts()
    for k in c.keys():
        if k in b.index:
            c[k] = b[k]
    c[0] = (len(x)-np.array(list(c.values())).sum())
    del x
    ret = np.array(list(c.values()))
    div = max(c[0],b.max())
    ret = ret / b.max()
    ret = np.reshape(ret,(1,19))
    ret = pd.DataFrame(ret,columns=company_keys)
    del c
    return ret
    
   
    
   



def getchain2(x,chains):
    c = chains.copy()
    b = x.value_counts()
    for k in b.index:
        c[k] = b[k] / b.max()
    del k
    del b
    ret = np.array(list(c.values()))
    del c
    ret = np.reshape(ret,(1,134))
    ret = pd.DataFrame(ret,columns=chain_keys)
    return ret







def getcat2(x,cat):
    c = cat.copy()
    b = x.value_counts()
    for k in b.index:
        c[str(k)] = b[k]
    ret = np.array(list(c.values()))
    ret = ret/b.max()
    ret = np.reshape(ret,(1,83))
    ret = pd.DataFrame(ret,columns=category_keys)
    return ret    






def getmoney(x,y):
    if y == 0:
        return 0
    else:
        return x/y
    





'''  
for ind in range(6229,len(train)):   
    print(ind)
    print('------------')
    curr = pd.read_csv('data//' + str(train.iloc[ind]['id']) + '.csv')
    curr.drop('date',axis=1)
    curr.drop('id',axis=1)
    curr['category'] = np.vectorize(cat2dept)(curr['category'])
    categ = getcat2(curr['category'],cats)
    chain3 = getchain2(curr['chain'],chains)
    brand3 = getbrand2(curr['brand'],brand)
    comp3 = getcomp2(curr['company'],comp)
    curr['money'] = np.vectorize(getmoney)( curr['purchaseamount'] , curr['purchasequantity'] )
    money = curr['money'].sum() / len(curr)
    
    chain2 = np.zeros((1,13))
    n = 0
    c = 0
    tot = 0
    for z in chain3.iloc[0]:
        n+=1
        tot += z
        if n == 10 and c != 12:
            chain2[0,c] = tot
            tot = 0
            n = 0
            c += 1
            if n == 14 and c == 12:
                chain2[0,c] = tot
            
    chain2=pd.DataFrame(chain2,columns=['chain_1','chain_2','chain_3','chain_4','chain_5','chain_6','chain_7','chain_8','chain_9','chain_10','chain_11','chain_12','chain_13'])
    
    
    
    row = categ.join(comp3)
    row = row.join(chain2)
    row = row.join(brand3)
            
            
    offerchain = getchain(train.iloc[ind]['chain'], chains)
    offerchain2 = np.zeros((1,13),dtype='byte')
    n = 0
    c = 0
    tot = 0
    for z in offerchain.iloc[0]:
        n+=1
        tot += z
        if n == 10 and c != 12:
            offerchain2[0,c] = tot
            tot = 0
            n = 0
            c += 1
        if n == 14 and c == 12:
            offerchain2[0,c] = tot
    offerchain2=pd.DataFrame(offerchain2,columns=['offerchain_1','offerchain_2','offerchain_3','offerchain_4','offerchain_5','offerchain_6','offerchain_7','offerchain_8','offerchain_9','offerchain_10','offerchain_11','offerchain_12','offerchain_13'])
    row = row.join(offerchain2)
            
            
    offercat = getcat(train.iloc[ind]['category'], cats)
    offercat2 = np.zeros((1,8),dtype='byte')
    n = 0
    c = 0
    tot = 0
    for z in offercat.iloc[0]:
        n+=1
        tot += z
        if n == 10 and c != 7:
            offercat2[0,c] = tot
            tot = 0
            n = 0
            c += 1
        if n == 13 and c == 7:
            offercat2[0,c] = tot
    offercat2=pd.DataFrame(offercat2,columns=['offercat_1','offercat_2','offercat_3','offercat_4','offercat_5','offercat_6','offercat_7','offercat_8'])
        
    row = row.join(offercat2)
            
            
            
    offercompany = getcomp(train.iloc[ind]['company'],comp)
    offercompany.astype('byte')
    offercompany.columns = ['offercomp_0', 'offercomp_1', 'offercomp_2', 'offercomp_3', 'offercomp_4', 'offercomp_5', 'offercomp_6',
               'offercomp_7', 'offercomp_8', 'offercomp_9', 'offercomp_10', 'offercomp_11', 'offercomp_12',
                   'offercomp_13', 'offercomp_14', 'offercomp_15', 'offercomp_16', 'offercomp_17', 'offercomp_18']
    offerbrand = getbrand(train.iloc[ind]['brand'], brand)
    offerbrand.astype('byte')
    offerbrand.columns = ['offerbrand_0', 'offerbrand_1', 'offerbrand_2', 'offerbrand_3', 'offerbrand_4', 'offerbrand_5',
                   'offerbrand_6', 'offerbrand_7', 'offerbrand_8', 'offerbrand_9', 'offerbrand_10', 'offerbrand_11',
                   'offerbrand_12', 'offerbrand_13', 'offerbrand_14', 'offerbrand_15', 'offerbrand_16', 'offerbrand_17',
                   'offerbrand_18', 'offerbrand_19']
    row = row.join(offerbrand)
    row = row.join(offercompany)
    row['repeattrips'] = train.iloc[ind]['repeattrips']
    row['offervalue'] = train.iloc[ind]['offervalue']
    row['repeater'] = train.iloc[ind]['repeater']
    row['money'] = money
            
    row.to_csv('train//' + str(ind)+'.csv',index=False)



'''












# -------------------------------------- delete this later / "other" brand and company fix / -----------------------------------------------------
'''
for i in range(len(train)):
    print(i)
    df = pd.read_csv('train//'+str(i)+'.csv')
    df = df.drop(['comp_18','brand_19'],axis = 1)
    df.to_csv('train//'+str(i)+'.csv',index=False)
    del df
'''    
  
    
  
    
  
    
#-------------------------------------------------<< feature transform >>-------------------------------------------------------


'''
# create 100 data sets with 10,000 examples each
for i in range(11,21):
    print(i)
    np.random.seed(i)
    rows = np.random.choice(range(0,150000),size=10000,replace=False)
    df = pd.read_csv('train//'+str(rows[0])+'.csv')
    cols = df.columns
    f = np.array(df.iloc[0])
    f = f.reshape([1,197])
    for j in range(1,len(rows)):
        df = pd.read_csv('train//'+str(rows[j])+'.csv')
        d = np.array(df.iloc[0])
        d = d.reshape([1,197])
        f = np.append(f,d,axis=0)
        del df
        del d
    df = pd.DataFrame(f,columns=cols)
    df.to_csv('trainsets//'+str(i)+'.csv',index=False)
'''






        
df = pd.read_csv('test//150001.csv')
cols = df.columns
f = np.array(df.iloc[0])
f = f.reshape([1,197])

for j in range(150002,160057):
    print(j)
    df = pd.read_csv('test//'+str(j)+'.csv')
    d = np.array(df.iloc[0])
    d = d.reshape([1,197])
    f = np.append(f,d,axis=0)
    del df
    del d
df = pd.DataFrame(f,columns=cols)
df.to_csv('trainsets//test.csv',index=False)



























  
    
    
    
    
    
    
    
    
    

        
        