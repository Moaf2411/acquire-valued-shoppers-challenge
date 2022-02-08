import pandas as pd
list = []
cols = ['id','chain','dept','category','company','brand','date','productsize','productmeasure','purchasequantity','purchaseamount']

with open('transactions.csv') as f:
    z = f.readline()
    id = ''
    pre = ''
    x = f.readline()
    x = x.split('\n')[0].split(',')
    list.append(x)
    while(True):
        x = f.readline()
        if x == '':
            break
        x = x.split('\n')[0].split(',')
        if list[0][0] == x[0]:
            list.append(x)
        else:
            df = pd.DataFrame(list,columns=cols)
            df.to_csv('data//'+str(list[0][0])+'.csv',index=False)
            del df
            del list[:]
            list.append(x)
            
        






