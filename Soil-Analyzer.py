import pandas as pd
import numpy as np
import pickle
df = pd.read_csv('data.csv')
n = 0
p = 0
k = 0
s = 0
oc = 0
ca = 0

mg=0
zn=0
cu=0
fe=0
mn=0
b=0
mo=0
tex=0

ph=0
ec=0

score = 0

label = []
for i in range(df.shape[0]):
    if(df.loc[i]['N']<=280):
        n=1
    elif(df.loc[i]['N']>280 and df.loc[i]['N']<=560):
        n=2
    elif(df.loc[i]['N']>560):
        n=3
        
        
    
    if(df.loc[i]['P']<=10):
        p=1
    elif(df.loc[i]['P']>10 and df.loc[i]['P']<=24.6):
        p=2
    elif(df.loc[i]['P']>24.6):
        p=3
        
        
        
    if(df.loc[i]['K']<=117.6):
        k=1
    elif(df.loc[i]['K']>117.6 and df.loc[i]['K']<=280):
        k=2
    elif(df.loc[i]['K']>280):
        k=3
        
    if(df.loc[i]['S']<=10):
        s=1
    elif(df.loc[i]['S']>10 and df.loc[i]['S']<=20):
        s=2
    elif(df.loc[i]['S']>20):
        s=3
        
    if(df.loc[i]['OC']<=0.5):
        oc=3
    elif(df.loc[i]['OC']>0.5 and df.loc[i]['OC']<=0.75):
        oc=6
    elif(df.loc[i]['OC']>0.75):
        oc=9
        
    if(df.loc[i]['Ca']<=2500):
        ca=1
    elif(df.loc[i]['Ca']>2500 and df.loc[i]['Ca']<=4000):
        ca=2
    elif(df.loc[i]['Ca']>4000):
        ca=3   
        
    if(df.loc[i]['Mg']<=450):
        mg_=1
    elif(df.loc[i]['Mg']>450 and df.loc[i]['Mg']<=1000):
        mg_=2
    elif(df.loc[i]['Mg']>1000):
        mg_=3 
        
    if(df.loc[i]['Zn']<=1):
        zn_=1
    elif(df.loc[i]['Zn']>1 and df.loc[i]['Zn']<=12):
        zn_=2
    elif(df.loc[i]['Zn']>12):
        zn_=3 
        
    if(df.loc[i]['Cu']<=0.4):
        cu_=1
    elif(df.loc[i]['Cu']>0.4 and df.loc[i]['Cu']<=1.6):
        cu_=2
    elif(df.loc[i]['Cu']>1.6):
        cu_=3 
    if(df.loc[i]['Fe']<=20):
        fe_=1
    elif(df.loc[i]['Fe']>20 and df.loc[i]['Fe']<=100):
        fe_=2
    elif(df.loc[i]['Fe']>100):
        fe_=3 
    if(df.loc[i]['Mn']<=12):
        mn_=1
    elif(df.loc[i]['Mn']>12 and df.loc[i]['Mn']<=12):
        mn_=2
    elif(df.loc[i]['Mn']>25):
        mn_=3 
    if(df.loc[i]['B']<=450):
        b_=1
    elif(df.loc[i]['B']>450 and df.loc[i]['B']<=1000):
        b_=2
    elif(df.loc[i]['B']>1000):
        b_=3 
    score = ((ph+ec+n+p+k+s+oc+ca+mg+zn+cu+fe+mn+b+mo+tex)*10)/24
    
    label.append((score))
my_formatted_label = [ '%.3f' % elem for elem in label ]

sadf = pd.DataFrame(my_formatted_label)

sadf.columns = ['SFI'] 

sadf = pd.concat((df, sadf), axis=1)
sadf=sadf.drop(['District'], axis=1)
SA_data = sadf.to_csv('SA_Data.csv', index = True)
sadf_c = sadf.copy(deep=True)
sadf_c[['pH', 'EC', 'OC', 'N', 'P', 'K', 'S', 'Ca', 'Mg', 'Zn', 'Cu',
       'Fe', 'Mn', 'B', 'Mo', 'Tex', 'SFI']] = sadf_c[['pH', 'EC', 'OC', 'N', 'P', 'K', 'S', 'Ca', 'Mg', 'Zn', 'Cu',
       'Fe', 'Mn', 'B', 'Mo', 'Tex', 'SFI']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution
sadf_c['pH'].fillna(sadf_c['pH'].mean(), inplace=True)

sadf_c['EC'].fillna(sadf_c['EC'].median(), inplace=True)
sadf_c['OC'].fillna(sadf_c['OC'].median(), inplace=True)
sadf_c['N'].fillna(sadf_c['N'].median(), inplace=True)
sadf_c['P'].fillna(sadf_c['P'].median(), inplace=True)
sadf_c['K'].fillna(sadf_c['K'].median(), inplace=True)
sadf_c['S'].fillna(sadf_c['S'].median(), inplace=True)
sadf_c['Ca'].fillna(sadf_c['Ca'].median(), inplace=True)
sadf_c['Mg'].fillna(sadf_c['Mg'].median(), inplace=True)
sadf_c['Zn'].fillna(sadf_c['Zn'].median(), inplace=True)
sadf_c['Cu'].fillna(sadf_c['Cu'].median(), inplace=True)
sadf_c['Fe'].fillna(sadf_c['Fe'].median(), inplace=True)
sadf_c['Mn'].fillna(sadf_c['Mn'].median(), inplace=True)
sadf_c['B'].fillna(sadf_c['B'].median(), inplace=True)
sadf_c['P'].fillna(sadf_c['Mo'].median(), inplace=True)

sadf_c['Tex'].fillna(sadf_c['Tex'].mean(), inplace=True)
sadf=sadf_c
X = sadf.drop(columns=['id','label','SFI'])
y = sadf['SFI']

# Standardize the data 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
y.unique()
X = (X - X.mean())/X.std()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


A = sadf.drop(columns=['id','label','SFI'])
B = sadf['SFI']
values = np.array(B)
features = np.array(A)
train_features, test_features, train_values, test_values = train_test_split(features, values, test_size = 0.25, random_state = 42)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_values);
test_values=test_values.astype('float')
sadf=sadf.astype('float')

# Creating a pickle file for the classifier
filename = 'Soil-Analysis-Prediction.pkl'
pickle.dump(rf, open(filename, 'wb'))

