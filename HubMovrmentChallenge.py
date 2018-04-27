#
# Author: L. Salud, April 26.2018
#
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.getcwd() # Get and place .py in same directory as .xls initially

os.chdir('./') # Path to .xls file

from pandas import read_excel      
df = read_excel('rssi_data_challenge2.xls')
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.head(n=5)
df.tail()

df.describe(include = 'all')


pca = PCA(n_components=3)

X = df[['attributesfirstnodemeanrssi','attributessecondnodemeanrssi', 'attributesthirdnodemeanrssi','attributesfourthnodemeanrssi','attributesfifthnodemeanrssi','attributessixthnodemeanrssi']]

X.loc[1:10]

list(df) 

S = df[['attributesfirstnodestddevrssi','attributessecondnodestddevrssi', 'attributesthirdnodestddevrssi','attributesfourthnodestddevrssi','attributesfifthnodestddevrssi','attributessixthnodestddevrssi']]

S.loc[1:10]

S.columns = X.columns

S1 = S.replace(0.00, 0.01)

CD = (X*X) + 300*S1 # TODO: Refine
# TODO: See if any of these and other publications may be applicable
#  https://www.hindawi.com/journals/jcnc/2013/185138/abs/
#  https://www.ncbi.nlm.nih.gov/pubmed/28895879
#  https://dl.acm.org/citation.cfm?id=2790093
#  https://en.wikipedia.org/wiki/Short-time_Fourier_transform

# Standardizing the features
CD = StandardScaler().fit_transform(CD)
CD[1:10]

principalComponents = pca.fit_transform(CD)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal_component_1', 'principal_component_2', 'principal_component_3'])

principalDf = principalDf[principalDf['principal_component_1'] < 18 ]

max(principalDf['principal_component_1'])

finalDf = pd.concat([principalDf, df[['movestate']]], axis = 1)

# Visualize
fig = plt.figure(figsize = (26, 26))
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 2', fontsize = 15)
ax.set_ylabel('Principal Component 1', fontsize = 15)
#ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = ['nonmove', 'move']
colors = ['r', 'b', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['movestate'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal_component_2']
               , finalDf.loc[indicesToKeep, 'principal_component_1']
#               , finalDf.loc[indicesToKeep, 'principal_component_3']
               , c = color
               , alpha=0.5, 
               label="Point")
ax.legend(targets)
ax.grid()
# TODO: Run SVM for cluster separation