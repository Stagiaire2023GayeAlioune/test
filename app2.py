# Contents of ~/my_app/streamlit_app.py
import streamlit as st
from PIL import Image, ImageOps
import ydata_profiling   
import numpy as np
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg , predict_model as predict_model_reg, plot_model as plot_model_reg,create_model as create_model_reg
from pycaret.classification import setup, compare_models, blend_models, finalize_model, predict_model, plot_model,create_model
from pycaret.classification import *
#from pycaret.regression import *
from pycaret import *
import tensorflow as tf
import keras.preprocessing.image
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications import VGG16
from keras.models import Model
import os;
#import cv2 
import seaborn as sns
from PIL import Image
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import os
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt;
from scipy.optimize import curve_fit,fsolve
from scipy.signal import savgol_filter
from scipy import signal
import sympy as sp 
from scipy.integrate import quad
import scipy.integrate as spi
from sklearn import preprocessing
from scipy import stats
from sklearn.linear_model import LinearRegression
#from tkinter import *
#from tkinter import filedialog
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler 
import seaborn as sns
from sympy import symbols
from sympy import cos, exp
from sympy import lambdify
import statsmodels.formula.api as smf
from sympy import *
import csv
from scipy import optimize
from sklearn.metrics import r2_score#pour calculer le coeff R2
from sklearn.linear_model import RANSACRegressor
from colorama import init, Style
from termcolor import colored
import streamlit as st

url ="https://www.linkedin.com/in/alioune-gaye-1a5161172/"
@st.cache_data
def load_data(file):
    data=pd.read_csv(file)
    return data


def Code_identification():
    def main():
        st.sidebar.markdown('<h1 style="text-align: center;">Les codes pour la partie identification:  🎈</h1>', unsafe_allow_html=True)
        st.code('''
#!/usr/bin/env python
# coding: utf-8

# ## Importation des biliotheques necessaires

# In[53]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt  
from tkinter import filedialog
from tkinter import *
import seaborn as sns
from sklearn.model_selection import train_test_split,KFold , cross_val_score,cross_validate 
from sklearn.tree import DecisionTreeClassifier ,plot_tree,ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,precision_score ,classification_report,RocCurveDisplay , auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.metrics import recall_score,fbeta_score, make_scorer,roc_curve ,roc_auc_score
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier
from sklearn.svm import SVC
import time                                                         
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pycaret.regression import setup, compare_models, blend_models, finalize_model, predict_model, plot_model
from pycaret.classification import *
from scipy.stats import kstest, expon
from scipy.stats import chisquare, poisson
from fitter import Fitter, get_common_distributions, get_distributions
from sklearn.decomposition import PCA
from scipy.stats import expon, poisson, gamma, lognorm, weibull_min, kstest,norm
import scipy
import scipy.stats
from mlxtend.plotting import plot_pca_correlation_graph 
import scipy.stats as stats
from sklearn.ensemble import GradientBoostingClassifier
import pickle ### on utilise ce bibliotheque pour sauvegarder notre modél , qui nous servira pour la partie deployement .  


# ## Fonction qui nous permettte de selectionner n'importe quel fichier
# dans l'ordinateur 

def browseFiles():
	filename = filedialog.askopenfilename(initialdir = "Z:\1_Data\1_Experiments\1_FENNEC\2_Stagiaires\2022_Alvin\7 Samples\ATMP_DTPMP",
										title = "Select a File",
										filetypes = (("Csv files",
														"*.csv*"),
													("all files",
														"*.*")))
	return(filename)


# ## Appel de la fonction  browseFiles() :La base de donnée est le fichier nommé "Base_amecT_bdt_gly (3).csv" 

VAR=browseFiles()

VAR

df2=pd.read_csv(VAR,sep=',',index_col=0)

df2


# ## Aprés avoir visualiser notre base de donnée , on a cpnstaté que les deux premieres colonnes ne nous interessent pas , aussi les 5 derniéres colonnes . Mais avant de supprimer les colonnes A, D et G , on essaye de faire un encodage de ces derniéres en creant une variable cible nommé "clf " 

df2=df2.drop(["A+D","A+G","D+G","sum"],axis=1)  ## Supression des colonne ""A+D","A+G","D+G","sum"" 


# ## Aprés la supresion 

df2

# ## Tran sformation des colonnes A , D , G en une seule variable cible 

# ### Label pour les 3 polluants 

## Fonction pour créer un label pour  trois polluants 
clf=[]
for i in range(len(df2['A'])):
    if(df2['A'][i]!=0. and df2['D'][i]!=0 and df2['G'][i]!=0):
        clf.append('[A,D,G]')
    if(df2['A'][i]!=0 and df2['D'][i]!=0):    
         clf.append('[A,D]')
    if(df2['A'][i]!=0 and df2['G'][i]!=0):    
         clf.append('[A,G]')        
    if(df2['G'][i]!=0 and df2['D'][i]!=0):    
         clf.append('[G,D]')     
    if(df2['A'][i]==0 and df2['D'][i]==0 and df2['G'][i]!=0):    
         clf.append('[G]')     
    if(df2['A'][i]==0 and df2['D'][i]!=0 and df2['G'][i]==0):
         clf.append('[D]')   
    if(df2['A'][i]!=0 and df2['D'][i]==0 and df2['G'][i]==0):
         clf.append('[A]')                


# 

# ### la variable cible 

clf


# ## Maintenant , on rajoute la colonne clf dans notre base de donnée .

df2['clf']=clf


df2


# #### Distribution des differents polluants 


ax=df2['A'].hist(bins=15, density=True , stacked=True, color='green' , alpha=0.8)
df2['A'].plot(kind='density' , color='blue')
ax.set(xlabel='A')


ax=df2['D'].hist(bins=15, density=True , stacked=True, color='green' , alpha=0.8)
df2['D'].plot(kind='density' , color='yellow')
ax.set(xlabel='D')

## Pour le trisiémé polluant
ax=df2['G'].hist(bins=15, density=True , stacked=True, color='green' , alpha=0.6)
df2['G'].plot(kind='density' , color='red')
ax.set(xlabel='G')


### On aura besoin plud d'experience avec les melanges de deux polluants surtout le polluants G . En effet , on a moins de données pour ce polluant 
## 1 ) Spectre d'excitation  avec un seul polluant : le G surtout et si possible le D 
## 2) Spectre d'excitation  avec le melange de polluants : (G,D), (G,A) , (G,D) .....


# ## Les detailes de notre base de donnée 

df2.info()

n=len(df2.columns)
numerical = [var for var in df2.columns] ## les diferentes colonnes de notre jeu de donnée .....
features = numerical
##Recherce des fichiers dupliquer , pour  aprés supprimer les doublons . 
print(df2[features[0:(n-4)]].duplicated().sum())


# 

# ## Supression des lignes doubles en gardant une d'eux 

df2=df2.drop_duplicates()

df2


# ## Enregistrement de la base de donnée netoyer et on eleve maintenant les colonne A , D et G qui ne nous interessent pas  . 


from pathlib import Path
df3=df2.drop(["A","G","D"],axis=1)
filepath=Path("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/machine_Learning/base_de_donnee/csv_Melange de polluant/base de donnée/base_de_donnée_final.csv")
filepath.parent.mkdir(parents=True, exist_ok=True)
df3.to_csv(filepath,index=False)


# ## Ainsi notre base de donnée netoyer et préparer se nomme "base_de_donnée_final.csv"

# ## Decomposition des données  en targets et variables ( variables explivcatives et expliquées )



df=df2
n=len(df.columns)
X=df[df.columns[2:(n-4)]] # on prend les variables numériques 
y=df[df.columns[-1]] # le target , le variables cible ("clf")  

y


# ## Statistiques descriptives


import seaborn as sns
colors = ['#06344d', '#00b2ff' , '#00b1ff' ]
sns.set(palette = colors, font = 'Serif', style = 'white', 
        rc = {'axes.facecolor':'#f1f1f1', 'figure.facecolor':'#f1f1f1'})

fig = plt.figure(figsize = (10, 6))
ax = sns.countplot(x = 'clf', data = df)


for i in ax.patches:
    ax.text(x = i.get_x() + i.get_width()/2, y = i.get_height()/7, 
            s = f"{np.round(i.get_height()/len(df)*100, 0)}%", 
            ha = 'center', size = 50, weight = 'bold', rotation = 90, color = 'white')

    
plt.title("histogramme des polluants", size = 20, weight = 'bold')

plt.annotate(text = "polluant A", xytext = (-0.4, 140), xy = (0.1, 100),
             arrowprops = dict(arrowstyle = "->", color = 'black', connectionstyle = "angle3, angleA = 0, angleB = 90"), 
             color = 'green', weight = 'bold', size = 14)

plt.annotate(text = "polluant D ", xytext = (0.15, 150), xy = (1,110), 
             arrowprops = dict(arrowstyle = "->", color = 'black', connectionstyle = "angle3, angleA = 0, angleB = 90"), 
             color = 'red', weight = 'bold', size = 14)

plt.annotate(text = "polluant [A,D] ", xytext = (1, 150), xy = (2, 110), 
             arrowprops = dict(arrowstyle = "->", color = 'black',  connectionstyle = "angle3, angleA = 0, angleB = 90"), 
             color = 'blue', weight = 'bold', size = 14)


plt.xlabel('classes', weight = 'bold')
plt.ylabel('observation', weight = 'bold')


# ## On constate que , 35% des données sont des spectres avec 100% polluant A , 27%  des données sont des spectres  avec 100%  de polluant D et les 12 %  qui repreente le melange [A,D] , 4% le polluant [A,G] et 4 % du melange [G,D] .  Ainsi ,  il faut prevoir plus de données avec seulement le polluant G  et les mélanges des polluants . Les codes pour refaire une autre base de données , faire l'apprentissage automatique , créer votre model , enregistrement du model (le pipeline) et le deploiment du model et faire des predictions avec le model deployé seront automatique dans une application qu'on va créer à la fin de ce stage .

# ## Etude des variables descriptives 

X.describe()


numerical = [var for var in df.columns ]
features = numerical
colors = ['blue']
df[features[2:(n-4)]].hist(figsize=(9, 6), color=colors, alpha=0.7)
plt.show()


# ### On constate que les Ai suivent la meme distribution ::: qui resemble à peut prés a une loi logarithmique ou exponentielle deroissante 

# ## LA distribution des variables explicatives 


a=[df['A1'],df['A2'],df['A3'],df['A4']]
for aa in a:
    f = Fitter(aa,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                         'poisson','norm','exp'])
    f.fit()
    f.summary()
    print(aa.name,f.get_best(method = 'sumsquare_error'))


# ## Correlation entre les variables 

# matrice de corrélation 
plt.figure(figsize=(12,6))
corr_matrix = df[features[2:(n-4)]].corr()
sns.heatmap(corr_matrix,annot=True)
plt.show()


# # Comme on peut le constater A1 est fortement positevement correlé avec E1 et faiblement corrélé avec les autres variables ,C1 fortement negativement corrélé avec C2 et faiblement corrélé avec les autres variables  

# ## Coorelation entre les variables explicatives 


## l'encodage pour la partie sklearn 
from sklearn.preprocessing import LabelEncoder # nous permet de faire l'encodage , avec ordinalencoder fait la même mais avec plusieurs variable encoder
encoder=LabelEncoder()
y_code=encoder.fit_transform(y)


# ## Utilusons le PCA pour regarder les corélations des variables 

#grâce à sklearn on peut importer standardscaler .

ss=StandardScaler()# enleve la moyenne et divise par l'ecartype
ss.fit(X)
X_norm=ss.transform(X)# tranform X 
figure, correlation_matrix = plot_pca_correlation_graph(X_norm, features[2:(n-4)], dimensions=(1, 2),figure_axis_size=8)
figure, correlation_matrix = plot_pca_correlation_graph(X_norm, features[2:(n-4)],dimensions=(1, 3),figure_axis_size=8)
figure, correlation_matrix = plot_pca_correlation_graph(X_norm, features[2:(n-4)],dimensions=(1, 4),figure_axis_size=8)


# # Ici on utilise pycaret pour chercher notre meilleures modél de prediction et ses pérformances . 

# Dataset Sampling
def data_sampling(dataset, frac: float, random_seed: int):
    data_sampled_a = dataset.sample(frac=frac,
                                    random_state=random_seed)
    data_sampled_b =  dataset.drop(data_sampled_a.index).\
    reset_index(drop=True)
    data_sampled_a.reset_index(drop=True, inplace=True)
    return data_sampled_a, data_sampled_b  


# ### Separation des données  en donnée  d'entrainement et de test ( 75%  ,  30%)

# On dois supprimer les collonnes A et D et garder  que la colonne label qu'on a encoder (clf) , car c'est cette qui represente notre target .

# ## data_sampling est une fonction que j'ai creer pour separer la base de donnée en base de données d'apprentissage et de test pour simplifier les calcul


train, unseen = data_sampling(df, 0.75, RANDOM_SEED)
train=train.drop(["A","D","G"], axis=1)
unseen=unseen.drop(["A","D","G","clf"], axis=1)
l=len(train.columns)
train=train[train.columns[2:l]]
unseen=unseen[unseen.columns[2:l]]
unseen


# # Ici , le dataframe unseen sera utilisé pour la prediction aprés le deploiement , on va l'aapelé "base_de_donnée_de_test.csv" 


### Enregistrement de notre base de donnée de test  
from pathlib import Path
filepath=Path("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/machine_Learning/base_de_donnee/csv_Melange de polluant/base de donnée/base_de_donnée_de_test.csv")
filepath.parent.mkdir(parents=True, exist_ok=True)
unseen.to_csv(filepath,index=False)


train


# ### On commence par créer un setup avec  les données non normalisés et faire l'apprentisage puis comparer avec celui des données normalisée et celui normaliser + PCA . 

# Ici on spécifit les données utilisés , si on doit les normaliser ou pas , utiliser acp ou pas , donner le pourcentage train ...
# ici on test avec données sans normalisés 
colonne=features[2:(n-4)]+['clf']
setup_data = setup(data =train,target = 'clf',index=False,
                   train_size =0.8,categorical_features =None,
                   normalize = False,normalize_method = 'zscore' ,remove_multicollinearity =True
                   ,multicollinearity_threshold =0.8,pca =False, pca_method =None,
                   pca_components = None,log_experiment='mlflow',experiment_name="polluant_heterogene")


# ### Le model à été deployé et sauvegarder dans  Mlflow ...n'empeche le model sera aussi enregistrer sous format pkl ou H dans mon dosier , puis dans l'application prévue avent la fin de l'stage 


# ### On note que l'encodage pour  les variables cathégorielles (3 polluiants) est  :: [A,D]: 0, [A,G]: 1, [A]: 2, [D]: 3, [G,D]: 4, [G]: 5

# 
## PyCaret dispose d’un module NLP qui peut automatiser la plupart des choses ennuyeuses, comme l’abaissement de la casse, la suppression des mots vides, le stemming, etc. Donc, une bonne partie de cette partie consiste simplement à configurer PyCaret pour fonctionner. Importons le module.


# ## Comparaison de plusieurs modeles en fonction des metriques comme l'accurancy...

top_model = compare_models()

#Le meilleur modèle  est soit EXTRA trees Classifier ou Light Gradient Boosting Machine	 , ces modèles ont obtenu un meilleur score sur les autres métriques, prenons  EXTRA trees Classifier comme modèle de base. Ajustez le modèle pour voir s’il peut être amélioré.
# ## Ajustement des parametres  du model 

tuned_model = tune_model(top_model[1])


# # Le modèle accordé ne reçoit aucune amélioration, donc le modèle de base est le meilleur.
# Il est temps de construire un ensemble d’ensachages.

bagged_model = ensemble_model(tuned_model) 


# ## Et maintenant un Boosting Ensemble.

boosted_model = ensemble_model(top_model[1],method="Boosting") 

Le modèle initial (top_model)  est le meilleur et est enregistré comme le meilleur modèle et utilisé pour prédire sur l’ensemble de test.
# ### Une prediction de notre model avec les données de test générées par pycaret

best_model = top_model
predict_model(best_model)


# ### On a obtenue une bonne prediction avec de meilleurs métriques voir proche de 1 , donc notre model est capable de bien classée les polluants 

# ## Affichzge des hyperparamètres du modèle.

plot_model(best_model, plot="parameter")


# ## les performances du model 

final_model1 = best_model
plot_model(final_model1,plot='auc')
plot_model(final_model1,plot='class_report')
plot_model(final_model1 , plot='boundary')


# ### Les variables les plus pertinantes 

plot_model(final_model1,plot='feature')

# ### Finalisons notre model pour aprés enrégistrer le pipeline 


final_model_ = finalize_model(final_model1)
final_model_


# ## Resumé des performances du model 


evaluate_model(final_model1)#Cette fonction affiche une interface utilisateur pour analyser les performances


# ### Sauvegarder le model et passons au deploiement 

# Ainsi , notre model est préte l'emploi , le dep^loiement  , car elle regroupe mainteenant touts les éléments necessaires ppour son deploieement ::Exxemple: pour les entreprise ;pretes pôur l'zmploie business 


save_model(final_model_,"best_classS_model1")


# ### Maintenant essayons avec les données normalisé 


setup_data = setup(data =train,target = 'clf',
                   train_size =0.8,categorical_features =None,index=False,
                   normalize = True,normalize_method = 'zscore' ,remove_multicollinearity =True,log_experiment=True,experiment_name="polluant_heterogene",
                   multicollinearity_threshold =0.8,pca =False, pca_method =None,
                   pca_components = None,numeric_features =features[2:(n-4)])



top_model = compare_models()


# 
# 


type(top_model)


# ### Reglage des hyperparametres

tuned_model = tune_model(top_model[1]) 

#Le modèle accordé ne reçoit aucune amélioration, donc le modèle de base est le meilleur. Il est temps de construire un ensemble d’ensachages.
# In[89]:


bagged_model = ensemble_model(tuned_model) 


# On a  les meilleurs performances avec le model (top_model )

#  

# ## Passons à verifier les performances du model par des graphes 

# In[100]:


final_model = top_model
plot_model(final_model,plot='auc')
plot_model(final_model,plot='class_report')
plot_model(final_model,plot='confusion_matrix')
plot_model(final_model,plot='feature')
plot_model(final_model , plot='boundary')


# On concatete que avec les données non normaliser et sans faire  l'ACP , on obtient les meilleurs perforùance avec notre model . 

# ## Ainsi , on enregistre le model obtenu avec les données non normaliés , ensuite faire le deploiement .
On garde le model (final_model1)
# In[103]:


type(final_model1)


# ## Notre pipeline 

# In[110]:


final_model_


# ### Aprés comparaison , on a constaté que la meilleur facon de faire une classification des polluant est d'utiliser les données non normaliser , en effet , avec ces derniéres l'algorithme commet peut d'erreurs (confusion ) , en plus on a les meilleurs perdformance aussi .
# 

# ## Comparaison avec skeatlearn 

# ## Separation des données en data d'enprentissage et de test 

# In[64]:





# In[68]:


# on utilise train_test_split de sklearn 
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=1)
X_train
Y_train


# In[69]:


pour_A=np.sum(Y_train=='[A]')/len(Y_train)
pour_D=np.sum(Y_train=='[D]')/len(Y_train)
pour_G=np.sum(Y_train=='[A,D]')/len(Y_train)
print(" A pour le train :",np.sum(Y_train=='[A]'),"D pour le train :",np.sum(Y_train=='[D]'),"[A,D] pour le train :",np.sum(Y_train=='[A,D]'))
print("pourcentage d'exemple A :",pour_A*100,"%")
print("pourcentage d'exemple D :",pour_D*100,"%")
pour_pos=np.sum(Y_test=='[A]')/len(Y_test)
pour_neg=np.sum(Y_test=='[D]')/len(Y_test)
pour_mixte=np.sum(Y_test=='[A,D]')/len(Y_test)

print(" A pour le test :",np.sum(Y_test=='[A]'),"D pour le test :",np.sum(Y_test=='[D]'))
print("pourcentage d'exemple A :",pour_pos*100,"%")
print("pourcentage d'exemple D :",pour_neg*100,"%")
print("Nombre d'éléments dans le jeu d'entraîntement : {}".format(len(X_train)))
print("Nombre d'éléments dans le jeu de test : {}".format(len(X_test)))


# ## On a eu 43% de polluant A dans le train et 49% dans le test , pour le polluant D on a eu 33.9 % dans le train et 33.4 % dans le test .
# ## De plus le jeux de donnée d'entrainement conttient 1027 données et le test 257 données . 

# # **<center><font color='blue'>  Comparaison de plusieurs algorithmes d’apprentissage :</font></center>**
# - On vas essayer de construire un dictionnaire de plusieurs algorithmes pour comparer plusieurs algorithmes sur une même validation croisée
# - On vas utiliser la technique de KFold cross validate 

# In[106]:


from sklearn.tree import DecisionTreeClassifier ,plot_tree,ExtraTreeClassifier


# In[70]:


#Comparaison de plusieurs algorithmes d’apprentissage , ici clfs regroupe plusieurs algorithmes d'apprentisssage en meme temps , puis 
### on fait la comparaison de ces algorithmes en utilisant l'accurancy (' le pourcentage de bonne prediction ')
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
clfs = {
'RF': RandomForestClassifier(n_estimators=100, random_state=1),
'NBS' : GaussianNB() ,
'decision_tree' : DecisionTreeClassifier(criterion='gini',random_state=1) ,
'id3' : DecisionTreeClassifier(criterion='entropy',random_state=1),
'MLP' : MLPClassifier(hidden_layer_sizes=(100,2),activation='tanh',solver='lbfgs',random_state=1, max_iter=300),
'BAG': BaggingClassifier( n_estimators=100, random_state=1),
'AdA': AdaBoostClassifier(n_estimators=100, random_state=1),# algo de boosting , creer un 1er classifier , prédit , il prend ce qui sont mal classés
'Gau': GaussianNB(),
'LG' :LogisticRegression(),
'svc': SVC(),
'Ext': ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                    criterion='gini', max_depth=None, max_features='sqrt',
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_samples_leaf=1,
                     min_samples_split=2, min_weight_fraction_leaf=0.0,
                    n_estimators=100, n_jobs=-1, oob_score=False,
                     random_state=1052, verbose=0, warm_start=False),
'gbc':GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)}



## La fonction run-classifiers qui regroupe les algorithmes ci-dessus et les compare par leur accurancy, precission , courbe ROC , recal 
def run_classifiers(clfs , X,Y) : 
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    df=pd.DataFrame(columns=['algo','accuracy','precision','rappel','air sous la courbe'])
    for i in clfs:
        clf = clfs[i]
        start = time.time()  
        scoring = {'acc': 'accuracy', 'rec' : 'recall', 'prec' : 'precision','roc' : 'roc_auc'} 
        scores = cross_validate(clf, X, Y, scoring=scoring, cv=kf, return_train_score=False)
        print(f"\033[031m {i} \033[0m",'\n')
        print("Accuracy for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(scores['test_acc']), np.std(scores['test_acc'])))
        print("Recall for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(scores['test_rec']), np.std(scores['test_rec'])))
        print("Precision for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(scores['test_prec']), np.std(scores['test_prec'])))
        print("Aire sous la courbe for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(scores['test_roc']), np.std(scores['test_roc'])))
        print ('time', time.time() - start, '\n\n')
        
        
        df=df.append({'algo':i,'accuracy':np.mean(scores['test_acc']),'precision':np.mean(scores['test_prec']),
                     'rappel':np.mean(scores['test_rec']),'air sous la courbe': np.mean(scores['test_roc'])},ignore_index=True)
        # Ajouter la matrice de confusion
        y_pred = clf.fit(X, Y).predict(X)
        cm = confusion_matrix(Y, y_pred)
        print("Confusion Matrix for {0}:\n{1}".format(i, cm))
        # Tracer la matrice de confusion avec les valeurs
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Matrice de confusion - {0}".format(i))
        plt.colorbar()
        classes = np.unique(Y)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # Afficher les valeurs dans la matrice
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.xlabel('Classe prédite')
        plt.ylabel('Classe réelle')
        plt.tight_layout()
        plt.show()       
    return(df)



#### évaluation des algorithmes de classification ci-dessus en utilisant l'accurancy . 
def evaluate_classifiers(X, y, classifiers):
    accur = {}
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=5)
        accuracy = scores.mean()
        accur[name]=accuracy

    return accur


# ## On pourai aussi utiliser  LazyClassifier  qui permette de comparer plusieurs algo d'apprentissage en meme temps . 

# In[75]:


pip install lazypredict


# In[ ]:


from lazypredict.Supervised import LazyClassifier 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=5,random_state=123)
clf=LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions=clf.fit(X_train, X_test, y_train, y_test)
print(models)

## Essayons de regerder ExtratreeClassifier à part pour comparer les resultats avec les autres algorithmes de classifications definies ci-dessus 
# In[109]:


kf = KFold(n_splits=10, shuffle=True, random_state=0)
scores = cross_validate(clfs['Ext'],X_train,Y_train , cv=kf, return_train_score= False)
scores


# In[ ]:





# ##  Comparaison des modéles en utilisant les metriques ('accurancy  ... ')

# In[111]:


accuracy=evaluate_classifiers(X_train, Y_train, clfs)
accuracy


# ### Cherchons l'algorithme la plus efficace pour classer les polluants avec la plus grande accurancy .

# In[112]:


ac=pd.DataFrame(list(accuracy.items()),columns=['algo', 'accuracy'])
ac.style.highlight_max(subset=['accuracy'], color='orange')  

## On constae que l'algorithme du extratreeClassifier est beaucoup plus performant pour classser les polluants avec un score de 0.877 % , comme en  pycaret aussi ..
# ### performance de Chacun des algorithmes 

# In[113]:


r=run_classifiers(clfs , X_train,Y_train)


# ### Regroupons tout les algorithmes avec leur precissions dans un dataframe allant du plus performant au moins performant 

# In[117]:


r_sorted = r.sort_values(by='accuracy',ascending=False)
r_sorted.style.highlight_max(subset=['accuracy'], color='orange')


# In[130]:


# fonction qui nous permet de  comparer les trois models  
def Classifieur(Xtrain,Xtest,Ytrain,Ytest,clfs):
    df=pd.DataFrame(columns=['algo','accuracy','precision','air'])
    for i in clfs:
        clf = clfs[i]
        print(f"\033[031m {clf} \033[0m")
        clf.fit(Xtrain,Ytrain)
        YDT=clf.predict(Xtest)
        print('Accuracy :{0:.2f}'.format(accuracy_score(Ytest,YDT)*100))# {0:0.2f} deux chiffres aprés la virgule 
        print('roc :{0:.2f}'.format(roc_auc_score(Ytest,YDT)*100),'\n')
        print('Precision :{0:.2f}'.format(precision_score(Ytest,YDT)*100),'\n')
        df=df.append({'algo':i,'accuracy':accuracy_score(Ytest,YDT)*100,
                     'precision':precision_score(Ytest,YDT)*100,
                     'air':roc_auc_score(Ytest,YDT)*100},ignore_index=True)
        cm = confusion_matrix(Ytest, YDT, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['A', 'D'])
        disp.plot()
        plt.show()
    return(df)


# In[ ]:


c=Classifieur(X_train,X_test,Y_train,Y_test,clfs)


# In[ ]:


c1 = c.sort_values(by='accuracy',ascending=False)
c1.style.highlight_max(subset=['accuracy'], color='orange')


# 

# In[116]:





# In[ ]:





# In[ ]:





# ## Maintenant cherchons les variables les plus significatifs (importantes pour  classifier les polluants )  , avec des tests d'hypothéses  sous R .... Ensuite , passons au deployement de notre model avec mlfow puis l'utiliser pour la prediction....

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[56]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[34]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




''', language='python')
    if __name__ == "__main__":
         main()    
    



def identification():
    st.sidebar.markdown('<h1 style="text-align: center;">La partie Identification des polluants:  🎈</h1>', unsafe_allow_html=True)
    def main():
        st.markdown('<h1 style="text-align: center;">Identification des polluants</h1>', unsafe_allow_html=True)
        st.markdown('Chercher un model de classification le plus efficace qui permet de mieux classers les polluants :  🎈', unsafe_allow_html=True)
        st.markdown('Charger la base de donnée</h1>',unsafe_allow_html=True)
        col3,col4,col5=st.sidebar.columns(3)
        col3.image("https://www.researchgate.net/profile/Ousama-Aamar/publication/324429959/figure/fig6/AS:631621631881292@1527601740766/Spectres-dexcitation-et-demission-de-fluorescence-de-lHpD-3fJg-ml-en-PBS-2-SVF.png", use_column_width=True)
        col4.image("https://ilm.univ-lyon1.fr/templates/mojito/images/logo.jpg", use_column_width=True)
        col5.image("https://formation-professionnelle.universite-lyon.fr/var/site/storage/images/3/3/5/0/533-17-fre-FR/Lyon-1-Claude-Bernard.png", use_column_width=True)
        st.sidebar.write("<p style='text-align: center;'>Alioune Gaye : Stagiaire ILM (%s)</p>" % url, unsafe_allow_html=True)
        st.sidebar.write("<p style='text-align: center;'> Encadrent:Martini Matéo </p>", unsafe_allow_html=True)
        st.sidebar.write("<p style='text-align: center;'>Apprentissage par classification.</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>Dans cette partie vous allez : :</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>1 - Chargement la base de donnée en premier lieu</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>2 - Ensuite vous tapez sur [Statistiques descriptives] , l' analyse exploratoire des données s'affiche</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>3 - Aprés,vous sélectionez la variable cible(clf) et la méthode d'apprentissage (classification)</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>4 - En cliquant sur [les performances du model] le model se construit tout seul et toutes les performances modèle s'afficheront</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>5 - Ainsi, vous pouvez téléchargemer le pipeline du modèle pour le deploiement</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>6 -Maintenant le model est deja deploiement et préte à faire des Prédictions(</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>6 -Dans la partie [prediction avec le model deployé] ,importer votre fichier et la prediction s'affichera(</p>", unsafe_allow_html=True)
        file = st.file_uploader("entrer les données ", type=['csv'])
        if file is not None:
            df=load_data(file)
            #type=st.selectbox("selectionner le target",["Homogene","Heterogene"])
            n=len(df.columns)
            X=df[df.columns[:(n-1)]]# on prend les variables numériques 
            y=df[df.columns[-1]] # le target
            st.dataframe(df)
            pr = df.profile_report()
            if st.button('statistique descriptive'):
                 st_profile_report(pr)
            if st.button('Save'):
                 df.to_csv('data.csv')
            target=st.selectbox("selectionner le target",df.columns)
            methode=st.selectbox("selectionner la méthode ",["Regression","Classification"])
            df=df.dropna(subset=target)
            if methode=="Classification":
                if st.button(" les performances du modèle "):
                     setup_data = setup(data=df,target = target,
                        train_size =0.75,categorical_features =None,
                        normalize = False,normalize_method = None,fold=5)
                     r=compare_models(round=2)
                     save_model(r,"best_model")
                     st.success("youpiiiii  votre model de  classification est prete \U0001F604")
                     st.write("Maintenant verifions  les performances de votre model de  classification \U0001F604")
                
                     final_model1 = create_model(r,fold=5,round=2)
                     #final_model1=best_class_model
                     #final_model1=load_model('best_model.pkl')
                     col5,col6=st.columns(2)
                     col5.write('AUC')
                     plot_model(final_model1,plot='auc',save=True)
                     col5.image("AUC.png")
                     col6.write("class_report")
                     plot_model(final_model1,plot='class_report',save=True)
                     col6.image("Class Report.png")
                  
                     col7,col8=st.columns(2)
                     col7.write("Confusion_matrix")
                     plot_model(final_model1,plot='confusion_matrix',save=True)
                     col7.image("Confusion Matrix.png")
                     tuned_model = tune_model(final_model1,optimize='AUC',round=2,n_iter=10);# optimiser le modéle
                     col8.write("boundary")
                     plot_model(final_model1 , plot='boundary',save=True)
                     col8.image("Decision Boundary.png")
                    
                     col9,col10=st.columns(2)
                     col9.write("feature")
                     plot_model(estimator = tuned_model, plot = 'feature',save=True)
                     col9.image("Feature Importance.png")
                     col10.write("learning")
                     plot_model(estimator = final_model1, plot = 'learning',save=True)
                     col10.image("Learning Curve.png")
                     with open("best_model.pkl",'rb') as f :
                          st.download_button("Telecharger le pipline du modele" , f, file_name="best_model.pkl")
          
          
            if methode=="Regression":
                if st.button("les performances du modèle "):
                    setup_data = setup_reg(data=df,target = target,
                        train_size =0.75,categorical_features =None,
                        normalize = False,normalize_method = None)
                    r=compare_models_reg()
                    save_model(r,"best_model")
                    st.success("youpiiiii le model de Regression est prete pour le deploiement ")
                    final_model1 = create_model_reg(r)
        
        else:
            st.image("https://ilm.univ-lyon1.fr//images/slides/SLIDER10.png")

        
    if __name__ == "__main__":
         main()
    st.markdown('<h1 style="text-align: center;">Prédiction avec le model deployé </h1>', unsafe_allow_html=True)
    st.markdown('Charger les données  ', unsafe_allow_html=True)

    def main():
        file_to_predict = st.file_uploader("Choisirun fichier à prédire", type=['csv'])
        if file_to_predict is not None:
            #rain(emoji="🎈",font_size=54,falling_speed=5,animation_length="infinite",)
            df_to_predict = load_data(file_to_predict)
            st.subheader("Résultats des prédictions")
            def predict_quality(model, df):
                  predictions_data = predict_model(estimator= model, data = df)
                  return predictions_data
            model = load_model('poluant_pipeline')
            pred=predict_quality(model, df_to_predict)
            st.dataframe(pred[pred.columns[-2:]])
        else:
            st.image("https://ilm.univ-lyon1.fr//images/slides/Germanium%20ILM.jpg")
    if __name__ == "__main__":
         main()

def image():
    st.markdown("# Idendification des polluants  ❄️")
    st.sidebar.markdown('<h1 style="text-align: center;">Identification des polluants à partir des scan spectrale 3D ❄️ </h1>', unsafe_allow_html=True)
    col3,col4,col5=st.sidebar.columns(3)
    col3.image("https://www.researchgate.net/profile/Daniel-Jirak/publication/339362052/figure/fig3/AS:860297812246529@1582122390225/3D-excitation-emission-maps-of-A-98BSA-AuNCs-and-B-df98BSA-AuNCs-Note-Strong.ppm", use_column_width=True)
    col4.image("https://ilm.univ-lyon1.fr/templates/mojito/images/logo.jpg", use_column_width=True)
    col5.image("https://formation-professionnelle.universite-lyon.fr/var/site/storage/images/3/3/5/0/533-17-fre-FR/Lyon-1-Claude-Bernard.png", use_column_width=True)
    st.sidebar.write("<p style='text-align: center;'>Alioune Gaye : Stagiaire ILM (%s)</p>" % url, unsafe_allow_html=True)
    st.sidebar.write("<p style='text-align: center;'> Encadrent:Martini Matéo </p>", unsafe_allow_html=True)
    st.sidebar.write("<p style='text-align: center;'> modèle pré-entrainé (prete pour la prediction d'image) </p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>Nous avons procéder comme suit :</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>1 -On a créer une base de données d'images 3D à partir de données spectrales  , nous avons divisé la base de données en ensembles d'entraînement  et de validation </p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>2 - créé un modèle de classification basé sur le modèle VGG16 en utilisant Keras</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>3 -  utilisé les 15 premiers couches du modèle </p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>4 - ajouté une couche de classification à la sortie du modèle avec une activation sigmoid pour la classification multi-classes.</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>5 - compilé le modèle en utilisant une fonction de perte de binary_crossentropy (car les étiquettes sont encodées en tant que vecteurs binaires) et un optimiseur SGD</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>6 -  créé des générateurs d'images d'apprentissage  et de validation pour alimenter le modèle pendant l'entraînement, en utilisant ImageDataGenerator de Keras pour augmenter les données d'entraînement (rotation, ré-échelle, retournement, etc.).</p>", unsafe_allow_html=True)
    from keras.models import load_model
    st.markdown('<h1 style="text-align: center;">Prédiction image 3D </h1>', unsafe_allow_html=True)
    model = load_model('model_final2.pkl')
    f=['A','A+D','D','E']
    from PIL import Image
    def preprocess_image(image_path, target_size=(224, 224)):
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = image.resize(target_size)
        return np.array(image) / 255.0

    def predict_class(model, image_path):
        img = preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        index = np.argmax(pred)
        f.sort()
        pred_value = f[index]
        return pred_value

    file = st.file_uploader("Entrer l'image", type=["jpg", "png"])
    if file is None:
        st.text("entrer l'image à prédire")
    else:
        label = predict_class(model, file)
        st.image(file, use_column_width=True)
        st.markdown("## Résultats de la prédiction ")
        st.markdown("## Il s'agit du polluant")
        st.write(label)
    st.image("https://ilm.univ-lyon1.fr//images/slides/SLIDER7.png")


def Quantification():
    st.markdown('<h1 style="text-align: center;"> Quantification des polluants: la méthode du double ajouts dosées 🎉 </h1>', unsafe_allow_html=True)
    st.sidebar.markdown('<h1 style="text-align: center;"> Quantification des polluants heterogéne 🎉 </h1>', unsafe_allow_html=True)
    st.sidebar.write("<p style='text-align: center;'> Encadrent:Martini Matéo </p>", unsafe_allow_html=True)

    def cal_conc2(x,y,z,h,Ca,Cd):
        a=h/Ca
        a1=z/Cd
        C_A=(y-a1*x)/(1-a1*a)
        C_D=(x-a*y)/(1-a1*a)
        conc=pd.DataFrame([C_A,C_D])
        conc.index=['C_A','C_D']
        return(conc) 
    
    
    def cal_conc(x,y,z,h,Ca,Cd):
        a=-z/Ca # serie
        a1=-h/Cd
        y1=-y
        y3=-x
        C_A=(a*y3-y1)/(a1*a-1)
        C_D=(a1*y1-y3)/(a1*a-1)
        conc=pd.DataFrame([C_A,C_D])
        conc.index=['C_A','C_D']
        return(conc)
    def find_delimiter(filename):
        sniffer = csv.Sniffer()
        with open(filename) as fp:
             delimiter = sniffer.sniff(fp.read(5000)).delimiter
        return delimiter
    def mono_exp(df,VAR):
         #-------------Nettoyage du dataframe----------------#
        for i in df.columns:
            if (df[i].isnull()[0]==True):# On elimine les colonnes vides
                del df[i];
        df=df.dropna(axis=0);#On elimine les lignes contenant des na
        df=df[1:];
        df=df.astype(float); # On convertit les valeurs contenues dans les colonnes en float (à la place de string)
        df=df[df[df.columns[0]]>=0.1]
        ncol=(len(df.columns)) # nombre de colonnes
        najout=(ncol/2)-3; # nombre d'ajouts en solution standard
        #---------------------First step----------------------#
        def f_decay(x,a,b,c):
            return(c+a*np.exp(-x/b));
        df1=pd.DataFrame(columns=['A'+VAR.split('/')[-1],'Tau'+VAR.split('/')[-1]]);
        row=int(len(df.columns)/5)
        row2=int(len(df.columns)/2)
        for  i in range(int(ncol/2)):
            x=df[df.columns[0]]; # temps
            y=df[df.columns[(2*i)+1]]; # Intensités de fluorescence
            popt,pcov=curve_fit(f_decay,x,y,bounds=(0, np.inf));
            df1=df1.append({'A'+VAR.split('/')[-1] :popt[0] , 'Tau'+VAR.split('/')[-1] :popt[1]} , ignore_index=True)
        return(df1) 

    def double_exp2(df,VAR):
        T1=0.092
        T2=1.317
        #df=pd.read_csv(VAR,sep=delimit)
        #-------------Nettoyage du dataframe----------------#
        for i in df.columns:
            if (df[i].isnull()[0]==True):# On elimine les colonnes vides
                del df[i];
        df=df.dropna(axis=0);#On elimine les lignes contenant des na
        df=df[1:];
        df=df.astype(float); # On convertit les valeurs contenues dans les colonnes en float (à la place de string)
        df=df[df[df.columns[0]]>=0.1]
        ncol=(len(df.columns)) # nombre de colonnes
        najout=(ncol/2)-3; # nombre d'ajouts en solution standard
        #---------------------First step----------------------#
        def f_decay(x,a1,a2,r):
            return(r+a1*np.exp(-x/T1)+a2*np.exp(-x/T2));
    
        df1=pd.DataFrame(columns=['A_'+VAR.split('/')[-1],'Aire_'+VAR.split('/')[-1]]);
        for i in range(int(ncol/2)):
            x=df[df.columns[0]]; # temps
            y=df[df.columns[(2*i)+1]]; # Intensités de fluorescence
            y=list(y)
            y0=max(y)#y[1]
            popt,pcov=curve_fit(f_decay,x,y,bounds=(0.1,[y0,y0,+np.inf]));
            #tau=(popt[0]*(popt[1])**2+popt[2]*(popt[3])**2)/(popt[0]*(popt[1])+popt[2]*(popt[3]))
            A1=popt[0]*T1
            A2=popt[1]*T2
            A=A1+A2 # l'aire sous la courbe de l'intensité de fluorescence 
            df1=df1.append({'A_'+VAR.split('/')[-1] :A1,'Aire_'+VAR.split('/')[-1] :A} , ignore_index=True);
        return(df1)  

    

    
    def tri_exp(df,VAR):
        for i in df.columns:
            if (df[i].isnull()[0]==True): # On elimine les colonnes vides
                del df[i];
        df=df.dropna(axis=0); # On elimine les lignes qui contiennent des na;
        df=df[1:];
        df=df.astype(float); # On convertit les valeurs contenues dans les colonnes en float (à la place de string)
        df=df[df[df.columns[0]]>=0.1]
        ncol=(len(df.columns)) # nombre de colonnes
        def f_decay(x,a1,b1,c,r): # Il s'agit de l'équation utilisée pour ajuster l'intensité de fluorescence en fonction du temps(c'est à dire la courbe de durée de vie)
            return(a1*np.exp(-x/b1)+(a1/2)*np.exp(-x/(b1+1.177*c))+(a1/2)*np.exp(-x/(b1-1.177*c))+r)
                                           
        df2=pd.DataFrame(columns=["préexpo_"+VAR.split('/')[-1],"tau_"+VAR.split('/')[-1]]); # Il s'agit du dataframe qui sera renvoyé par la fonction
        #### Ajustement des courbes de durée de vie de chaque solution en fonction du temps#### 
        print('polluant '+VAR.split('/')[-1].split('.')[0])
        row=int(len(df.columns)/5)
        row2=int(len(df.columns)/2)
        fig, axs = plt.subplots(nrows=3, ncols=row, figsize=(20, 20))
        for ax, i in zip(axs.flat, range(int(ncol/2))):
            x=df[df.columns[0]]; # temps
            y=df[df.columns[(2*i)+1]]; # Intensités de fluorescence
            y=list(y)
            yo=max(y)#y[1]
            bound_c=1
            while True:
                try:
                   popt,pcov=curve_fit(f_decay,x,y,bounds=(0,[yo,+np.inf,bound_c,+np.inf]),method='trf') # On utilise une regression non linéaire pour approximer les courbes de durée de vie  
                   #popt correspond aux paramètres a1,b1,c,r de la fonction f_decay de tels sorte que les valeurs de f_decay(x,*popt) soient proches de y (intensités de fluorescence)
                   break;
                except ValueError:
                    bound_c=bound_c-0.05
                    print("Oops")
            df2=df2.append({"préexpo_"+VAR.split('/')[-1]:2*popt[0],"tau_"+VAR.split('/')[-1]:popt[1]} , ignore_index=True);# Pour chaque solution , on ajoute la préexponentielle et la durée de vie tau à la dataframe
    
            ax.plot(x,y,label="Intensité réelle");
            ax.plot(x,f_decay(x,*popt),label="Intensité estimée");
            ax.set_title(" solution "+df.columns[2*i]);
            ax.set_xlabel('Temps(ms)');
            ax.set_ylabel('Intensité(p.d.u)');
            plt.legend();
        plt.show();
    
        return(df2)

    
    ## regression avec linearregression
    def regression1(result,std,unk,ss,d):
        concentration=pd.DataFrame(columns=['polyfit'])
        col1, col2 ,col3,col4= st.columns(4)
        col=[col1,col2,col3,col4]
        for t in range(len(ss)): 
            fig, ax = plt.subplots()
            tau=result[result.columns[2*t+1]]
            cc=tau;
            y=np.array(cc); 
            std=np.array(std)
            conc=ss[t]*std/unk
            x=conc;
            n=len(x)
            x=x[1:(n-1)]
            y=y[1:(n-1)]
            plt.scatter(x,y);
            mymodel = np.poly1d(np.polyfit(x, y, d)) # polynome de degré 1
            x=x.reshape(-1,1);
            y_intercept = mymodel(0)
            R3=r2_score(y, mymodel(x))
            # tracer les courbes de calibérations 
            #print('\n',f"\033[031m {result.columns[2*t+1][2:]} \033[0m",'\n')
            plt.plot(x, mymodel(x),'m',label='np.polyfit : R² = {}'.format(round(R3,2)))
            plt.xlabel('Concentration solution standard(ppm)')
            plt.ylabel('durée de vie(ms)');
            plt.title('Courbe de calibration'+'du polluant '+result.columns[2*t+1][4:])
            plt.legend();
            col[t].pyplot(fig)
            y_intercept = mymodel(0)
            col[t].write("y_intercept")
            col[t].write(y_intercept)
            # Calcul des racines (x_intercept)
            roots = np.roots(mymodel)
            x_intercepts = [root for root in roots if np.isreal(root)]
            x_inter=fsolve(mymodel,0)
            col[t].write("x_intercept")
            col[t].write(x_inter)
            slope=mymodel.coef[0]
            col[t].write("slope")
            col[t].write(slope)
            x_inter=fsolve(mymodel,0)
            Cx=(y_intercept-tau[0])/slope
            concentration=concentration.append({'polyfit':round(x_inter[0],2)},ignore_index=True)
        return(concentration)

   
    def fun(tau):
        sum_k=1/tau
        kch=-sum_k+sum_k[0]
        return(sum_k,kch)


    
    def regression2(result, std,unk, ss, sum_kchel):
        col1, col2 ,col3,col4=st.columns(4)

        col=[col1,col2,col3,col4]

        con_poly3 = []

        con2 = []

        for i in range(len(ss)):

            fig, ax = plt.subplots()

            tau = result[result.columns[2*i+1]]

            cc = tau

            y = np.array(cc)

            std = np.array(std)

            conc = ss[i]* std / unk

            x = conc

            n = len(x)

            x = x[1:(n-1)]

            kchel = sum_kchel[sum_kchel.columns[2*i+1]]

            sum_k = sum_kchel[sum_kchel.columns[2*i+1]]

            kchel = kchel[1:(n-1)]

            def func(x, a,b, c): # x-shifted log
                 return a*np.log(x+ b)/2+c

            initialParameters = np.array([1.0,1.0, 1.0])

            log_params, _= curve_fit(func,x, kchel, initialParameters,maxfev=50000)

            log_r2 = r2_score(kchel,func(x,*log_params))

            best_model = func(x,*log_params)

            plt.scatter(x, kchel)

            plt.plot(x, best_model,'m')

            col[i].pyplot(fig)

            col[i].write(log_r2)

            y_intercept = func(0,*log_params)

            col[i].write("y_intercept")

            col[i].write(y_intercept)

            x_inter = np.exp(-2*log_params[2]/log_params[0])- log_params[1]

            x_inter=np.array([x_inter])

            col[i].write("x_intercept")

            col[i].write(x_inter)

            slope = -log_params[1]*func(x_inter, *log_params)

            #[i].write("slope")

            #col[i].write(slope)

            con_poly3.append(x_inter)

            con2.append(x_inter)
        return con_poly3

    
    
    Taux4 = pd.DataFrame()

    
    
    def main():
        col3,col4=st.sidebar.columns(2)
        col3.image("https://ilm.univ-lyon1.fr/templates/mojito/images/logo.jpg", use_column_width=True)
        col4.image("https://formation-professionnelle.universite-lyon.fr/var/site/storage/images/3/3/5/0/533-17-fre-FR/Lyon-1-Claude-Bernard.png", use_column_width=True)
        st.sidebar.write("<p style='text-align: center;'>Alioune Gaye : Stagiaire ILM (%s)</p>" % url, unsafe_allow_html=True)
        st.sidebar.write("<p style='text-align: center;'> Encadrent:Martini Matéo </p>", unsafe_allow_html=True)
        st.sidebar.write("<p style='text-align: center;'>  Les methodes utilisées sont:  </p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>1 - Méthode mono_exponentielle : </p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>2 - Méthode double_exponentielle :</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>3 - Méthode gaussienne  : </p>", unsafe_allow_html=True) 
        st.sidebar.markdown("<p style='text-align: center;'>Dans chacune des  méthode nous avons procéder comme suit : On Calcul de la concentration inconnue pour chacune des séries, en passant par les courbes de calibrations (durée de vie et nombre d'ion chélaté ) en fonction de la concentration en solution standard par une regression linéaire et non linéaire succesivement afin d'utiliser le systéme d'équation (P) pour trouver la concentration de chaque polluant dans le mélange\n </p>", unsafe_allow_html=True)
        st.latex(r'''P = \begin{cases} -C_{HD} = C_D + K_{A-D}C_A &\text{S1 }  \\ - C_{HA}= C_A + K_{D-A}C_D &\text{S2 } \\- C_{DA} = K_{D-A}C_D^0  &\text{S3 }   \\ - C_{AD} = K_{A-D}C_A^0 &\text{S4 }  \end{cases}''')
        uploaded_files = st.file_uploader("Choisir les fichiers csv ", accept_multiple_files=True)
        st.image("https://ilm.univ-lyon1.fr//images/slides/CARROUSSEL-17.png")
        unk = st.number_input("Volume unk")
        ss1 = st.number_input("Solution standard",value=1, step=1, format="%d")
        rev = st.number_input("Volume rev")
        Ca = st.number_input("Concentration initiale de A",value=1, step=1, format="%d")
        Cd = st.number_input("Concentration initiale de D",value=1, step=1, format="%d")
        #std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1]
        #std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2] # 06-06
        #std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2,1] 
        #std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2,1.7]
        #std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1] # Volume standard 07-06 , 12-06
        #std=[0,0,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.5,1] # Volume standard 08-06 
        #std=[0,0,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1] # 09-06
        std=[0,0,0.025,0.075,0.125,0.2,0.5,0.7,1,1.5,4] # Volume standard 20-06 , 21-06
        # Calculer ss en fonction de ss1
        ss = [ss1] * 4
        #global Taux4
        if uploaded_files is not None:
            col5,col6,col7=st.columns(3)
            Taux4 = pd.DataFrame()
            if col5.button("Méthode mono_exponentielle"):
                st.latex(r''' \fcolorbox{red}{green}{$f_decay(x,a,tau) =  \epsilon + a\exp (\frac{-x}{tau} ) $}''')
                for uploaded_file in uploaded_files:
                    df = pd.read_csv(uploaded_file, delimiter="\t")
                    Q=mono_exp(df,uploaded_file.name)
                    T=pd.concat([Taux4,Q], axis=1)
                    Taux4=T
                Taux=Taux4
                sum_kchel1=pd.DataFrame() # gaussienne
                sum_kchel2=pd.DataFrame()# double exp
                sum_kchel3=pd.DataFrame() # mono exp
                for j in range(4):
                    tt3=Taux4[Taux4.columns[2*j+1]]
                    s_k=pd.DataFrame(fun(tt3))
                    s_k=s_k.T
                    s_k.columns=['sum_k'+Taux.columns[2*j+1].split('_')[-1],'kchel'+Taux.columns[2*j+1].split('_')[-1]]
                    sum_kchel3=pd.concat([sum_kchel3,s_k],axis=1)
                st.markdown('## <h1 style="text-align: center;"> Les valeurs de Taux et la pré_exponentielle</h1>',unsafe_allow_html=True) 
                st.write(Taux4.style.background_gradient(cmap="Greens")) 
                st.markdown('## <h1 style="text-align: center;">Calcul de la concentration en fonction de durée de vie</h1>',unsafe_allow_html=True) 
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression1(Taux4,std,unk,ss,1) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Greens"))
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression non linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression2(Taux4,std,unk,ss,Taux4)
                col1.write(concentration4)
                #polyfit=concentration4[concentration4.columns[0]]
                #r2=cal_conc(*polyfit,Ca,Cd)
                #col2.write(r2.style.background_gradient(cmap="Greens"))
                concentration4=regression6(Taux4,std,unk,ss,Taux4)
                #col1.write(concentration4)

                #polyfit=concentration4[concentration4.columns[0]]

                concen =pd.DataFrame(concentration4)

                serie=['s1','s2','s3','s4']

                concen.index=serie

                col1.dataframe(concen)

                r2=cal_conc(*concentration4,Ca,Cd)

                col2.write(r2.style.background_gradient(cmap="Greens"))


                st.markdown('## <h1 style="text-align: center;">Calcul de la concentration en fonction de nombre d\'ion chélaté </h1>',unsafe_allow_html=True) 
                st.write(sum_kchel3.style.background_gradient(cmap="Greens")) 
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression non linéaire </h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentrationC4=regression2(Taux4,std,unk,ss,sum_kchel3)
                concen =pd.DataFrame(concentrationC4)
                serie=['s1','s2','s3','s4']
                concen.index=serie
                col1.dataframe(concen)
                r1=cal_conc(*concentrationC4,Ca,Cd)
                col2.dataframe(r1.style.background_gradient(cmap="Greens"))
                    
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression1(sum_kchel3,std,unk,ss,1) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Greens"))

    
            if col6.button("Méthode double_exponentielle"):
                st.latex(r'''\fcolorbox{red}{blue}{$f_decay(x,a1,t1,a2,t2) =  \epsilon + a1\exp (\frac{-x}{t1} ) +a2\exp (\frac{-x}{t2})$}''')
                st.latex(r'''\fcolorbox{red}{blue}{$Aire = a1t1 + a2t2 $}''')
                Taux4 = pd.DataFrame()
                for uploaded_file in uploaded_files:
                       df = pd.read_csv(uploaded_file, delimiter="\t")
                       Q=double_exp(df,uploaded_file.name)
                       T=pd.concat([Taux4,Q], axis=1)
                       Taux4=T
                Taux=Taux4
                sum_kchel1=pd.DataFrame() # gaussienne
                sum_kchel2=pd.DataFrame()# double exp
                sum_kchel3=pd.DataFrame() # mono exp
                for j in range(4):
                    tt3=Taux4[Taux4.columns[2*j+1]]
                    s_k=pd.DataFrame(fun(tt3))
                    s_k=s_k.T
                    s_k.columns=['sum_k'+Taux.columns[2*j+1].split('_')[-1],'kchel'+Taux.columns[2*j+1].split('_')[-1]]
                    sum_kchel3=pd.concat([sum_kchel3,s_k],axis=1)
                st.markdown('## <h1 style="text-align: center;"> Les valeurs de Taux et la pré_exponentielle</h1>',unsafe_allow_html=True)
                st.write(Taux4.style.background_gradient(cmap="Blues")) 
                st.markdown('## <h1 style="text-align: center;">Calcul de la concentration en fonction de durée de vie</h1>',unsafe_allow_html=True) 
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression1(Taux4,std,unk,ss,1) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Blues"))
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression2(Taux4,std,unk,ss,Taux4) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Blues"))

                st.markdown('## <h1 style="text-align: center;">Calcul de la concentration en fonction de nombre d\'ion chélaté </h1>',unsafe_allow_html=True) 
                st.write(sum_kchel3.style.background_gradient(cmap="Blues")) 
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression non linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentrationC4=regression2(Taux4,std,unk,ss,sum_kchel3)
                concen =pd.DataFrame(concentrationC4)
                serie=['s1','s2','s3','s4']
                concen.index=serie
                col1.dataframe(concen)
                r1=cal_conc(*concentrationC4,Ca,Cd)
                col2.dataframe(r1.style.background_gradient(cmap="Blues"))
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression1(sum_kchel3,std,unk,ss,1) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Blues"))
            if col7.button("Méthode gaussienne "):
                st.latex(r'''\fcolorbox{red}{purple}{$f_decay (x,a1,t1,c) =  \epsilon + a1\exp (\frac{-x}{t1} )  +\frac{a2}{2}\exp (\frac{-x}{t1+1.177c} ) +\frac{a2}{2}\exp (\frac{-x}{t1-1.177c})$}''')
                Taux4 = pd.DataFrame()
                for uploaded_file in uploaded_files:
                       df = pd.read_csv(uploaded_file, delimiter="\t")
                       Q=tri_exp(df,uploaded_file.name)
                       T=pd.concat([Taux4,Q], axis=1)
                       Taux4=T
                Taux=Taux4
                sum_kchel1=pd.DataFrame() # gaussienne
                sum_kchel2=pd.DataFrame()# double exp
                sum_kchel3=pd.DataFrame() # mono exp
                for j in range(4):
                    tt3=Taux4[Taux4.columns[2*j+1]]
                    s_k=pd.DataFrame(fun(tt3))
                    s_k=s_k.T
                    s_k.columns=['sum_k'+Taux.columns[2*j+1].split('_')[-1],'kchel'+Taux.columns[2*j+1].split('_')[-1]]
                    sum_kchel3=pd.concat([sum_kchel3,s_k],axis=1)
                st.markdown('## <h1 style="text-align: center;"> Les valeurs de Taux et la pré_exponentielle</h1>',unsafe_allow_html=True)
                st.write(Taux4.style.background_gradient(cmap="Purples")) 
                st.markdown('## <h1 style="text-align: center;">Calcul de la concentration en fonction de durée de vie</h1>',unsafe_allow_html=True) 
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression1(Taux4,std,unk,ss,1) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Purples"))
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression2(Taux4,std,unk,ss,Taux4)
                col1.write(concentration4)
                #polyfit=concentration4[concentration4.columns[0]]
                #r2=cal_conc(*polyfit,Ca,Cd)
                #col2.write(r2.style.background_gradient(cmap="Purples"))

                

                concentration4=regression6(Taux4,std,unk,ss,Taux4)

                #col1.write(concentration4)

                #polyfit=concentration4[concentration4.columns[0]]

                concen =pd.DataFrame(concentration4)

                serie=['s1','s2','s3','s4']

                concen.index=serie

                col1.dataframe(concen)

                r2=cal_conc(*concentration4,Ca,Cd)

                col2.write(r2.style.background_gradient(cmap="Greens"))


                st.markdown('## <h1 style="text-align: center;">Calcul de la concentration en fonction de nombre d\'ion chélaté </h1>',unsafe_allow_html=True) 
                st.write(sum_kchel3.style.background_gradient(cmap="Purples")) 
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression non linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentrationC4=regression2(Taux4,std,unk,ss,sum_kchel3)
                concen =pd.DataFrame(concentrationC4)
                serie=['s1','s2','s3','s4']
                concen.index=serie
                col1.dataframe(concen)
                r1=cal_conc(*concentrationC4,Ca,Cd)
                col2.dataframe(r1.style.background_gradient(cmap="Purples"))
                st.markdown('## <h1 style="text-align: center;"> Resultats de la regression linéaire</h1>',unsafe_allow_html=True)
                col1,col2=st.columns(2)
                concentration4=regression1(sum_kchel3,std,unk,ss,1) 
                col1.write(concentration4)
                #polyfit=concentration4[concentration4.columns[0]]
                #r2=cal_conc(*polyfit,Ca,Cd)
                #col2.write(r2.style.background_gradient(cmap="Purples"))

                concentration4=regression6(Taux4,std,unk,ss,Taux4)

                #col1.write(concentration4)

                #polyfit=concentration4[concentration4.columns[0]]

                concen =pd.DataFrame(concentration4)

                serie=['s1','s2','s3','s4']

                concen.index=serie

                col1.dataframe(concen)

                r2=cal_conc(*concentration4,Ca,Cd)

                col2.write(r2.style.background_gradient(cmap="Greens"))


                
    if __name__ == "__main__":
         main()




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

page_names_to_funcs = {
    "identification": identification,
    "image": image,
    "Quantification": Quantification,
    "Code_identification":Code_identification
}

selected_page = st.sidebar.selectbox("Selectionner ", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
