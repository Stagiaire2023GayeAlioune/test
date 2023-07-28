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


def Code_classification_polluants_heterogene():
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


# ## Maintenant cherchons les variables les plus significatifs (importantes pour  classifier les polluants )  , avec des tests d'hypothéses  sous R .... Ensuite , passons au deployement de notre model avec mlfow puis l'utiliser pour la prediction....


''', language='python')
    if __name__ == "__main__":
         main()    


def Code_apprentissage_3D():
    def main():
	    st.code('''#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt;
from tkinter import filedialog
from tkinter import *
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import matplotlib as ml
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import tensorflow as tf
import keras.preprocessing.image
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.tree
import sklearn.ensemble
import os;
import datetime  
import cv2 
import seaborn as sns
from PIL import Image
from tensorflow.keras.models import Sequential
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import datetime
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
from PIL import Image
import os
import cv2
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard


# In[12]:


def browseFiles2():
	filename = filedialog.askopenfilenames(initialdir = "http://localhost:8888/tree/Stage",
										title = "Select a File",
										filetypes = (("Csv files",
														"*.csv*"),
													("all files",
														"*.*")))
	return(filename)


# ### Selectionner plussieurs fichiers  d'expectre 3D pour creer la base de donnée .

# In[ ]:


VAR=browseFiles2()
VAR


# ## fonction pour prendre en compte toute srte de delimiteur 

# In[ ]:


import csv
def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter


# In[ ]:


fichier=[]
col=['A+D','D','E','A']
labels=[]
for VAR in range(len(VARS)):
    image=[]
    row=int(len(VARS[VAR])/5)
    i=0
    for V in VARS[VAR]:
        df=pd.read_csv(V,sep=';')
        df=df.dropna()
        d1=df.drop(df.index[1])
        d1=d1[1:]
        x1=df[df.columns[:]]
        x1=x1[1:].astype(float)
        y=df[df.columns[0]]
        I=df[df.columns[1]]
        y=y[1:].astype(float)
        I=I[1:].astype(float)
        z=df[df.columns[2]]
        z=z[1:].astype(float)

        x1_values = x1.iloc[:, 1:]
        Y, X = np.meshgrid(y, range(x1_values.shape[1]))
        fig1, ax = plt.subplots()
        cmap = ax.contourf(Y, X, x1_values.T)
        fig1.colorbar(cmap)
        ax.set_title(V.split('/')[-1])
        ax.set_xlabel('longueur d \'onde \n '+V.split('/')[-1])
        ax.set_ylabel('Z_axis')
        fig1.savefig('Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/'+V.split('/')[-1]+'.png')
        f=r'Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/'+V.split('/')[-1]+'.png'
        fichier.append(f)
        labels.append(col[VAR])
    plt.show()


# ## Calcul des contribution de chaque pollant

# In[ ]:


ratio=[]
for V in VAR:
    r=[]
    r.append(V.split('-')[0].split('/')[8])
    r.append(V.split('-')[1])
    ratio.append(r)


# In[ ]:


ratio=pd.DataFrame(ratio)
ratio=ratio.astype(float)
ratio=ratio*0.01


# In[ ]:


ratio


# In[ ]:


VV=list(fichier)
VV=pd.DataFrame(VV)


# In[ ]:


data=pd.concat([VV,ratio],axis=1)


# In[ ]:


fichier[0].split('/')[-1]


# In[ ]:


from PIL import Image
labels1=[]
fichier1=[]
for j in range(len(fichier)):
    im = Image.open(fichier[j])
    im = im.rotate(180)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+labels[j]+"rotation"+fichier[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D//base/"+labels[j]+"rotation"+fichier[j].split('/')[-1]
    fichier1.append(f)
    labels1.append(labels[j])
    plt.imshow(im)
    plt.show()


# In[ ]:


from PIL import Image
labels2=[]
fichier2=[]
for j in range(len(fichier)):
    im = Image.open(fichier[j])
    im = im.convert("L")  # Grayscale
    plt.imshow(im)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+labels[j]+"couleur"+fichier[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+labels[j]+"couleur"+fichier[j].split('/')[-1]
    fichier2.append(f)
    labels2.append(labels[j])


# #### Creation d'une base de donnée  et la statistique descriptive des données 

# In[ ]:


base=np.transpose(pd.DataFrame([fichier,labels]))
base1=np.transpose(pd.DataFrame([fichier1,labels1]))
base2=np.transpose(pd.DataFrame([fichier2,labels2]))
Base=pd.concat([base,base1,base2],axis=0)
Base.columns=['image','labels']
Base['labels'].value_counts().plot.bar()
base=Base
base


# ## On a remarquer qu'on a pas assez se données ainsi , on essaye d'augmenter les données en fessant des rotations d'image de changer les contratstes ....

# In[ ]:


### from PIL import Image
from PIL import Image, ImageOps
from PIL import Image, ImageFilter
labels3=[]
fichier3=[]
t=base[base['labels']=='D']

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(30)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_30"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_30"+t['image'].values[j].split('/')[-1]
    fichier3.append(f)
    labels3.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(60)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_60"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_60"+t['image'].values[j].split('/')[-1]
    fichier3.append(f)
    labels3.append(t['labels'].values[j])
    plt.show()
for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(130)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_130"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_130"+t['image'].values[j].split('/')[-1]
    fichier3.append(f)
    labels3.append(t['labels'].values[j])
    plt.show()


for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im.putalpha(128)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/base/"+t['labels'].values[j]+"couleur_D_60_col"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_60_col"+t['image'].values[j].split('/')[-1]
    fichier3.append(f)
    labels3.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im= im.convert("L")
    im= im.filter(ImageFilter.FIND_EDGES)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_bruit"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_bruit"+t['image'].values[j].split('/')[-1]
    fichier3.append(f)
    labels3.append(t['labels'].values[j])
    plt.show()
    
t=base[base['labels']=='A']
fichier4=[]
labels4=[]
for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(30)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_30"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_30"+t['image'].values[j].split('/')[-1]
    fichier4.append(f)
    labels4.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(60)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_60"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_60"+t['image'].values[j].split('/')[-1]
    fichier4.append(f)
    labels4.append(t['labels'].values[j])
    plt.show()
for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(130)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_130"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_130"+t['image'].values[j].split('/')[-1]
    fichier4.append(f)
    labels4.append(t['labels'].values[j])
    plt.show()


for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im.putalpha(128)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_60_col"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_60_col"+t['image'].values[j].split('/')[-1]
    fichier4.append(f)
    labels4.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im= im.convert("L")
    im= im.filter(ImageFilter.FIND_EDGES)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_bruit"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_bruit"+t['image'].values[j].split('/')[-1]
    fichier4.append(f)
    labels4.append(t['labels'].values[j])
    plt.show()
    
    
t=base[base['labels']=='E']
fichier5=[]
labels5=[]
for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(30)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_30"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_30"+t['image'].values[j].split('/')[-1]
    fichier5.append(f)
    labels5.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(60)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_60"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_60"+t['image'].values[j].split('/')[-1]
    fichier5.append(f)
    labels5.append(t['labels'].values[j])
    plt.show()
    
for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(130)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_130"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_130"+t['image'].values[j].split('/')[-1]
    fichier5.append(f)
    labels5.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im.putalpha(128)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_60_col"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_60_col"+t['image'].values[j].split('/')[-1]
    fichier5.append(f)
    labels5.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im= im.convert("L")
    im= im.filter(ImageFilter.FIND_EDGES)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_bruit"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_bruit"+t['image'].values[j].split('/')[-1]
    fichier5.append(f)
    labels5.append(t['labels'].values[j])
    plt.show()


# In[ ]:


base=np.transpose(pd.DataFrame([fichier,labels]))
base1=np.transpose(pd.DataFrame([fichier1,labels1]))
base2=np.transpose(pd.DataFrame([fichier2,labels2]))
base3=np.transpose(pd.DataFrame([fichier3,labels3]))
base4=np.transpose(pd.DataFrame([fichier4,labels4]))
base5=np.transpose(pd.DataFrame([fichier5,labels5]))
Base=pd.concat([base,base1,base2,base3,base4,base5],axis=0)
Base.columns=['image','labels']
Base['labels'].value_counts().plot.bar()
base=Base
base.index=range(len(base))


# In[ ]:


base=base[base.duplicated()!=True]


# In[ ]:


base.index=range(len(base))


# ### separation des données en base de train et de validation 

# In[ ]:


train_df, test_df= train_test_split(base, test_size=0.05)
train_df, validate_df = train_test_split(train_df, test_size=0.3)
train_df = train_df.reset_index()
validate_df = validate_df.reset_index()
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
print(f' ==> base train contient  : {train_df.shape[0]} images \n  ==> base de validation contient :  {validate_df.shape[0]} images  ')
print(' ==> base train contient  :' ,train_df['labels'].unique(),'  ==> base de validation contient : ' ,validate_df['labels'].unique() ) 


# In[ ]:


for i in test_df['image'].values:
    im = Image.open(i)
    plt.savefig('Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/test/')


# In[ ]:


im


# In[ ]:





# In[ ]:





# ### Algorithme reseau des neurones en utilisant  VGG16 de keras   , les performance du model , et l'enregistrement du model 

# In[10]:


from sklearn.model_selection import train_test_split
#import streamlit as st
from PIL import Image, ImageOps
import ydata_profiling   
import numpy as np
##from streamlit_pandas_profiling import st_profile_report
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
import cv2
import seaborn as sns
from PIL import Image
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import os
from tensorflow.keras.preprocessing import image
dir_path='Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/base1'
fichier=[]
labels=[]
for i in os.listdir(dir_path):
    fichier.append(dir_path+'/'+i)
    labels.append(i)
l=[]
for j in range(len(labels)):
    if (labels[j].find('A+D')!=-1) | (labels[j].find('D+A')!=-1) | (labels[j].find('DA')!=-1 ):
        l.append('A+D')
    elif labels[j].find('EDTA')!=-1:
        l.append('E')
    elif labels[j].find('AMPA')!=-1:
        l.append('A')
    else :
        l.append('D')
labels=l
base=np.transpose(pd.DataFrame([fichier,labels]))

Base=base
base.columns=['image','labels']
train_df, validate_df = train_test_split(base, test_size=0.3)
train_df = train_df.reset_index()
validate_df = validate_df.reset_index()
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
image_size = 224
input_shape = (image_size, image_size, 3)
epochs = 30
batch_size = 10
pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")

for layer in pre_trained_model.layers[:15]:
    layer.trainable = False
for layer in pre_trained_model.layers[15:]:
    layer.trainable = True
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
# Flatten la couche de sortie 1 dimension
x = GlobalMaxPooling2D()(last_output)
# # Ajoutez une couche entièrement connectée avec 512 unités cachées et activation ReLU
x = Dense(512, activation='relu')(x)
# ajouter un taux d'abandon 0.5
x = Dropout(0.5)(x)
# il faut donner le nombre de classe et ajouter l'activation sigmoid
x = layers.Dense(4, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image',
    y_col='labels',
    target_size=(image_size, image_size),
    batch_size=batch_size
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    x_col='image',
    y_col='labels',
    target_size=(image_size, image_size),
    batch_size=batch_size
)



history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size)
model.save('Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Alioune/Identification_2023/spectres_excitation/donnees/3D/image_3D/model_final3.pkl')
##### Les performances de notre model 
plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0, 1])


#### accurancy et loss model 
plot_accuracy(history_11class,'accuracy')
plot_loss(history_11class,'loss')


# In[ ]:




  ''',language='python')

    if __name__ == "__main__":
         main()    

def Code_lissage_deconvolution_spectrale():
    def main():
	    st.code('''
             #!/usr/bin/env python
# coding: utf-8

# # Bibliothéque 

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit;
import math
import scipy.integrate as spi


# 
# # **<center><font color='blue'>I) Spectre d'excitation </font></center>**

# # **<center><font color='blue'>I.1) Traitement et Préparation des données </font></center>**
# 

# In[2]:


def browseFiles():
	filename = filedialog.askopenfilename(initialdir = "http://localhost:8888/tree/Stage",
										title = "Select a File",
										filetypes = (("Csv files",
														"*.csv*"),
													("all files",
														"*.*")))
	return(filename)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Importation des données

# In[3]:


VAR=browseFiles()


# In[4]:


VAR


# In[5]:


df=pd.read_csv(VAR,delimiter=";")
df


# 
# ## Pour voir à quoi resemble les données en entier 

# In[6]:


pd.set_option('display.max_rows',df.shape[0]+1)
df


# ## Remarque 
On constate des valeurs 'NA' dans le les données , des lignes vides , des colonnes vides et des valeurs anormales  .En plus , les colonnes nous indiques les intensités en fonction des longeurs d'ondes pour chaque echantillon , allant de l'echantillon ion à l'échantillon en excé .
# ## Affichage de quelques lignes 

# In[7]:


pd.set_option('display.max_rows',8)  ## 4 premieres ligne et 4 dernieres lignes
df


# ## Voir les valeurs manquantes 

# In[8]:


df.info()


# ## Remarque 
Comme on peut le constater on a observer plusieurs valeurs manquantes dans le jeux de données
# ## On suprime les valeurs manquantes  et les colonnes vides 

# In[9]:


for i in df.columns:
        if (df[i].isnull()[0]==True): # On elimine les colonnes vides
            del df[i]
df=df.dropna(axis=0)  # On elimine les lignes qui contiennent des na;
df=df[1:] ## la premiere ligne du dataframe ne nous interesse pas 
df=df.astype(float) ## on transforme les valeurs en float
df


# In[10]:


df.info()


# ## Statistique descriptive

# In[11]:


df.describe()


# ## Remarque1
On constate que il y'a plus de valeurs manquantes. Aprés netoyage , on a 151 individus (lignes) au total dans notre jeux de données avec 24 collones ( longueurs d'ondes et intensités pour chaque echantillon et succesivement).
Ensuite , l'excitation à eu lieu à partir de 250nm et se termine à 400nm pour tout les echantillons . En outre, le jeux de données ne contient pas de variables cathegorielles ou qualitatives .
# 
# 
# 
# # **<center><font color='blue'>I.2) Visualisation et Deconvolution </font></center>**

# In[12]:


df


# ##  Visualisation des  spectres d'excitations pour chaque echantillons 

# In[13]:


row=int(len(df.columns)/4)  ## nombre d'echantillons / 4 = 6
row2=int(len(df.columns)/2) ##  nombre d'echantillons 


fig, axs = plt.subplots(nrows=2, ncols=row, figsize=(20,6)) ## pour  afficher les courbes sur deux lignes de 6 courbes au max chacune
for ax, i in zip(axs.flat, range(row2)):
    x = df[df.columns[0]]   ### colonne longueurs d'ondes
    y = df[df.columns[2*i+1]]  ### les intensités pour chaque echantillons (colonnes impaires)
    ax.plot(x,y, label=df.columns[2*i])  
    ax.set_xlabel("Longeur d'onde")
    ax.set_ylabel("Intensité")
    ax.legend()
plt.show()


# ## Remarque1
On constate que les spectres d'excitation sont trés bruité , ainsi on aura en mesure de determiner le nombre total de pic
qui exciste dans chaque spectre . Donc , pour remedier à cela , on doit lisser les spectres . Car pour faire la decovulution , on aura besoin des nombres de pics , pour savoir combien de gausienne on aura besoin pour fiter les spectres .
# ## Lissage des spectres  , recherche de pics , Longeurs d'onde correspondants à chaque pic et les bornes des pics

# 

# Lissage avec algorithme de Savitz-golay
# La fonction savgol_filter prend en paramètre:
# 
# -y ou x : il s'agit de la donnée à filtrer.
# 
# - La longueur de la fenetre de lissage.
# 
# - Le degré du polynome de lissage. 
# Elle renvoie: La donnée filtrée.

# In[14]:


row=int(len(df.columns)/6)
row2=int(len(df.columns)/2)
p=[]
les_peaks=[]
borne=[]
fig, axs = plt.subplots(nrows=3, ncols=row, figsize=(20, 10))
for ax, i in zip(axs.flat, range(row2)):
    bor=[]
    x1=df[df.columns[0]]
    y=df[df.columns[2*i+1]]
    y_hat=savgol_filter(y, 11, 2)
    ax.plot(x1,y, label=df.columns[2*i])
    ax.plot(x1,y_hat,label='Savitzki-gol')
    x = y_hat
    peaks, properties = find_peaks(x, prominence=1, width=1)
    p.append(len(peaks))
    xmin=properties["left_ips"]
    xmax=properties["right_ips"]
    print("Longeur d'onde de chaque pic pour l'echantillon  ",df.columns[2*i],':',list(x1[peaks]),"nombre de peaks : ",len(peaks))
    for j in range(len(properties['left_ips'])):
        bor.append(list([x1[np.around(properties['left_ips'][j])],x1[np.around(properties['right_ips'][j])]]))
    #print('les bornes des pics pour ',df.columns[2*j],'est',borne)
    ax.plot(x1[peaks], x[peaks], "x")
    ax.vlines(x=x1[peaks], ymin=0,ymax = x[peaks], color = "C2")
    #ax.hlines(y=properties["widths"], xmin=properties['left_ips'],xmax=properties['right_ips'], color = "C1")
    ax.set_xlabel("Longeur d'onde")
    ax.set_ylabel("Intensité")
    ax.legend()
    les_peaks.append(list(x1[peaks]))
    les_peak=pd.DataFrame(les_peaks)
    if (len(bor)==5):
        borne.append(bor)           ### l'intervalles ou se trouve chaque pics ,pour tout les spectres
plt.show()


# ## Remarque

# In[15]:


p

On constate que le  nombre de pics varie d'un echantillon à un autre de 4 à 6 pics, mais le plus frequent est 5 pics .
# In[16]:


borne


# ## Déconvolution 

# In[ ]:





# In[17]:


## Gausienne 
def expS(x,I,m,b):
    return(I*np.exp(-((x-m)**2)/(2*(b**2)))/(b*np.sqrt(2*np.pi)))


# In[18]:


#### 5 Gausiennes 
def expT(x,I1,m1,b1,I2,m2,b2,I3,m3,b3,I4,m4,b4,I5,m5,b5):
        return(I1*np.exp(-((x-m1)**2)/(2*(b1**2)))/(b1*np.sqrt(2*np.pi))+
               I2*np.exp(-((x-m2)**2)/(2*(b2**2)))/(b2*np.sqrt(2*np.pi))+
               I3*np.exp(-((x-m3)**2)/(2*(b3**2)))/(b3*np.sqrt(2*np.pi))+
               I4*np.exp(-((x-m4)**2)/(2*(b4**2)))/(b4*np.sqrt(2*np.pi))+
               I5*np.exp(-((x-m5)**2)/(2*(b5**2)))/(b5*np.sqrt(2*np.pi))
              )
    


# In[19]:


def deconvol1(df1,bounds,nombre_peak):
    #row=int(len(df.columns)/4)
    row=3
    fig, axs = plt.subplots(nrows=4, ncols=row, figsize=(25, 15))
    for ax, i in zip(axs.flat, range(row2)):
            x=df1[df1.columns[0]]
            y=df1[df1.columns[2*i+1]]; # 
            y = y  / np.max(y) # pour normaliser les intensités
            y_hat=savgol_filter(y, 11, 2) ### spectre lissé
            pop1,pcov1=curve_fit(expT,x,y_hat,bounds=bounds) ### on fite avec 5 gaussienne
            ax.plot(x,y,label=df1.columns[2*i]) ### courbe bruité (initiale)
            ax.plot(x,y_hat,label='Savitzki-gol') ### spectre lissé
            ax.plot(x,expT(x,*pop1),label='somme') #### courbe fitée 
            for j in range(nombre_peak):
                ax.plot(x, expS(x, *pop1[3*j:3*(j+1)]), label=f'{j+1}ème déconvoluée') ### les deconvolutions 
            ax.legend()
    plt.show()
    return()
            


# In[20]:


borne[-1]


# In[21]:


nbr_p=5 ### nombre de pics 
def bornes():
    if(borne==[]):
        bounds=([0,250,0,0,270,0,0,300,0,0,340,0],[np.inf,270,np.inf,np.inf,300,np.inf,np.inf,340,np.inf,np.inf,360,np.inf])
    else:    
        bounds_lower =[0,borne[-1][0][0],0,0,borne[-1][0][1],0,0,borne[-1][2][0],0,0,borne[-1][3][0],0,0,borne[-1][4][0],0]
        bounds_upper =[np.inf,borne[-1][0][1],np.inf,np.inf,borne[-1][1][1],np.inf,np.inf,borne[-1][2][1],np.inf,np.inf,borne[-1][3][1],np.inf,np.inf,borne[-1][4][1],np.inf]
        bounds = (bounds_lower, bounds_upper)
    return(bounds)    
## Deconvolution des spectres
deconvol1(df,bornes(),nbr_p)       


# ## Remarque
On constate que avec 5 gausiennes , on obtient un spectre lisé qui fite trés bien avec le spectre initiale .Par contre , avec quatre gausienne il reste toujours des parties qui ne cole pas bien avec le spectre initial.
# 
# 
# 
# # **<center><font color='blue'>I.3) Calcule des parametres de chaque gaussienne et Construction de la base de donnée</font></center>**

# In[46]:


def browseFiles2():
	filename = filedialog.askopenfilenames(initialdir = "Z:\1_Data\1_Experiments\1_FENNEC\2_Stagiaires\2022_Alvin\7 Samples\ATMP_DTPMP",
										title = "Select a File",
										filetypes = (("Csv files",
														"*.csv*"),
													("all files",
														"*.*")))
	return(filename)


# ## Importation de plusieurs fichiers pour gagner du temps 

# In[47]:


VARS=browseFiles2()


# In[48]:


VARS


# ## Fonction pour detrminer le nombre de pics qui existe dans chaque fichiers 

# In[49]:


def pics(VARS,nombre_pics):
     for var in VARS:
        df=pd.read_csv(var,delimiter=",")
        for i in df.columns:
            if (df[i].isnull()[0]==True):  #On elimine les colonnes vides
                del df[i]
        df=df.dropna(axis=0); # On elimine les lignes qui contiennent des na;
        df=df[1:]; # voir colonne ci-dessous pour les détails de cette ligne.
        df=df.astype(float)
            
            
            
     p=[]
     les_peaks=[]
     row2=int(len(df.columns)/2)
     for i in range(row2):
        x1=df[df.columns[0]]
        y=df[df.columns[2*i+1]];
        y_hat=savgol_filter(y, 11, 2);
        x = y_hat
        peaks, properties = find_peaks(x, prominence=1, width=1)
        p.append(len(peaks))
        les_peaks.append(list(x1[peaks]))
        les_peak=pd.DataFrame(les_peaks)
     nombre_peak=set(p)
     nombre_peak=list(nombre_peak)
     les_peaks=np.transpose(les_peak)
     m_min = [np.nanmin(les_peak[pk]) for pk in range(len(les_peak.columns))]
     m_max =[np.nanmax(les_peak[pk]) for pk in range(len(les_peak.columns))]
     return(nombre_peak,m_min,m_max)       


# ## Fonction pour estimer les variables pour chaque gausienne pour tout les fichiers en meme temps et elle retourne la base de donnée qui sera utilisé dans  la partie machine learning

# In[27]:


def calcul_para(VARS,bornes,nombre_peak):
    df_dp=pd.DataFrame(columns = ['Fichier','Type','A1','M1','E1','C1','A2','M2','E2','C2','A3','M3','E3','C3','A4','M4','E4','C4','A5','M5','E5','C5'])
    for var in VARS:
        df1=pd.read_csv(var,delimiter=",")
        try:
            for i in df1.columns:
                if (df1[i].isnull()[0]==True): # On elimine les colonnes vides
                    del df1[i];
            df1=df1.dropna(axis=0); # On elimine les lignes qui contiennent des na;
            df1=df1[1:]; # voir colonne ci-dessous pour les détails de cette ligne.
            df1=df1.astype(float)
    
    
    
            for k in range(int(len(df1.columns)/2)):
                x=df1[df1.columns[0]]
                y=df1[df1.columns[2*k+1]]  # 
                y = (y - np.min(y)) / (np.max(y) - np.min(y)) # pour normaliser les intensités
                y_hat=savgol_filter(y, 11, 2)
                pop1,pcov1=curve_fit(expT,x,y,bounds=bounds)
                c1=spi.simps(expS(x,*pop1[0:3]),x)/spi.simps(expT(x,*pop1),x)
                c2=spi.simps(expS(x,*pop1[3:6]),x)/spi.simps(expT(x,*pop1),x)
                c3=spi.simps(expS(x,*pop1[6:9]),x)/spi.simps(expT(x,*pop1),x)
                c4=spi.simps(expS(x,*pop1[9:12]),x)/spi.simps(expT(x,*pop1),x)
                c5=spi.simps(expS(x,*pop1[12:]),x)/spi.simps(expT(x,*pop1),x)
                df_dp=df_dp.append({'Fichier':var.split('/')[-1],'Type':df1.columns[2*k], 
                                  'A1':pop1[0],'M1':pop1[1],'E1':pop1[2],'C1':c1,'A2':pop1[3],
                                  'M2':pop1[4],'E2':pop1[5],'C2':c2,'A3':pop1[6],'M3':pop1[7],
                                  'E3':pop1[8],'C3':c3,'A4':pop1[9],'M4':pop1[10],'E4':pop1[11],
                                  'C4':c4, 'A5':pop1[12],'M5':pop1[13],'E5':pop1[14],
                                  'C5':c5},ignore_index=True)
        except:
            print('erreur')       
                              
    return(df_dp)
    


# In[28]:


bounds_lower =[0,borne[-1][0][0],0,0,borne[-1][0][1],0,0,borne[-1][2][0],0,0,borne[-1][3][0],0,0,borne[-1][4][0],0]
bounds_upper =[np.inf,borne[-1][0][1],np.inf,np.inf,borne[-1][1][1],np.inf,np.inf,borne[-1][2][1],np.inf,np.inf,borne[-1][3][1],np.inf,np.inf,borne[-1][4][1],np.inf]
bounds = (bounds_lower, bounds_upper)
calcul_para(VARS,bornes(),5) ''', language='python')
    if __name__ == "__main__":
	    main()    



def Quatification_polluant_heterogene():
    def main():
	    st.code('''
#!/usr/bin/env python
# coding: utf-8 
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
from tkinter import *
from tkinter import filedialog
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


# 
# ###  **<center><font>  Fonction qui détermine le délimiter du fichier   </font></center>**

# In[2]:


def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter


# 
# 
# ###  **<center><font>  Fonction qui calcul les concentrations finale    </font></center>**



           
# In[ ]:





# In[ ]:





# def cal_conc(x,y,z,h,Ca,Cd):
#     ##    x,y,z,h    sont les concentrations obtenue avec les courbes de calibrations (x_intercept)
#     a=-z/Ca   ## serie 3 , Cd est la concentratin de D dans la serie3 
#     a1=-h/Cd  ## serie 4 , Ca est la concentration de A dans la serie4
#     y1=y   
#     y3=x
#     C_A=(-a*y3+y1)/(a1*a-1)
#     C_D=(-a1*y1+y3)/(a1*a-1)
#     conc=pd.DataFrame([C_A,C_D])
#     conc.index=['C_A','C_D']
#     return(conc) 

# In[4]:


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


# 
# 
# # **<center><font color='blue'> méthode  monoexponentielle </font></center>**

# In[5]:


def mono_exp(VAR):
    delimit=find_delimiter(VAR)
    df=pd.read_csv(VAR,sep=delimit)
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
    
    
    
    #--------------First step-----------------#
    
    def f_decay(x,a,b,c):
        return(c+a*np.exp(-x/b))
    df1=pd.DataFrame(columns=['A'+VAR.split('/')[-1],'Tau'+VAR.split('/')[-1]])
    row=int(len(df.columns)/5)
    row2=int(len(df.columns)/2)
    fig, axs = plt.subplots(nrows=3, ncols=row, figsize=(20, 20))
    for ax, i in zip(axs.flat, range(int(ncol/2))):
        x=df[df.columns[0]] # temps
        y=df[df.columns[(2*i)+1]]   # Intensités de fluorescence
        plt.scatter(x,np.log(y),label="curve "+df.columns[2*i])
        ax.set_xlabel('Temps(ms)');
        ax.set_ylabel('log(Intensité(p.d.u))')    
        plt.legend()
        popt,pcov=curve_fit(f_decay,x,y,bounds=(0, np.inf))  
        df1=df1.append({'A'+VAR.split('/')[-1] :popt[0] , 'Tau'+VAR.split('/')[-1] :popt[1]} , ignore_index=True);
        ax.plot(x,np.log(y),label="Intensité réelle");
        ax.plot(x,np.log(f_decay(x,*popt)),label="Intensité estimée");
        ax.set_title(" solution "+df.columns[2*i]);
        ax.set_xlabel('Temps(ms)');
        ax.set_ylabel('log(Intensité(p.d.u))');
        plt.legend();
    plt.show();
    return(df1)   
    


# # **<center><font color='blue'> méthode  double exponentielle </font></center>**

# In[6]:


def double_exp(VAR):
    delimit=find_delimiter(VAR)
    df=pd.read_csv(VAR,sep=delimit)
    #-------------Nettoyage du dataframe----------------#
    for i in df.columns:
        if (df[i].isnull()[0]==True):# On elimine les colonnes vides
            del df[i]
    df=df.dropna(axis=0);#On elimine les lignes contenant des na
    df=df[1:];
    df=df.astype(float); # On convertit les valeurs contenues dans les colonnes en float (à la place de string)
    df=df[df[df.columns[0]]>=0.1]
    ncol=(len(df.columns)) # nombre de colonnes
    najout=(ncol/2)-3; # nombre d'ajouts en solution standard
    #---------------------First step----------------------#
    def f_decay(x,a1,T1,a2,T2,r):
        return(r+a1*np.exp(-x/T1)+a2*np.exp(-x/T2));
    
    df1=pd.DataFrame(columns=['A'+VAR.split('/')[-1],'Tau'+VAR.split('/')[-1]]);
    row=int(len(df.columns)/5)
    row2=int(len(df.columns)/2)
    fig, axs = plt.subplots(nrows=3, ncols=row, figsize=(20, 20))
    for ax, i in zip(axs.flat, range(int(ncol/2))):
        x=df[df.columns[0]]; # temps
        y=df[df.columns[(2*i)+1]] # Intensités de fluorescence
        plt.scatter(x,np.log(y),label="curve "+df.columns[2*i])
        ax.set_xlabel('Temps(ms)');
        ax.set_ylabel('log(Intensité(p.d.u))')    
        plt.legend()
        y=list(y)
        y0=max(y)   #                                                                                                                                                                                                                                                                            y[1]
        popt,pcov=curve_fit(f_decay,x,y,bounds=(0,[y0,y0,+np.inf,+np.inf,+np.inf]));
        tau=(popt[0]*(popt[1])**2+popt[2]*(popt[3])**2)/(popt[0]*(popt[1])+popt[2]*(popt[3]))
        A=(popt[0]+popt[2])/2
        df1=df1.append({'A'+VAR.split('/')[-1] :A , 'Tau'+VAR.split('/')[-1] :tau} , ignore_index=True);
        ax.plot(x,np.log(y),label="Intensité réelle");
        ax.plot(x,np.log(f_decay(x,*popt)),label="Intensité estimée");
        ax.set_title(" solution "+df.columns[2*i]);
        ax.set_xlabel('Temps(ms)');
        ax.set_ylabel('log(Intensité(p.d.u))');
        plt.legend();
    plt.show()
    return(df1)   


# ## Calcul de la concentration à partir de l'aire sous la courbe de l'intensité 

# # f_decay $(x,a1,t1,a2,t2)$ = $ \epsilon + a1\exp (\frac{-x}{t1} ) +a2\exp (\frac{-x}{t2})  $
# ## $ Aire=\int I(t) \, dt  = a1t1 + a2t2 $
Ici , on fixe Tau1 et Tau2 puis estimé les valeurs des préexponentielles pour determiner l'aire sur la courbe de l'intensité  
# In[7]:


def double_exp2(VAR,T1,T2): 
    delimit=find_delimiter(VAR)
    df=pd.read_csv(VAR,sep=delimit)
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
        return(r+a1*np.exp(-x/T1)+a2*np.exp(-x/T2))

    
    df1=pd.DataFrame(columns=['Aire_'+VAR.split('/')[-1]]);
    for i in range(int(ncol/2)):
        x=df[df.columns[0]]; # temps
        y=df[df.columns[(2*i)+1]]; # Intensités de fluorescence
        y=list(y)
        y0=max(y)   #y[1]
        popt,pcov=curve_fit(f_decay,x,y,bounds=(0.1,[y0,y0,+np.inf]))
        #tau=(popt[0]*(popt[1])**2+popt[2]*(popt[3])**2)/(popt[0]*(popt[1])+popt[2]*(popt[3]))
        A1=popt[0]*T1  ## Laire sur la courbe du premier monoexponentielle 
        A2=popt[1]*T2   ## l'aire sur la courbe du deuxiéme monoexponentielle 
        A=A1+A2  # l'aire sous la courbe de l'intensité de fluorescence 
        df1=df1.append({'Aire_'+VAR.split('/')[-1] :A} , ignore_index=True) ### on retourne l'aire sous la courbe de l'intensité pour chacune des series 
    return(df1)   


# 
# 
# # **<center><font color='blue'>   méthode  gaussiennes  </font></center>**

def tri_exp(VAR):
    delimit=find_delimiter(VAR)
    df=pd.read_csv(VAR,sep=delimit);
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
        plt.scatter(x,np.log(y),label="curve "+df.columns[2*i])
        ax.set_xlabel('Temps(ms)');
        ax.set_ylabel('log(Intensité(p.d.u))')    
        plt.legend()
        
        y=list(y)
        yo=max(y)#y[1]
        bound_c=1
    
        while True:
            try:
                popt,pcov=curve_fit(f_decay,x,y,bounds=(0.1,[yo,+np.inf,bound_c,+np.inf]),method='dogbox') # On utilise une regression non linéaire pour approximer les courbes de durée de vie  
                #popt correspond aux paramètres a1,b1,c,r de la fonction f_decay de tels sorte que les valeurs de f_decay(x,*popt) soient proches de y (intensités de fluorescence)
                break;
            except ValueError:
                bound_c=bound_c-0.05
                print("Oops")
        df2=df2.append({"préexpo_"+VAR.split('/')[-1]:2*popt[0],"tau_"+VAR.split('/')[-1]:popt[1]} , ignore_index=True);# Pour chaque solution , on ajoute la préexponentielle et la durée de vie tau à la dataframe
    
        ax.plot(x,np.log(y),label="Intensité réelle");
        ax.plot(x,np.log(f_decay(x,*popt)),label="Intensité estimée");
        ax.set_title(" solution "+df.columns[2*i]);
        ax.set_xlabel('Temps(ms)');
        ax.set_ylabel('log(Intensité(p.d.u))');
        plt.legend();
    plt.show();
    
    return(df2)


# # **<center><font color='blue'> Fonction pour  regression linéaire </font></center>**

# In[9]:


## regression avec linearregression
def regression1(result,std,unk,ss):
    concentration=pd.DataFrame(columns=['polyfit'])
    for t in range(len(ss)): 
        tau=result[result.columns[2*t+1]]
        cc=tau;
        y=np.array(cc); 
        std=np.array(std)
        conc=ss[t]*std/unk
        x=conc;
        n=len(x)
        x=x[1:(n-1)]
        y=y[1:(n-1)]
        plt.scatter(x,y)
        ####Construction de la courbe de calibration des durées de vie 
         #les modéles 
        
        
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x, y) # On effectue une régression linéaire entre les concentrations en solution standard (x) et les durées de vie (y)
        modeleReg1=LinearRegression()
        modeleReg2=RANSACRegressor()      #regression optimal
        mymodel = np.poly1d(np.polyfit(x, y, 1))   #polynome de degré 1
        x=x.reshape(-1,1);
        modeleReg1.fit(x,y);
        modeleReg2.fit(x,y)
        fitLine1 = modeleReg1.predict(x)      #valeurs predites de la regression
        slope2 = modeleReg2.estimator_.coef_[0]
        intercept2 = modeleReg2.estimator_.intercept_
        inlier_mask = modeleReg2.inlier_mask_
        fitLine2 = modeleReg2.predict(x)       #valeurs predites de la regression
        y_intercept = mymodel(0)
        R2=modeleReg2.score(x,y)
        R1=modeleReg1.score(x,y)
        r_value = r2_score(y, fitLine2)
        residuals = y - fitLine2
        R3=r2_score(y, mymodel(x))
        
        # tracer les courbes de calibérations 
        
        plt.plot(x, fitLine1, c='r',label='stats.linregress : R² = {} '.format(round(R1,2)));
        plt.plot(x, mymodel(x),'m',label='np.polyfit : R² = {}'.format(round(R1,)))
        plt.plot(x, fitLine2, color="black",label='RANSACRegressor : R² = {} '.format(round(R3,2)))
        plt.xlabel('Concentration solution standard(ppm)');
        plt.ylabel('durée de vie(ms)');
        plt.title('Courbe de calibration'+'du polluant '+result.columns[2*t+1][4:])
        plt.legend()
        plt.show()
        # calcul des concentrations
        Cx1=-(intercept1)/slope1
        Cx2=-(intercept2)/slope2
        std_err = np.std(residuals)
        roots = np.roots(mymodel)
        x_intercepts = [root for root in roots if np.isreal(root)]
        x_inter=fsolve(mymodel,0)
    
        equation_text1 = 'y = {}x + {}'.format(slope1, intercept1)
        equation_text2 = 'y = {}x + {}'.format(slope2, intercept2)
        print("stats.linregress :",equation_text1, '\n'," polyfit :", mymodel , '\n', "RANSACReg : " , equation_text2)
        concentration=concentration.append({'polyfit':round(x_inter[0],2),'lineaire':Cx1, 'RANSACReg':Cx2},ignore_index=True)
    return(concentration)


# In[10]:


## regression avec linearregression
def regression11(result,std,unk,ss):
    concentration=pd.DataFrame(columns=['polyfit'])
    for t in range(len(ss)): 
        ax1=plt.subplot(211)
        tau=result[result.columns[t]]
        cc=tau;
        y=np.array(cc); 
        std=np.array(std)
        conc=ss[t]*std/unk
        x=conc;
        n=len(x)
        x=x[1:(n-1)]
        y=y[1:(n-1)]
        plt.scatter(x,y);
        ## Construction de la courbe de calibration des durées de vie 
        # les modéles 
        modeleReg2=RANSACRegressor() # regression optimal
        mymodel = np.poly1d(np.polyfit(x, y, 1)) # polynome de degré 1
        x=x.reshape(-1,1);
        modeleReg2.fit(x,y)
        slope2 = modeleReg2.estimator_.coef_[0]
        intercept2 = modeleReg2.estimator_.intercept_
        inlier_mask = modeleReg2.inlier_mask_
        fitLine2 = modeleReg2.predict(x);# valeurs predites de la regression
        y_intercept = mymodel(0)
        R2=modeleReg2.score(x,y)
        r_value = r2_score(y, fitLine2)
        residuals = y - fitLine2
        R3=r2_score(y, mymodel(x))
        # tracer les courbes de calibérations 
        print('\n',f"\033[031m {result.columns[t][4:]} \033[0m",'\n')
        plt.plot(x, mymodel(x),'m',label='np.polyfit : R² = {}'.format(round(R3,2)))
        plt.plot(x, fitLine2, color="black",label='RANSACRegressor : R² = {} '.format(round(R2,2)))
        plt.xlabel('Concentration solution standard(ppm)');
        plt.ylabel('Aire sous la courbe');
        plt.title('Courbe de calibration'+'du polluant '+result.columns[t][4:])
        plt.legend();
        plt.show();
        y_intercept = mymodel(0)
        print("y_intercept:", y_intercept)
        # Calcul des racines (x_intercept)
        roots = np.roots(mymodel)
        x_intercepts = [root for root in roots if np.isreal(root)]
        x_inter=fsolve(mymodel,0)
        print("x_intercepts:", x_inter)
        slope=mymodel.coef[0]
        print("slope", slope)
        # calcul des concentrations
        Cx2=-(intercept2)/slope2
        std_err = np.std(residuals)
        roots = np.roots(mymodel)
        x_intercepts = [root for root in roots if np.isreal(root)]
        x_inter=fsolve(mymodel,0)
        equation_text2 = 'y = {}x + {}'.format(slope2, intercept2)
        print(" polyfit :",equation_text2, '\n', "RANSACReg : " , mymodel)
        concentration=concentration.append({'polyfit':round(x_inter[0],2)},ignore_index=True)
    return(concentration)


# # **<center><font color='blue'> Fonction pour  regression non linéaire </font></center>**
def regression2(result,std,unk,ss,sum_kchel):
    con_poly3=[]
    con2=[]
    for i in range(len(ss)):
        ax1=plt.subplot(211)
        tau=result[result.columns[2*i+1]]
        cc=tau;
        y=np.array(cc); 
        std=np.array(std) 
        conc=ss[i]*std/unk
        x=conc;
        n=len(x)
        x=x[1:(n-1)]
        kchel=sum_kchel[sum_kchel.columns[2*i+1]]
        sum_k=sum_kchel[sum_kchel.columns[2*i+1]]
        kchel=kchel[1:(n-1)]
        mymodel = np.poly1d(np.polyfit(x, kchel, 3))
        plt.scatter(x, kchel)
        plt.plot(x, mymodel(x),'m')
        plt.show() 
        print(mymodel,'\n','R² = {:.5f}'.format(r2_score(kchel, mymodel(x))))
        # Calcul de l'ordonnée à l'origine (y_intercept)
        y_intercept = mymodel(0)
        print("y_intercept:", y_intercept)
        # Calcul des racines (x_intercept)
        roots = np.roots(mymodel)
        x_intercepts = [root for root in roots if np.isreal(root)]
        x_inter=fsolve(mymodel,0)
        print("x_intercepts:", x_inter)
        con_poly3.append(x_inter)
        slope=mymodel.coef[0]
        xinter=y_intercept/slope
        con2.append(xinter)
    return(con_poly3)
# ## On utilise  plusieur fonction pour faire l'ajustement des données , on choisi la meilleure d'entre elle en comparant leur errurs quadratiques 

# In[ ]:





# In[84]:


def regression2(result, std, unk, ss, sum_kchel):
    con_poly3 = []
    con2 = []
    for i in range(len(ss)):
        tau = result[result.columns[2*i+1]]
        cc = tau
        y = np.array(cc)
        std = np.array(std)
        conc = ss[i] * std / unk
        x = conc
        n = len(x)
        x = x[1:(n-1)]
        kchel = sum_kchel[sum_kchel.columns[2*i+1]]
        sum_k = sum_kchel[sum_kchel.columns[2*i+1]]
        kchel = kchel[1:(n-1)]
        
        # Fonction polynôme de degré 1
        poly1 = np.poly1d(np.polyfit(x, kchel, 1))
        poly1_r2 = r2_score(kchel, poly1(x))
        
        # Fonction polynôme de degré 2
        poly2 = np.poly1d(np.polyfit(x, kchel, 2))
        poly2_r2 = r2_score(kchel, poly2(x))
        
        ## Fonction exponentielle
        exp_func = lambda x, a, b, c: np.exp(2*(a * x - b))+c 
        exp_params, _ = curve_fit(exp_func, x, kchel)
        exp_r2 = r2_score(kchel, exp_func(x, *exp_params))
        
        
        # Sélection de la meilleure fonction
        best_func = np.argmax([exp_r2])
        print(best_func)
        
        if best_func == 0:
            
        
            # Polynôme de degré 1
            #best_model = poly1
        #elif best_func == 1:
            
            # Polynôme de degré 2
            #best_model = poly2
        #elif best_func == 2:
            # Exponentielle
            best_model = lambda x: exp_func(x, *exp_params)

        plt.scatter(x, kchel)
        plt.plot(x, best_model(x), 'm')
        plt.show()
        
        print("Best model:", best_model)
        print("R² = {:.5f}".format(r2_score(kchel, best_model(x))))
        if best_func==0:
            y_intercept = exp_func(0, *exp_params)
            print("y_intercept:", y_intercept)
            x_inter = a*np.log(exp_params[1] /2) / exp_params[2]
            x_inter=np.array([x_inter])
            print("x_intercepts:", x_inter)
            slope = -exp_params[1] * exp_func(x_inter, *exp_params)
            con_poly3.append(x_inter)
            con2.append(x_inter)
            
        if best_func==1 or best_func==0:     
            # Calcul de l'ordonnée à l'origine (y_intercept)
            y_intercept = best_model(0)
            print("y_intercept:", y_intercept)
            # Calcul des racines (x_intercept)
            x_inter = fsolve(best_model, 0)
            print("x_intercepts:", x_inter)
    
            con_poly3.append(x_inter)
            slope = best_model.coef[0]
            con2.append(x_inter)
    
    return con_poly3


# In[ ]:





# # **<center><font color='blue'>  Séléctionner les 4 Séries  </font></center>**

# In[12]:


def browseFiles2():
	filename = filedialog.askopenfilenames(initialdir = "http://localhost:8888/tree/Stage",
										title = "Select a File",
										filetypes = (("Csv files",
														"*.csv*"),
													("all files",
														"*.*")))
	return(filename)


# In[13]:


VARS=browseFiles2()


# ## Les fichiers selectionners 

# In[14]:


VARS


# 
# # **<center><font color='blue'> Ce qu'on doit changer dans le code </font></center>**
On modifiera les volumes standards(dans std) , les concentrations  en solution  standrad pour chaque serie , le volume revelatrice , la concentration du polluant A(ref) dans la serie 4 et la concentration du polluant D(ref) dans la seri 3 .
# In[15]:


#unk=3 # volume inconnue 08/06 , 09/06 
#unk=2.80  ## inconnue 10/06 , 12/06 ,15/06 ,  16/06 
unk=2.60 ### 19/06 , 20/06 , 21/06


#unk=3.5 ## 06/06 , 07/06 
#std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2,1]
#std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2]  ## 06/06 
#std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1] # Volume standard 07/06 (50,50)  , 12/06 
#std=[0.00,0.00,0.025,0.050,0.075,0.100,0.125,0.150,0.175,0.200,3.000 ]  ## 15/06 ,  16/06 
#std=[0,0,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.5,1] # Volume standard 08/06 , 09/06 
std=[ 0.00,0.00,0.025,0.075,0.125,0.200,0.500,0.700,1.000,1.500,3.000 ]     ## 19/06
#std=[ 0.00,0.00,0.025,0.075,0.125,0.200,0.500,0.700,1.000,1.500,4.000]   ## 20/06  , 21/06 


ss1=100 # solution standard serie 1
ss2=100 # standard serie 2
ss3=100 # standard serie 3
ss4 =100 # standard serie 4
rev=0.4 # volume reveratrice ## 08/06 , 09/06 , 15/06 , 16/06 , 19/06 , 20/06 
#rev=0.3 ## 06/06   12/06
Ca=10# concentration initiale du polluant A dans la serie 4
Cd=10# concentration initiale du polluant D dans la serie 3


# 
# 
# # **<center><font color='blue'>   Resultats    </font></center>**

# 
# 
# 
# ## **<center><font>  mono_exponentielle </font></center>**

# 
# 
# 
# ### **<center><font>    I )  On fit l'intensité avec une mono_exponentielle puis on calcul les concentrations en utilisant une regressions linéaire ensuite non lineaire ( degré 3)  pour chaque serie    </font></center>**

# In[16]:


Taux4=pd.DataFrame()
for VAR in VARS:
    #print("Serie : " , VAR.split('/')[-1])
    Q=mono_exp(VAR)
    T=pd.concat([Taux4,Q], axis=1)
    Taux4=T
result4=pd.DataFrame()
j=[1,1,2,2,3,3,4,4]
for i in j:
    for k in range(len(Taux4.columns)) : 
        if Taux4.columns[k].find('S'+str(i))!=-1:
            result4=pd.concat([result4,Taux4[Taux4.columns[k]]],axis=1)
result4 = result4.loc[:,~result4.columns.duplicated()]       


# ## **<center><font> Tableau qui contient les taux et les pré_exponentielle de chaque échantillon dans chacune des séries   </font></center>**

# In[17]:


result4.style.background_gradient(cmap="Greens")


# 
# ## **<center><font> I-1) Calcul des concentrations par une regression linéaire  </font></center>**

# In[18]:


ss=[ss1,ss2,ss3,ss4]
concentration4=regression1(result4,std,unk,ss) 


# ## Concentration obtenue  pour chacune des series 

# In[ ]:





# In[19]:


concentration4
serie=['s1','s2','s3','s4']
concentration4.index=serie
concentration4.style.background_gradient(cmap="Greens")


# ### Les concentrations finales pour chaque polluant 

# In[20]:


polyfit=concentration4[concentration4.columns[0]]
r2=cal_conc(*polyfit,Ca,Cd)
r2.style.background_gradient(cmap="Blues")


# In[ ]:





# 
# ## **<center><font>   I-2 ) Calcul des concentrations en utilisant une regression non lineaire ( degré 3) </font></center>**

# ### Calcul kchel et sum_k pour chaque serie 

# In[85]:


def fun(tau):
    sum_k=1/tau
    kch=-sum_k+sum_k[0]
    return(sum_k,kch)
sum_kchel1=pd.DataFrame() # gaussienne
sum_kchel2=pd.DataFrame()# double exp
sum_kchel3=pd.DataFrame() # mono exp
for j in range(4):
    tt3=result4[result4.columns[2*j+1]]
    s_k=pd.DataFrame(fun(tt3))
    s_k=s_k.T
    s_k.columns=['sum_k'+result4.columns[2*j+1].split('_')[-1],'kchel'+result4.columns[2*j+1].split('_')[-1]]
    sum_kchel3=pd.concat([sum_kchel3,s_k],axis=1)


# ### Tableau qui donne le nombre d'ion chélaté et le pourcentage de chaque taux pour chaque série 

# In[86]:


sum_kchel3.style.background_gradient(cmap="Greens")


# In[87]:


ss=[ss1,ss2,ss3,ss4]
concentrationC4=regression2(result4,std,unk,ss,sum_kchel3)


# ## Resultast des concentrations obtenuent dans chaque serie 

# ### Resultats des concentrations de chaque polluant

# In[80]:


concen =pd.DataFrame(concentrationC4)
serie=['s1','s2','s3','s4']
concen.index=serie
concen.style.background_gradient(cmap="Greens")


# In[81]:


r1=cal_conc(*concentrationC4,Ca,Cd)
r1.style.background_gradient(cmap="Greens")


# In[ ]:





# 
# # **<center><font>  méthode double_exponentielle   </font></center>**

# In[90]:


Taux2=pd.DataFrame()
for VAR in VARS:
    #print("Serie : " , VAR.split('/')[-1])
    Q=double_exp(VAR)
    T2=pd.concat([Taux2,Q], axis=1)
    Taux2=T2
result2=pd.DataFrame()
j=[1,1,2,2,3,3,4,4]
for i in j:
    for k in range(len(Taux2.columns)) : 
        if Taux2.columns[k].find('S'+str(i))!=-1:
            result2=pd.concat([result2,Taux2[Taux2.columns[k]]],axis=1)
result2 = result2.loc[:,~result2.columns.duplicated()]   


# ## Tableau qui donne les taux et les pré_exponentielle pour chaque série

# In[91]:


result2.style.background_gradient(cmap="Blues")


# 
# # **<center><font>  calcule de la concentration pour la  regression linéaire  </font></center>**

# In[92]:


ss=[ss1,ss2,ss3,ss4]
concentrationC3=regression1(result2,std,unk,ss) 


# ## Concentration obtenue dans chacune des series

# In[93]:


serie=['s1','s2','s3','s4']
concentrationC3.index=serie
concentrationC3.style.background_gradient(cmap="Blues")


# In[ ]:





# 
# # **<center><font>  Concentration obtenue pour les deux polluants   </font></center>**

# In[95]:


polyfit=concentrationC3[concentrationC3.columns[0]]
r2=cal_conc(*polyfit,Ca,Cd)
r2.style.background_gradient(cmap="Blues")


# # **<center><font> Calcul de la concentration en utilisant laire sous la courbe de l intensité de fluorescence </font></center>**

# In[48]:


### On fixe Tau1 et Tau2 
## Pour differentes valeurs de Tau1 et tau2 
t1=[0.429,0.975,0.639,1.199,1.050,1.090,1.151,1.281,1.311,1.347,0.092]
t2=[0.429,0.423,0.367,0.452,0.428,0.434,0.434,0.434,0.441,0.435,1.317]
l=0
while(l<len(t1)):
    print(colored('les valeurs de Tau : ','red', attrs=['reverse', 'blink']))
    print('Tau1=',t1[l],'et' , 'Tau2=',t2[l])
    Taux4=pd.DataFrame()
    T1=t1[l]   
    T2=t2[l]
    for VAR in VARS:
        #print("Serie : " , VAR.split('/')[-1])
        Q=double_exp2(VAR,T1,T2)
        T=pd.concat([Taux4,Q], axis=1)
        Taux4=T
    result4=pd.DataFrame()
    j=[1,1,2,2,3,3,4,4]
    for i in j:
        for k in range(len(Taux4.columns)) : 
             if Taux4.columns[k].find('S'+str(i))!=-1:
                result4=pd.concat([result4,Taux4[Taux4.columns[k]]],axis=1)
    result4 = result4.loc[:,~result4.columns.duplicated()] 
    ss=[ss1,ss2,ss3,ss4]
    concentrationC1=regression11(result4,std,unk,ss) 
    serie=['s1','s2','s3','s4']
    concentrationC1.index=serie
    print(colored('concentration obtenue avec les courbes de calibrations ','red', attrs=['reverse', 'blink'] ))
    print(concentrationC1)
    polyfit=concentrationC1[concentrationC1.columns[0]]
    r2=cal_conc(*polyfit,Ca,Cd)
    r5=pd.concat([r2],axis=1)
    r5.columns=['polyfit']
    print(colored('Concentrations inconnues des polluants', 'red', attrs=['reverse', 'blink'] ))
    print(r5)
    l=l+1


# In[ ]:





# ## Une des resultats en detail ( tau1= 0.092 et tau2 = 1.317 )

# ## Les Aires pour chacune des series 

# In[49]:


result4.style.background_gradient(cmap="Greens")


# In[50]:


ss=[ss1,ss2,ss3,ss4]
concentrationC1=regression11(result4,std,unk,ss) 


# ## Concentration obtenue pour chaque courbe de calibration lineaire 

# In[332]:


concentrationC1
serie=['s1','s2','s3','s4']
concentrationC1.index=serie
concentrationC1.style.background_gradient(cmap="Purples")


# ## Concentration inconnue des polluants 

# In[51]:


polyfit=concentrationC1[concentrationC1.columns[0]]
r2=cal_conc(*polyfit,Ca,Cd)
r5=pd.concat([r2],axis=1)
r5.columns=['polyfit']
r5.style.background_gradient(cmap="Purples")


# 
# # **<center><font>  calcule de la concentration pour la  regression non linéaire  </font></center>**

# In[ ]:





# In[ ]:





# In[54]:


def fun(tau):
    sum_k=1/tau
    kch=-sum_k+sum_k[0]
    return(sum_k,kch)
sum_kchel1=pd.DataFrame() # gaussienne
sum_kchel2=pd.DataFrame()# double exp
sum_kchel3=pd.DataFrame() # mono exp
for j in range(4):
    tt3=result2[result2.columns[2*j+1]]
    s_k=pd.DataFrame(fun(tt3))
    s_k=s_k.T
    s_k.columns=['sum_k'+result2.columns[2*j+1].split('_')[-1],'kchel'+result2.columns[2*j+1].split('_')[-1]]
    sum_kchel2=pd.concat([sum_kchel2,s_k],axis=1)
sum_kchel2.style.background_gradient(cmap="Blues")


# In[55]:


ss=[ss1,ss2,ss3,ss4]
concentration3=regression2(result2,std,unk,ss,sum_kchel2) 


# ## Concentration obtenue pour chacune des series

# In[341]:


concen =pd.DataFrame(concentration3)
serie=['s1','s2','s3','s4']
concen.index=serie
concen.style.background_gradient(cmap="Blues")


# In[ ]:





# ### Resultats des concentrations de chaque polluant

# In[342]:


r1=cal_conc(*concentration3,Ca,Cd)
r1.style.background_gradient(cmap="Blues")


# 
# # **<center><font>  méthode gaussiennes    </font></center>**

# In[343]:


Taux=pd.DataFrame()
for VAR in VARS:
    #print("Serie : " , VAR.split('/')[-1])
    Q=tri_exp(VAR)
    T=pd.concat([Taux,Q], axis=1)
    Taux=T
result=pd.DataFrame()
j=[1,1,2,2,3,3,4,4]
for i in j:
    for k in range(len(Taux.columns)) : 
        if Taux.columns[k].find('S'+str(i))!=-1:
            result=pd.concat([result,Taux[Taux.columns[k]]],axis=1)
result = result.loc[:,~result.columns.duplicated()]       


# 
# # **<center><font>  resultats calcul de Taux et préexponentielle   </font></center>**

# In[ ]:


result.style.background_gradient(cmap="Purples")


# 
# # **<center><font>  resultats regression  linéaire  </font></center>**

# In[ ]:


ss=[ss1,ss2,ss3,ss4]
concentrationC1=regression1(result,std,unk,ss) 


# ## Concentration obtenue pour chacune des series

# In[ ]:


concentrationC1
serie=['s1','s2','s3','s4']
concentrationC1.index=serie
concentrationC1.style.background_gradient(cmap="Purples")


# ### Resultats des concentrations de chaque polluant

# In[ ]:


concentrationC1=concentrationC1[concentrationC1.columns[2]]
r1=cal_conc(*concentrationC1,Ca,Cd)
r1.style.background_gradient(cmap="Purples")


# 
# # **<center><font>  resultats regression non  linéaire  </font></center>**

# In[628]:


for j in range(4):
    tt1=result[result.columns[2*j+1]]
    s_k=pd.DataFrame(fun(tt1))
    s_k=s_k.T
    s_k.columns=['sum_k'+result.columns[2*j+1].split('_')[-1],'kchel'+result.columns[2*j+1].split('_')[-1]]
    sum_kchel1=pd.concat([sum_kchel1,s_k],axis=1)
sum_kchel1


# In[629]:


ss=[ss1,ss2,ss3,ss4]
concentration1=regression2(result,std,unk,ss,sum_kchel1) 


# ## Valeur des concentrations pour chacune des series 

# In[630]:


concen =pd.DataFrame(concentration1)
serie=['s1','s2','s3','s4']
concen.index=serie
concen.style.background_gradient(cmap="Purples")


# 
# # **<center><font>  Concentration obtenue pour les deux polluants   </font></center>**

# In[ ]:





# In[631]:


r2=cal_conc(*concentration1,Ca,Cd)
r2.style.background_gradient(cmap="Purples")


# In[ ]:





# In[ ]:





# In[ ]:






  ''',language='python')

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
    "Code_classification_polluants_heterogene":Code_classification_polluants_heterogene,
    "Code_lissage_deconvolution_spectrale":Code_lissage_deconvolution_spectrale,
    " Quatification_polluant_heterogene":Quatification_polluant_heterogene,
    "Code_apprentissage_3D":Code_apprentissage_3D	
	
}

selected_page = st.sidebar.selectbox("Selectionner ", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
