import streamlit as st

import numpy as np

import pandas as pd

import ydata_profiling

from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup as setup_reg

from pycaret.regression import compare_models as compare_models_reg

from pycaret.regression import save_model as save_model_reg

from pycaret.regression import plot_model as plot_model_reg




from pycaret.classification import setup as setup_class

from pycaret.classification import compare_models as compare_models_class

from pycaret.classification import save_model as save_model_class

from pycaret.classification import plot_model as plot_model_class

import mlflow




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

### les bilbiotheque pour la partie quantification  un seul polluant .... 

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
#from tensorflow.keras.preprocessing import image

#url="https://www.linkedin.com/in/alioune-gaye-1a5161172/"




### faire un catching (garder en memoire tout ce qui est deja calculer)

@st.cache



def load_data(file):

    data=pd.read_csv(file)

    return data

### la fonction principale ...

 

def main():

    ## l'entéte de mon code

    st.title('Alioune Gaye : mon appplication AutoML')

    #st.sidebar.write("[Author : Gaye Alioune](%)" % url)

    st.sidebar.markdown(

        "**This wep app is a No-code tool for Exploratory Data Analysis and building Machine Learning model for R**"

        "1.Load your dataset file (CSV file);\n"

        "2.Click on *profile Dataset* button in order to generate the pandas profiling of the dataset;\n"

        "3. Choose your target column ;\n"

        "4.Choose the machine learning task (Regression or Classification);\n"

        "5.Click on *Run Modeling * in order to start the training process.\n"

        "When the model is built , you can view the results like the pipeline model , Residuals plot , Roc Curve, confusion Matrix ..."

        "\n6. Download the Pipeline model in your local computer."

 

    )

 

   

 

    ## Charger le jeu de donnée

    file=st.file_uploader("Upload your dataset in csv format", type=["csv"])

    if file is not None: ## pour dire à l'utilisateur , si le fichier importer n'est pas nul alors fait ceci

        data=pd.read_csv(file)

        st.dataframe(data.head()) ## afficher les données importer

        #data=data.dropna(subset=target) ### suprimer les valeurs manquantes

        ## analyse exploiratoire du jeu de données

        ## creation d'un bouton de visualisation des données et graphe

        profile=st.button('profile dataset')

        if profile:

            profile_df=data.profile_report()

            st_profile_report(profile_df) ### afficher le profile

            ## Phase de modelisation

            ## Choix des targets

        target=st.selectbox('Select the target variable',data.columns)

        ## selection du type de model (classification ou regression)

        task=st.selectbox('Select a ML task', ["Classification","Regression"])

        ## Maintenant on peut commencer par ecrir le code

        ##Pour la regression et la classification

        if task=="Regression":

            if st.button("Run Modelling"):

                exo_reg= setup_reg(data,target=target)

                ## entrainer plusieurs models à la fois

                st.dataframe(exo_reg)

                model_reg=compare_models_reg()

                ### sauvegarder le model

                save_model_reg(model_reg,"best_reg_model")

                ### Message de succé si tout ce passe bien

                st.success("Regression model built successfully")

                ## Results

                ### les residus

                st.write("Residuals")

                plot_model_reg(model_reg,plot='residuals',save=True)

                st.image("Residuals.png") ### Sauvegarder le resultat

                ### Variables importantes

                st.write("Feature importance")

                plot_model_reg(model_reg,plot='feature',save=True)

                st.image("Feature Importance.png")

                ### Telecharger le pipeline

                with open('best_reg_model.pkl','rb') as f:

                    st.download_button('Download Pipeline Model',f,file_name="best_reg_model.pkl")

 

   

 

        if task=="Classification":

            if st.button("Run Modelling"):

                exo_class= setup_class(data,target=target,index=False,train_size =0.80,normalize = True,normalize_method = 'zscore',remove_multicollinearity = True,log_experiment=True, experiment_name="polluant-homogene"

                   ,pca =False, pca_method =None,

                   pca_components =None)

                st.write('les caracteristiques de notre setup')

                ## entrainer plusieurs models à la fois

                model_class=compare_models_class()

                tuned_model_class = tune_model(model_class)

                st.write('Votre meilleur model de classification est ', model_class)

                ### sauvegarder le model une fois qu'on es satisfait du model

                final_model1 = finalize_model(model_class)  ### notre pipeline(entrainement du model sur tout les donnée)

                save_model_class(final_model1,"best_class_model")

                st.write("notre pipeline",save_model_class(model_class,"best_class_model"))

                ### Message de succé si tout ce passe bien

                st.write('Les metrics')

                st.dataframe(pull(), height=200)

 

                st.success("Classification model built successfully")

 

                ## ResuLts

                col5, col6,col7,col8=st.columns(4)

                with col5:

                    st.write("ROC curve")

                    plot_model_class(model_class,save=True)

                    st.image("AUC.png")

 

                #with col6:

                 #   plot_model_class(model_class,plot='class_report',display_format='streamlit',save=True)

                  #  st.image("Class_repport.png")

 

                with col7:

                    st.write("Confusion Matrix")

                    plot_model_class(model_class,plot='confusion_matrix',save=True)

                    st.image("Confusion Matrix.png")

 

                with col8:

                    st.write("Feature Importance")

                    plot_model_class(model_class,plot='feature',save=True)

                    st.image("Feature Importance.png")

 

                col9,col10 =st.columns(2)

                #with col9:

                 #   st.write("Boundary")

                  #  plot_model_class(tuned_model_class,plot='boundary',display_format='streamlit',save=True)

                   # st.image("Boundary.png")

 

                ###prediction avec les données de test

                st.write("La prediction du model avec les données de test")    

                prediction=predict_model(final_model1)

                st.dataframe(prediction,height=200)

 

                ## Download the pipeline model

                with open('best_class_model.pkl','rb') as f:
                    st.download_button('Download Pipeline Model',f,file_name="best_class_model.pkl")
    else:

        st.image("https://cdn.futura-sciences.com/cdn-cgi/image/width=1280,quality=60,format=auto/sources/images/data_science_1.jpg")                    



    ### deploiement de notre model machine learning .

    # Prediction via mmlflow      

    file_1=st.file_uploader("Upload your dataset à predir  in csv format", type=["csv"])    
    if file_1 is not None: ## pour dire à l'utilisateur , si le fichier importer n'est pas nul alors fait ceci
            data1=pd.read_csv(file_1)
            n=len(data1.columns)
            data2=data1[data1.columns[0:(n-1)]]    
            st.write('les données que vous voulez predire est:',data2)  
            #logged_model = 'runs:/42ae053461bc4e4c9cd8faded887aeaa/model' ### chemin de mon meilleur model qui se trouve dans le dosier "mlruns"

            # Load model as a PyFuncModel.

            #loaded_model = mlflow.pyfunc.load_model(logged_model)

            #st.write('le model enregidtrer sur mlflow est',loaded_model)


            loaded_model=load_model('best_class_model')
            ### Affichage des resultats de la predition
            ### data2  est les données ----- à predir dans le future .............
            if st.button("Run prediction"):    
               prediction=predict_model(loaded_model,data=data2)
               #### On importe les données à predire dans le future , avec le codde deploié sur mlflow ........ Donc , on essaye de toujours  faire l'exploitationn des données aavent de les importer  dans le code
               st.write('la prediction de votre jeux de donner est:')
               st.dataframe(prediction.iloc[:,[len(prediction.columns)-2,len(prediction.columns)-1]],height=200)


    else:
        st.image("https://cdn.futura-sciences.com/cdn-cgi/image/width=1280,quality=60,format=auto/sources/images/data_science_1.jpg")    

             
            
    ### quatification 
                                 
    quant=st.selectbox('selectionner la methodce de quantification , un polluant ou pour deux polluant ', ['quantif_double_pol'])
    
    if quant=='quantif_double_pol':
        def find_delimiter(filename):
           sniffer = csv.Sniffer()
           with open(filename) as fp:
            delimiter = sniffer.sniff(fp.read(5000)).delimiter
           return delimiter
        
        
        ## On essye d'importer les codes qui correspondent aux calcules de la concentration pour un seul polluaqnt 
        ### dans le melange . Ainsi, on essaye d'afficher les valeurs de x-intercept , y-intercept , slope , delta-x
        ### la fonction utiliser pour fiter les données .
        ### Enfin , on doonne une conclusion pour chaque fichier importer .
        ### 1-on importe d'abord le fichier
        ### donner les concentration de la solution standard de meme que la solution revelatrice qu'on a utilisé pour ces donnnées 
        ### apres on donne une liste standard qui regroupe les volumes des differentes ajouts qu'on a fait dans le standard
        ## puis la valeur de la solution revalatrice 
        ### la valeur de la solution inconnue
        ### on donne qussi la valeur de la concentration du polluant initialement .
        ## On oublie pas que les valeurs revalatrice et inconnues sont toujours fixe 
        ### On met la fonction qui calcule les valeurs de tau  et de la préexponentielle sous format dataframe puis ,on l'affiche 
        ##  on met la fonction qui calculera la meilleur fonction d'ajustement : puis on  retourne , le y-intercept , x-intercept (qui correspondera à notre concentration )
        ### on affiche aussi , la courbe d'ajustement . 
        ## puis on determoine le delta taux 
        ### Et enfin , afficher la valeur de delta_x  pour comparer avec -10 ....add() 
        st.title(":blue[les variables de notre systeme d'équation :]")
        st.markdown("- $C_{HD}$ :  la concentration obtenue dans la serie 1")
        st.markdown("- $C_{HA}$ :  concentration obtenue dans la serie 2")
        st.markdown("- $C_{DA}$ :  concentration obtenue dans la serie 3")
        st.markdown("-  $C_{AD}$ :  concentration obtenue dans la serie 4")
        st.markdown("- $C_D^0$ : concentration initiale du polluant 1" )
        st.markdown("- $C_A^0$ : concentration initiale du polluant 2")
        st.title(":blue[Le systeme d'équation] :")
        
        st.markdown("- $ C_{HD} = C_D + K_{A-D}C_A $     =>  Serie 1 : mélange dans standard 1 (D)")
        st.markdown("- $ C_{HA}= C_A + K_{D-A}C_D $      =>  Serie 2 : mélange dans standard 2 (A)")
        st.markdown("- $ C_{DA} = K_{D-A}C_D^0 $         =>  Serie 3 : polluant 1 dans standard 2") 
        st.markdown("- $C_{AD} = K_{A-D}C_A^0$           =>  Serie 4 : polluant 2 dans la standard 1")
        st.markdown(" Cherchons les inconnues ($C_A$ et $C_D$)")
        
        def cal_conc(x,y,z,h,Ca,Cd):
          a=h/Ca
          a1=z/Cd
          C_A=(y-a1*x)/(1-a1*a)
          C_D=(x-a*y)/(1-a1*a)
          conc=pd.DataFrame([C_A,C_D])
          conc.index=['C_A','C_D']
          return(conc) 
        ###Monoexpo
        def mono_exp(VAR):
            delimit=find_delimiter(VAR)
            df=pd.read_csv(VAR,sep=delimit);
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
            fig, axs = plt.subplots(nrows=3, ncols=row, figsize=(20, 20))
            for ax, i in zip(axs.flat, range(int(ncol/2))):
                x=df[df.columns[0]]; # temps
                y=df[df.columns[(2*i)+1]]; # Intensités de fluorescence
                popt,pcov=curve_fit(f_decay,x,y,bounds=(0, np.inf));
                df1=df1.append({'A'+VAR.split('/')[-1] :popt[0] , 'Tau'+VAR.split('/')[-1] :popt[1]} , ignore_index=True)
                ax.plot(x,y,label="Intensité réelle");
                ax.plot(x,f_decay(x,*popt),label="Intensité estimée");
                ax.set_title(" solution "+df.columns[2*i]);
                ax.set_xlabel('Temps(ms)');
                ax.set_ylabel('Intensité(p.d.u)');
                plt.legend();
            plt.show();
            return(df1)  
        ### double expo 
        
        
        
        
        def double_exp(VAR):
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
            def f_decay(x,a1,T1,a2,T2,r):
                return(r+a1*np.exp(-x/T1)+a2*np.exp(-x/T2));
    
            df1=pd.DataFrame(columns=['A'+VAR.split('/')[-1],'Tau'+VAR.split('/')[-1]]);
            for i in range(int(ncol/2)):
                x=df[df.columns[0]]; # temps
                y=df[df.columns[(2*i)+1]]; # Intensités de fluorescence
                y=list(y)
                y0=max(y)#y[1]
                popt,pcov=curve_fit(f_decay,x,y,bounds=(0.1,[y0,+np.inf,y0,+np.inf,+np.inf]));
                tau=(popt[0]*(popt[1])**2+popt[2]*(popt[3])**2)/(popt[0]*(popt[1])+popt[2]*(popt[3]))
                A=(popt[0]+popt[2])/2
                df1=df1.append({'A'+VAR.split('/')[-1] :A , 'Tau'+VAR.split('/')[-1] :tau} , ignore_index=True);
            return(df1)   
        
        ### La gaussienne :
        
        
        def tri_exp(VAR):
            delimit=find_delimiter(VAR)
            df=pd.read_csv(VAR,sep=delimit)
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
                        popt,pcov=curve_fit(f_decay,x,y,bounds=(0,[yo,+np.inf,bound_c,+np.inf]),method='dogbox') # On utilise une regression non linéaire pour approximer les courbes de durée de vie  
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
        
        
        ## Linéaire :
        
        
        
        def regression1(result,std,unk,ss):
            concentration=pd.DataFrame(columns=['polyfit','stats_lingress','ransac'])
            for t in range(len(ss)): 
                ax1=plt.subplot(211)
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
                ####Construction de la courbe de calibration des durées de vie 
                # les modéles 
                slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x, y);# On effectue une régression linéaire entre les concentrations en solution standard (x) et les durées de vie (y)
                modeleReg1=LinearRegression()
                modeleReg2=RANSACRegressor() # regression optimal
                mymodel = np.poly1d(np.polyfit(x, y, 1)) # polynome de degré 1
                x=x.reshape(-1,1);
                modeleReg1.fit(x,y);
                modeleReg2.fit(x,y)
                fitLine1 = modeleReg1.predict(x);# valeurs predites de la regression
                slope2 = modeleReg2.estimator_.coef_[0]
                intercept2 = modeleReg2.estimator_.intercept_
                inlier_mask = modeleReg2.inlier_mask_
                fitLine2 = modeleReg2.predict(x);# valeurs predites de la regression
                y_intercept = mymodel(0)
                R2=modeleReg2.score(x,y)
                R1=modeleReg1.score(x,y)
                r_value = r2_score(y, fitLine2)
                residuals = y - fitLine2
                R3=r2_score(y, mymodel(x))
                # tracer les courbes de calibérations 
                print('\n',f"\033[031m {result.columns[2*t+1][4:]} \033[0m",'\n')
                plt.plot(x, fitLine1, c='r',label='stats.linregress : R² = {} '.format(round(R1,2)));
                plt.plot(x, mymodel(x),'m',label='np.polyfit : R² = {}'.format(round(R3,2)))
                plt.plot(x, fitLine2, color="black",label='RANSACRegressor : R² = {} '.format(round(R2,2)))
                plt.xlabel('Concentration solution standard(ppm)');
                plt.ylabel('durée de vie(ms)');
                plt.title('Courbe de calibration'+'du polluant '+result.columns[2*t+1][4:])
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
                Cx1=-(intercept1)/slope1;
                Cx2=-(intercept2)/slope2
                std_err = np.std(residuals)
                roots = np.roots(mymodel)
                x_intercepts = [root for root in roots if np.isreal(root)]
                x_inter=fsolve(mymodel,0)
                equation_text1 = 'y = {}x + {}'.format(slope1, intercept1)
                equation_text2 = 'y = {}x + {}'.format(slope2, intercept2)
                print("stats.linregress :",equation_text1, '\n'," polyfit :",equation_text2, '\n', "RANSACReg : " , mymodel)
                concentration=concentration.append({'polyfit':round(x_inter[0],2),'stats_lingress':round(Cx1,2),'ransac':round(Cx2,2)},ignore_index=True)
            return(concentration)
        
        
        
    
            
            
                 
            
            
        
        
       
            
            
        
    
        
        #### Polluant 1 
        
        
        def quantif(df):
            for i in df.columns:
                if (df[i].isnull()[0]==True): # On elimine les colonnes vides
                    del df[i];
            df=df.dropna(axis=0); # On elimine les lignes qui contiennent des na;
            df=df[1:];
            df=df.astype(float); # On convertit les valeurs contenues dans les colonnes en float (à la place de string)
            df=df[df[df.columns[0]]>=0.1] ### on selectionne les durée à partir de 0.1 
            ncol=(len(df.columns)) # nombre de colonnes
            najout=(ncol/2)-2 # nombre d'ajouts en solution standard ( le nombre d'echgantillons sans les ions et excée)
            print(najout)
            def f_decay(x,a1,b1,c,r): # Il s'agit de l'équation utilisée pour ajuster l'intensité de fluorescence en fonction du temps(c'est à dire la courbe de durée de vie)
                return(a1*np.exp(-x/b1)+(a1/2)*np.exp(-x/(b1+1.177*c))+(a1/2)*np.exp(-x/(b1-1.177*c)) + np.exp(-r))
                                           
            df2=pd.DataFrame(columns=["A","tau"]); # Il s'agit du dataframe qui sera renvoyé par la fonction
            #### Ajustement des courbes de durée de vie de chaque solution en fonction du temps#### 
            for i in range(int(ncol/2)):
                x=df[df.columns[0]]; # temps
                y=df[df.columns[(2*i)+1]]; # Intensités de fluorescence
            # yo=max(y)# la plus grand valeur de y 
            y=list(y)
            yo=y[1]
            bound_c=1.0
            x_max=max(x)
            while True:
                try:
                    popt,pcov=curve_fit(f_decay,x,y,bounds=(0.1,[yo,+np.inf,bound_c,+np.inf]),method='dogbox') # On utilise une regression non linéaire pour approximer les courbes de durée de vie  
                    #popt correspond aux paramètres a1,b1,c,r de la fonction f_decay de tels sorte que les valeurs de f_decay(x,*popt) soient proches de y (intensités de fluorescence)
                    break;
                except ValueError:
                    bound_c=bound_c-0.05
                    print("Oops")
           #plt.plot(x,y,label="Intensité réelle");
           #plt.plot(x,f_decay(x,*popt),label="Intensité estimée");
           #plt.title("Ajustement de la courbe de durée de vie de la solution "+df.columns[2*i]+' du polluant '+VAR.split('/')[-1].split('.')[0]);
           #plt.xlabel('Temps(ms)');
           #plt.ylabel('Intensité(p.d.u)');
           #plt.legend();
           #plt.show();
            df2=df2.append({"A":2*popt[0],"tau":popt[1]} , ignore_index=True);# Pour chaque solution , on ajoute la préexponentielle et la durée de vie tau à la dataframe
            return(df2)
        ### On importe les données ::::
        file2= st.file_uploader(" Importer vos données  en Format csv ", type=["csv"])
        if file2 is not None: ## pour dire à l'utilisateur , si le fichier importer n'est pas nul alors fait ceci
            data=pd.read_csv(file2,sep=';') ### lire le fichier 
            st.dataframe(data)  ## affichage des données 
            ## maintenant passons au calcule de taux et du préexponentielle
            Q=quantif(data)
            delta_tau=(Q['tau']-Q['tau'][0])*100/Q['tau'][0]  ## calcul de delta taux 
            delta_tau=list(delta_tau) ### on met delta_tau sous forme de list
            delta_tau=pd.DataFrame(delta_tau) ### on met delta_tau sous forme de dataframe
            delta_tau.columns=['delta_tau'] ####  on renome la colonne de delta_tau
            tau=Q['tau']  
            ## maintenant on affichage  le tableau de delta_tau sur notre application streamlit
            st.write(Q)    
        
        
        
        
        
           
           
           
           
           
    
if __name__=='__main__':

 

    main()    

 

   

 

 