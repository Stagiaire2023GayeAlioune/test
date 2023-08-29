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
import streamlit as st

url = "https://www.linkedin.com/in/sokhna-faty-bousso-110891190/"
@st.cache_data
def load_data(file):
    data=pd.read_csv(file)
    return data

def identification():
    st.sidebar.markdown('<h1 style="text-align: center;">Identification du polluants 🎈</h1>', unsafe_allow_html=True)
    def main():
        st.markdown('<h1 style="text-align: center;">Identification du polluants</h1>', unsafe_allow_html=True)
        st.markdown('<h1 style="text-align: center;">Base de donnée</h1>',unsafe_allow_html=True)
        col3,col4=st.sidebar.columns(2)
        col3.image("https://ilm.univ-lyon1.fr/templates/mojito/images/logo.jpg", use_column_width=True)
        col4.image("https://formation-professionnelle.universite-lyon.fr/var/site/storage/images/3/3/5/0/533-17-fre-FR/Lyon-1-Claude-Bernard.png", use_column_width=True)
        st.sidebar.write("<p style='text-align: center;'> Sokhna Faty Bousso : Stagiaire ILM (%s)</p>" % url, unsafe_allow_html=True)
        st.sidebar.write("<p style='text-align: center;'>Apprentissage par régression ou classification.</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>Nous allons procéder comme suit :</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>1 - Chargement des données</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>2 - Analyse exploratoire des données</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>3 - Sélection de la cible et de la méthode d'apprentissage</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>4 - Construction du modèle</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>5 - Téléchargement du modèle</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>6 - Prédiction</p>", unsafe_allow_html=True)
        file = st.file_uploader("entrer les données ", type=['csv'])
        if file is not None:
            df=load_data(file)
            #type=st.selectbox("selectionner le target",["Homogene","Heterogene"])
            n=len(df.columns)
            X=df[df.columns[:(n-1)]]# on prend les variables numériques 
            y=df[df.columns[-1]] # le target
            st.dataframe(df.head())
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
                     st.success("youpiiiii classification fonctionne \U0001F604")
                     st.write("Performances du modèle :")
                
                     final_model1 = create_model(r,fold=5,round=2)
                     col5,col6=st.columns(2)
                     col5.write('AUC')
                     plot_model(final_model1,plot='auc',save=True)
                     col5.image("AUC.png")
                     col6.write("class_report")
                     plot_model(final_model1,plot='class_report',save=True)
                     col6.image("Class Report.png")
                  
                     col7,col8=st.columns(2)
                     col7.write("confusion_matrix")
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
                    st.success("youpiiiii classition fonctionne")
                    final_model1 = create_model_reg(r)
        
        else:
            st.image("https://ilm.univ-lyon1.fr//images/slides/SLIDER10.png")

        
        
    if __name__ == "__main__":
         main()
         
    st.markdown('<h1 style="text-align: center;">Prédiction</h1>', unsafe_allow_html=True)

    def main():
        file_to_predict = st.file_uploader("Choisir un fichier à prédire", type=['csv'])
        if file_to_predict is not None:
            #rain(emoji="🎈",font_size=54,falling_speed=5,animation_length="infinite",)
            df_to_predict = load_data(file_to_predict)
            st.subheader("Résultats des prédictions")
            def predict_quality(model, df):
                  predictions_data = predict_model(estimator = model, data = df)
                  return predictions_data
    
            model = load_model('best_model')
            pred=predict_quality(model, df_to_predict)
            st.dataframe(pred[pred.columns[-3:]].head())
        else:
            st.image("https://ilm.univ-lyon1.fr//images/slides/Germanium%20ILM.jpg")
    if __name__ == "__main__":
        main()
        


def image():
    st.markdown("# image 3D ❄️")
    st.sidebar.markdown('<h1 style="text-align: center;">Identification d\'images ❄️ </h1>', unsafe_allow_html=True)
    col3,col4=st.sidebar.columns(2)
    col3.image("https://ilm.univ-lyon1.fr/templates/mojito/images/logo.jpg", use_column_width=True)
    col4.image("https://formation-professionnelle.universite-lyon.fr/var/site/storage/images/3/3/5/0/533-17-fre-FR/Lyon-1-Claude-Bernard.png", use_column_width=True)
    st.sidebar.write("<p style='text-align: center;'> Sokhna Faty Bousso : Stagiaire ILM (%s)</p>" % url, unsafe_allow_html=True)
    st.sidebar.write("<p style='text-align: center;'> modèle pré-entrainé </p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>Nous allons procéder comme suit :</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>1 - Aprés avoir la base d'images , nous avons divisé la base de données en ensembles d'entraînement  et de validation </p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>2 - créé un modèle de classification basé sur le modèle VGG16 en utilisant Keras</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>3 -  utilisé les 15 premiers couches du modèle </p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>4 - ajouté une couche de classification à la sortie du modèle avec une activation sigmoid pour la classification multi-classes.</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>5 - compilé le modèle en utilisant une fonction de perte de binary_crossentropy (car les étiquettes sont encodées en tant que vecteurs binaires) et un optimiseur SGD</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>6 -  créé des générateurs d'images d'apprentissage  et de validation pour alimenter le modèle pendant l'entraînement, en utilisant ImageDataGenerator de Keras pour augmenter les données d'entraînement (rotation, ré-échelle, retournement, etc.).</p>", unsafe_allow_html=True)
    from keras.models import load_model
    st.markdown('<h1 style="text-align: center;">Prédiction image 3D </h1>', unsafe_allow_html=True)
    model = load_model('model_final2.h5')
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
    if __name__ == "__main__":
        main()
   


def Quantification():
    st.markdown('<h1 style="text-align: center;"> Quantification du polluants 🎉 </h1>', unsafe_allow_html=True)
    st.sidebar.markdown('<h1 style="text-align: center;"> Quantification du polluants 🎉 </h1>', unsafe_allow_html=True)
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
    def double_exp(df,VAR):
        T1=0.3
        T2=1.3
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
    def regression2(result,std,unk,ss,sum_kchel):
        col1, col2 ,col3,col4= st.columns(4)
        col=[col1,col2,col3,col4]
        con_poly3=[]
        con2=[]
        for i in range(len(ss)):
           fig, ax = plt.subplots()
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
           #st.write(f"\033[031m {result.columns[2*i+1][4:]} \033[0m")
           plt.scatter(x, kchel)
           plt.plot(x, mymodel(x),'m')
           plt.xlabel('Concentration solution standard(ppm)');
           plt.ylabel('nombre d\'ion chélaté ' );
           plt.title('Courbe de calibration'+'du polluant '+result.columns[2*i+1][4:])
           plt.legend();
           col[i].pyplot(fig)
           col[i].write(r2_score(kchel, mymodel(x)))
           # Calcul de l'ordonnée à l'origine (y_intercept)
           y_intercept = mymodel(0)
           col[i].write("y_intercept")
           col[i].write(y_intercept)
           # Calcul des racines (x_intercept)
           roots = np.roots(mymodel)
           x_intercepts = [root for root in roots if np.isreal(root)]
           x_inter=fsolve(mymodel,0)
           con_poly3.append(x_inter)
           col[i].write("x_intercept")
           col[i].write(x_inter)
           slope=mymodel.coef[0]
           col[i].write("slope")
           col[i].write(slope)
           xinter=y_intercept/slope
           con2.append(xinter)
        return(con_poly3)
    
    Taux4 = pd.DataFrame()
    def main():
        col3,col4=st.sidebar.columns(2)
        col3.image("https://ilm.univ-lyon1.fr/templates/mojito/images/logo.jpg", use_column_width=True)
        col4.image("https://formation-professionnelle.universite-lyon.fr/var/site/storage/images/3/3/5/0/533-17-fre-FR/Lyon-1-Claude-Bernard.png", use_column_width=True)
        st.sidebar.write("<p style='text-align: center;'> Sokhna Faty Bousso : Stagiaire ILM (%s)</p>" % url, unsafe_allow_html=True)
        st.sidebar.write("<p style='text-align: center;'>  Mélange de polluants  </p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>1 - Méthode mono_exponentielle : </p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>2 - Méthode double_exponentielle :</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>3 - Méthode gaussienne  : </p>", unsafe_allow_html=True) 
        st.sidebar.markdown("<p style='text-align: center;'>Dans chaque méthode nous allons procéder comme suit : Calcul de la concentration dans chaque série  à partir de (durée de vie et nombre d'ion chélaté ) en fonction de la concentration standard dans chaque mélange avec une regression linéaire et non linéaire afin d'utiliser le systéme d'équation (P) pour trouver la concentration de chaque polluant dans le mélange\n </p>", unsafe_allow_html=True)
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
                    df = pd.read_csv(uploaded_file, delimiter=",")
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
                concentration4=regression1(Taux4,std,unk,ss,3) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
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
                       df = pd.read_csv(uploaded_file, delimiter=",")
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
                concentration4=regression1(Taux4,std,unk,ss,3) 
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
                       df = pd.read_csv(uploaded_file, delimiter=",")
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
                concentration4=regression1(Taux4,std,unk,ss,3) 
                col1.write(concentration4)
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Purples"))

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
                polyfit=concentration4[concentration4.columns[0]]
                r2=cal_conc(*polyfit,Ca,Cd)
                col2.write(r2.style.background_gradient(cmap="Purples"))
    if __name__ == "__main__":
         main()
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt;
from tkinter import filedialog
from tkinter import *
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit;
import math
import scipy.integrate as spi
import csv
def liss_deconv():
    def main():
    
        uploaded_files = st.file_uploader("entrer les données ", type=['csv'])
        st.image("https://ilm.univ-lyon1.fr//images/slides/CARROUSSEL-15.png")
        def find_delimiter(filename):
            sniffer = csv.Sniffer()
            with open(filename) as fp:
                delimiter = sniffer.sniff(fp.read(5000)).delimiter
            return delimiter
        def expS(x,I,m,b):
            return(I*np.exp(-((x-m)**2)/(2*(b**2)))/(b*np.sqrt(2*np.pi)))
        def expT(x,I1,m1,b1,I2,m2,b2,I3,m3,b3,I4,m4,b4,I5,m5,b5):
            return(I1*np.exp(-((x-m1)**2)/(2*(b1**2)))/(b1*np.sqrt(2*np.pi))+
               I2*np.exp(-((x-m2)**2)/(2*(b2**2)))/(b2*np.sqrt(2*np.pi))+
               I3*np.exp(-((x-m3)**2)/(2*(b3**2)))/(b3*np.sqrt(2*np.pi))+
               I4*np.exp(-((x-m4)**2)/(2*(b4**2)))/(b4*np.sqrt(2*np.pi))+
               I5*np.exp(-((x-m5)**2)/(2*(b5**2)))/(b5*np.sqrt(2*np.pi))
              )
        def expT2(x,I1,m1,b1,I2,m2,b2,I3,m3,b3,I4,m4,b4):
            return(I1*np.exp(-((x-m1)**2)/(2*(b1**2)))/(b1*np.sqrt(2*np.pi))+
               I2*np.exp(-((x-m2)**2)/(2*(b2**2)))/(b2*np.sqrt(2*np.pi))+
               I3*np.exp(-((x-m3)**2)/(2*(b3**2)))/(b3*np.sqrt(2*np.pi))+
               I4*np.exp(-((x-m4)**2)/(2*(b4**2)))/(b4*np.sqrt(2*np.pi))
              )
            
        def deconvol1(df1,bounds,nombre_peak):
            #row=int(len(df.columns)/4)
            row=3
            fig, axs = plt.subplots(nrows=4, ncols=row, figsize=(25, 15))
            for ax, i in zip(axs.flat, range(row2)):
                x=df1[df1.columns[0]]
                y=df1[df1.columns[2*i+1]]/max(df1[df1.columns[2*i+1]]); # 
                #y = (y - np.min(y)) / (np.max(y) - np.min(y)) # pour normaliser les intensités
                y_hat=savgol_filter(y, 11, 2);
                pop1,pcov1=curve_fit(expT,x,y,bounds=bounds)
                ax.plot(x,y,label=df1.columns[2*i]);
                ax.plot(x,y_hat,label='Savitzki-gol')
                ax.plot(x,expT(x,*pop1),label='somme')
                for j in range(nombre_peak):
                    ax.plot(x, expS(x, *pop1[3*j:3*(j+1)]), label=f'{j+1}ème déconvoluée')
                ax.legend()
            st.pyplot(fig)
            return()
        def deconvol3(df1,bounds,nombre_peak):
            #row=int(len(df.columns)/4)
            row=3
            fig, axs = plt.subplots(nrows=4, ncols=row, figsize=(25, 15))
            for ax, i in zip(axs.flat, range(row2)):
                x=df1[df1.columns[0]]
                y=df1[df1.columns[2*i+1]]/max(df1[df1.columns[2*i+1]]); # 
                #y = (y - np.min(y)) / (np.max(y) - np.min(y)) # pour normaliser les intensités
                y_hat=savgol_filter(y, 11, 2);
                pop1,pcov1=curve_fit(expT2,x,y,bounds=bounds)
                ax.plot(x,y,label=df1.columns[2*i]);
                ax.plot(x,y_hat,label='Savitzki-gol')
                ax.plot(x,expT2(x,*pop1),label='somme')
                for j in range(nombre_peak):
                    ax.plot(x, expS(x, *pop1[3*j:3*(j+1)]), label=f'{j+1}ème déconvoluée')
                ax.legend()
            st.pyplot(fig)
            return()      
        #for uploaded_file in uploaded_files:
        if uploaded_files is not None:
           df = pd.read_csv(uploaded_files, delimiter=";") 
           for i in df.columns:
               if (df[i].isnull()[0]==True): # On elimine les colonnes vides
                   del df[i];
           df=df.dropna(axis=0); # On elimine les lignes qui contiennent des na;
           df=df[1:];
           df=df.astype(float)
           st.markdown("## La base de donnée")  
           st.write(df) 
           row=int(len(df.columns)/4)
           row2=int(len(df.columns)/2)

           fig, axs = plt.subplots(nrows=2, ncols=row, figsize=(20, 6))
           st.markdown("## Spectre d'excitation")
           for ax, i in zip(axs.flat, range(row2)):
               x = df[df.columns[0]]
               y = df[df.columns[2*i+1]]
               ax.plot(x,y, label=df.columns[2*i])
               ax.set_xlabel("Longeur d'onde")
               ax.set_ylabel("Intensité")
               ax.legend()
           st.pyplot(fig)
           st.markdown("## Lissage")
           row=int(len(df.columns)/6)
           row2=int(len(df.columns)/2)
           p=[]
           les_peaks=[]
           borne=[]
           fig, axs = plt.subplots(nrows=3, ncols=row, figsize=(20, 10))
           for ax, i in zip(axs.flat, range(row2)):
               bor=[]
               x1=df[df.columns[0]]
               y=df[df.columns[2*i+1]];
               y_hat=savgol_filter(y, 15, 2);
               ax.plot(x1,y, label=df.columns[2*i])
               ax.plot(x1,y_hat,label='Savitzki-gol')
               x = y_hat
               peaks, properties = find_peaks(x, prominence=1, width=1)
               p.append(len(peaks))
               xmin=properties["left_ips"]
               xmax=properties["right_ips"]
               #st.write("moyenne longeur d'onde de : ",df.columns[2*i],list(x1[peaks]),"nombre de peaks : ",len(peaks))
               for j in range(len(properties['left_ips'])):
                   bor.append(list([x1[np.around(properties['left_ips'][j])],x1[np.around(properties['right_ips'][j])]]))
               ax.plot(x1[peaks], x[peaks], "x")
               ax.vlines(x=x1[peaks], ymin=0,ymax = x[peaks], color = "C1")
               ax.set_xlabel("Longeur d'onde")
               ax.set_ylabel("Intensité")
               ax.legend()
               les_peaks.append(list(x1[peaks]))
               les_peak=pd.DataFrame(les_peaks)
               if (len(bor)==5):
                   borne.append(bor)
           st.pyplot(fig)
           st.markdown("## Déconvolution")
           st.markdown("### déconvolution avec 5 gaussiennes ")
           nbr_p=5
           if borne==[]:
               bounds_lower =[0,250,0,0,270,0,0,300,0,0,340,0,0,360,0]
               bounds_upper =[np.inf,270,np.inf,np.inf,300,np.inf,np.inf,340,np.inf,np.inf,360,np.inf,np.inf,400,np.inf]
               bounds = (bounds_lower, bounds_upper)
           else :
               born=borne[0]
               bounds_lower =[0,born[0][0],0,0,born[0][1],0,0,born[2][0],0,0,born[3][0],0,0,born[4][0],0]
               bounds_upper =[np.inf,born[0][1],np.inf,np.inf,born[1][1],np.inf,np.inf,born[2][1],
               np.inf,np.inf,born[3][1],np.inf,np.inf,born[4][1],np.inf]
               bounds = (bounds_lower, bounds_upper)
           st.write(deconvol1(df,bounds,nbr_p))
           st.markdown("## déconvolution avec 4 gaussiennes ")
           nbr_p=4
           if borne==[]:
               bounds_lower =[0,250,0,0,300,0,0,340,0,0,360,0]
               bounds_upper =[np.inf,300,np.inf,np.inf,340,np.inf,np.inf,360,np.inf,np.inf,400,np.inf]
               bounds = (bounds_lower, bounds_upper)
           else :
               born=borne[0]
               bounds_lower =[0,born[0][0],0,0,born[0][1],0,0,born[2][0],0,0,born[3][0],0]
               bounds_upper =[np.inf,born[0][1],np.inf,np.inf,born[1][1],np.inf,np.inf,born[2][1],
               np.inf,np.inf,born[3][1],np.inf]
           bounds = (bounds_lower, bounds_upper)
           st.write(deconvol3(df,bounds,nbr_p))

    if __name__ == "__main__":
        main()
    
    def main():
    
        uploaded_files = st.file_uploader("Choisir les fichiers csv ", accept_multiple_files=True)
        st.image("https://ilm.univ-lyon1.fr//images/slides/CARROUSSEL-15.png")
        def expS(x,I,m,b):
            return(I*np.exp(-((x-m)**2)/(2*(b**2)))/(b*np.sqrt(2*np.pi)))
        def expT(x,I1,m1,b1,I2,m2,b2,I3,m3,b3,I4,m4,b4,I5,m5,b5):
            return(I1*np.exp(-((x-m1)**2)/(2*(b1**2)))/(b1*np.sqrt(2*np.pi))+
               I2*np.exp(-((x-m2)**2)/(2*(b2**2)))/(b2*np.sqrt(2*np.pi))+
               I3*np.exp(-((x-m3)**2)/(2*(b3**2)))/(b3*np.sqrt(2*np.pi))+
               I4*np.exp(-((x-m4)**2)/(2*(b4**2)))/(b4*np.sqrt(2*np.pi))+
               I5*np.exp(-((x-m5)**2)/(2*(b5**2)))/(b5*np.sqrt(2*np.pi))
              )
        def expT2(x,I1,m1,b1,I2,m2,b2,I3,m3,b3,I4,m4,b4):
            return(I1*np.exp(-((x-m1)**2)/(2*(b1**2)))/(b1*np.sqrt(2*np.pi))+
               I2*np.exp(-((x-m2)**2)/(2*(b2**2)))/(b2*np.sqrt(2*np.pi))+
               I3*np.exp(-((x-m3)**2)/(2*(b3**2)))/(b3*np.sqrt(2*np.pi))+
               I4*np.exp(-((x-m4)**2)/(2*(b4**2)))/(b4*np.sqrt(2*np.pi))
              )
        def interval(bor):
            if bor==[]:
               bounds_lower =[0,250,0,0,300,0,0,340,0,0,360,0]
               bounds_upper =[np.inf,300,np.inf,np.inf,340,np.inf,np.inf,360,np.inf,np.inf,400,np.inf]
               bounds = (bounds_lower, bounds_upper)
            else :
               born=bor[0]
               bounds_lower =[0,born[0][0],0,0,born[0][1],0,0,born[2][0],0,0,born[3][0],0]
               bounds_upper =[np.inf,born[0][1],np.inf,np.inf,born[1][1],np.inf,np.inf,born[2][1],
               np.inf,np.inf,born[3][1],np.inf]
            bounds = (bounds_lower, bounds_upper)  
            return(bounds)
        
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file, delimiter=";") 
            for i in df.columns:
                if (df[i].isnull()[0]==True): # On elimine les colonnes vides
                    del df[i];
            df=df.dropna(axis=0); # On elimine les lignes qui contiennent des na;
            df=df[1:];
            #df=df[1:list(np.where(df[df.columns[0]]=='400')[0])[0]]; # voir colonne ci-dessous pour les détails de cette ligne.
            df=df.astype(float)
            row=int(len(df.columns)/6)
            row2=int(len(df.columns)/2)
            p=[]
            les_peaks=[]
            borne=[]
            for i in range(row2):
                bor=[]
                x1=df[df.columns[0]]
                y=df[df.columns[2*i+1]];
                y_hat=savgol_filter(y, 11, 2);
                x = y_hat
                peaks, properties = find_peaks(x, prominence=1, width=1)
                p.append(len(peaks))
                for j in range(len(properties['left_ips'])):
                    bor.append(list([x1[np.around(properties['left_ips'][j])],x1[np.around(properties['right_ips'][j])]]))
                les_peaks.append(list(x1[peaks]))
                les_peak=pd.DataFrame(les_peaks)
                if (len(bor)==4):
                   borne.append(bor)
            bounds=interval(borne)
            df_dp=pd.DataFrame(columns = ['Fichier','Type','A1','M1','E1','C1','A2',
                                  'M2','E2','C2','A3','M3','E3','C3','A4','M4','E4','C4'])
            for i in range(int(len(df.columns)/2)):
                x=df[df.columns[0]]
                y=df[df.columns[2*i+1]]/max(df[df.columns[2*i+1]]); # 
                #y = (y - np.min(y)) / (np.max(y) - np.min(y)) # pour normaliser les intensités
                y_hat=savgol_filter(y, 11, 2);
                pop1,pcov1=curve_fit(expT2,x,y,bounds=bounds)
                c1=spi.simps(expS(x,*pop1[0:3]),x)/spi.simps(expT2(x,*pop1),x);
                c2=spi.simps(expS(x,*pop1[3:6]),x)/spi.simps(expT2(x,*pop1),x);
                c3=spi.simps(expS(x,*pop1[6:9]),x)/spi.simps(expT2(x,*pop1),x);
                c4=spi.simps(expS(x,*pop1[9:12]),x)/spi.simps(expT2(x,*pop1),x);
                df_dp=df_dp.append({'Fichier':uploaded_file.name,'Type':df.columns[2*i], 
                                      'A1':pop1[0],'M1':pop1[1],'E1':pop1[2],'C1':c1,'A2':pop1[3],
                                      'M2':pop1[4],'E2':pop1[5],'C2':c2,'A3':pop1[6],'M3':pop1[7],
                                      'E3':pop1[8],'C3':c3,'A4':pop1[9],'M4':pop1[10],'E4':pop1[11],
                                      'C4':c4},ignore_index=True)
            st.write("Création de la base de donnée")
            st.write(df_dp)
            

            
            
    
    
    if __name__ == "__main__":
        main()

def code_python():
    st.markdown("# Code Lissage et Déconvolution ")
    code='''#!/usr/bin/env python
# coding: utf-8

# 
# # Bibliothéque 

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt;
from tkinter import filedialog
from tkinter import *
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit;
import math
import scipy.integrate as spi
import csv


# # Préparation des données 

# In[2]:


def browseFiles():
	filename = filedialog.askopenfilename(initialdir = "http://localhost:8888/tree/Stage",
										title = "Select a File",
										filetypes = (("Csv files",
														"*.csv*"),
													("all files",
														"*.*")))
	return(filename)


# In[4]:


VAR=browseFiles()


# ## Spectre d'excitation

# In[3]:


def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter


# In[5]:


delimit=find_delimiter(VAR)


# In[6]:


df=pd.read_csv(VAR,delimiter=delimit)
for i in df.columns:
        if (df[i].isnull()[0]==True): # On elimine les colonnes vides
            del df[i];
df=df.dropna(axis=0); # On elimine les lignes qui contiennent des na;
df=df[1:];
df=df.astype(float)
df


# In[7]:


row=int(len(df.columns)/4)
row2=int(len(df.columns)/2)

fig, axs = plt.subplots(nrows=2, ncols=row, figsize=(20, 6))
for ax, i in zip(axs.flat, range(row2)):
    x = df[df.columns[0]]
    y = df[df.columns[2*i+1]]
    ax.plot(x,y, label=df.columns[2*i])
    ax.set_xlabel("Longeur d'onde")
    ax.set_ylabel("Intensité")
    ax.legend()
plt.show()


# ## Lissage 

# Lissage avec algorithme de Savitz-golay
# La fonction savgol_filter prend en paramètre:
# 
# -y ou x : il s'agit de la donnée à filtrer.
# 
# - La longueur de la fenetre de lissage.
# 
# - Le degré du polynome de lissage. 
# Elle renvoie: La donnée filtrée.

# In[8]:


row=int(len(df.columns)/6)
row2=int(len(df.columns)/2)
p=[]
les_peaks=[]
borne=[]
fig, axs = plt.subplots(nrows=3, ncols=row, figsize=(20, 10))
for ax, i in zip(axs.flat, range(row2)):
    bor=[]
    x1=df[df.columns[0]]
    y=df[df.columns[2*i+1]];
    y_hat=savgol_filter(y, 15, 2);
    ax.plot(x1,y, label=df.columns[2*i])
    ax.plot(x1,y_hat,label='Savitzki-gol')
    x = y_hat
    peaks, properties = find_peaks(x, prominence=1, width=1)
    p.append(len(peaks))
    xmin=properties["left_ips"]
    xmax=properties["right_ips"]
    print("moyenne longeur d'onde de : ",df.columns[2*i],list(x1[peaks]),"nombre de peaks : ",len(peaks))
    for j in range(len(properties['left_ips'])):
        bor.append(list([x1[np.around(properties['left_ips'][j])],x1[np.around(properties['right_ips'][j])]]))
    ax.plot(x1[peaks], x[peaks], "x")
    ax.vlines(x=x1[peaks], ymin=0,ymax = x[peaks], color = "C1")
    #ax.hlines(y=properties["widths"], xmin=properties['left_ips'],xmax=properties['right_ips'], color = "C1")
    ax.set_xlabel("Longeur d'onde")
    ax.set_ylabel("Intensité")
    ax.legend()
    les_peaks.append(list(x1[peaks]))
    les_peak=pd.DataFrame(les_peaks)
    if (len(bor)==5):
        borne.append(bor)
plt.show()


# In[9]:


borne


# ## Déconvolution 

# In[9]:


def expS(x,I,m,b):
    return(I*np.exp(-((x-m)**2)/(2*(b**2)))/(b*np.sqrt(2*np.pi)))


# In[10]:


def expT(x,I1,m1,b1,I2,m2,b2,I3,m3,b3,I4,m4,b4,I5,m5,b5):
        return(I1*np.exp(-((x-m1)**2)/(2*(b1**2)))/(b1*np.sqrt(2*np.pi))+
               I2*np.exp(-((x-m2)**2)/(2*(b2**2)))/(b2*np.sqrt(2*np.pi))+
               I3*np.exp(-((x-m3)**2)/(2*(b3**2)))/(b3*np.sqrt(2*np.pi))+
               I4*np.exp(-((x-m4)**2)/(2*(b4**2)))/(b4*np.sqrt(2*np.pi))+
               I5*np.exp(-((x-m5)**2)/(2*(b5**2)))/(b5*np.sqrt(2*np.pi))
              )
def expT2(x,I1,m1,b1,I2,m2,b2,I3,m3,b3,I4,m4,b4):
        return(I1*np.exp(-((x-m1)**2)/(2*(b1**2)))/(b1*np.sqrt(2*np.pi))+
               I2*np.exp(-((x-m2)**2)/(2*(b2**2)))/(b2*np.sqrt(2*np.pi))+
               I3*np.exp(-((x-m3)**2)/(2*(b3**2)))/(b3*np.sqrt(2*np.pi))+
               I4*np.exp(-((x-m4)**2)/(2*(b4**2)))/(b4*np.sqrt(2*np.pi))
              )


# In[11]:


def deconvol1(df1,bounds,nombre_peak):
    #row=int(len(df.columns)/4)
    row=3
    fig, axs = plt.subplots(nrows=4, ncols=row, figsize=(25, 15))
    for ax, i in zip(axs.flat, range(row2)):
            x=df1[df1.columns[0]]
            y=df1[df1.columns[2*i+1]]/max(df1[df1.columns[2*i+1]]); # 
            #y = (y - np.min(y)) / (np.max(y) - np.min(y)) # pour normaliser les intensités
            y_hat=savgol_filter(y, 11, 2);
            pop1,pcov1=curve_fit(expT,x,y,bounds=bounds)
            ax.plot(x,y,label=df1.columns[2*i]);
            ax.plot(x,y_hat,label='Savitzki-gol')
            ax.plot(x,expT(x,*pop1),label='somme')
            for j in range(nombre_peak):
                ax.plot(x, expS(x, *pop1[3*j:3*(j+1)]), label=f'{j+1}ème déconvoluée')
            ax.legend()
    plt.show()
    return()
def deconvol3(df1,bounds,nombre_peak):
    #row=int(len(df.columns)/4)
    row=3
    fig, axs = plt.subplots(nrows=4, ncols=row, figsize=(25, 15))
    for ax, i in zip(axs.flat, range(row2)):
            x=df1[df1.columns[0]]
            y=df1[df1.columns[2*i+1]]/max(df1[df1.columns[2*i+1]]); # 
            #y = (y - np.min(y)) / (np.max(y) - np.min(y)) # pour normaliser les intensités
            y_hat=savgol_filter(y, 11, 2);
            pop1,pcov1=curve_fit(expT2,x,y,bounds=bounds)
            ax.plot(x,y,label=df1.columns[2*i]);
            ax.plot(x,y_hat,label='Savitzki-gol')
            ax.plot(x,expT2(x,*pop1),label='somme')
            for j in range(nombre_peak):
                ax.plot(x, expS(x, *pop1[3*j:3*(j+1)]), label=f'{j+1}ème déconvoluée')
            ax.legend()
    plt.show()
    return()          


# In[12]:


print("déconvolution avec 5 gaussiennes ")
nbr_p=5
if borne==[]:
    bounds_lower =[0,250,0,0,270,0,0,300,0,0,340,0,0,360,0]
    bounds_upper =[np.inf,270,np.inf,np.inf,300,np.inf,np.inf,340,np.inf,np.inf,360,np.inf,np.inf,400,np.inf]
    bounds = (bounds_lower, bounds_upper)
else :
    born=borne[0]
    bounds_lower =[0,born[0][0],0,0,born[0][1],0,0,born[2][0],0,0,born[3][0],0,0,born[4][0],0]
    bounds_upper =[np.inf,born[0][1],np.inf,np.inf,born[1][1],np.inf,np.inf,born[2][1],
               np.inf,np.inf,born[3][1],np.inf,np.inf,born[4][1],np.inf]
    bounds = (bounds_lower, bounds_upper)
deconvol1(df,bounds,nbr_p)  
print("déconvolution avec 4 gaussiennes ")
nbr_p=4
if borne==[]:
    bounds_lower =[0,250,0,0,300,0,0,340,0,0,360,0]
    bounds_upper =[np.inf,300,np.inf,np.inf,340,np.inf,np.inf,360,np.inf,np.inf,400,np.inf]
    bounds = (bounds_lower, bounds_upper)
else :
    born=borne[0]
    bounds_lower =[0,born[0][0],0,0,born[0][1],0,0,born[2][0],0,0,born[3][0],0]
    bounds_upper =[np.inf,born[0][1],np.inf,np.inf,born[1][1],np.inf,np.inf,born[2][1],
               np.inf,np.inf,born[3][1],np.inf]
    bounds = (bounds_lower, bounds_upper)
deconvol3(df,bounds,nbr_p)


# In[ ]:





# # Construction de la base de donnée

# In[13]:


def browseFiles2():
	filename = filedialog.askopenfilenames(initialdir = "http://localhost:8888/tree/Stage",
										title = "Select a File",
										filetypes = (("Csv files",
														"*.csv*"),
													("all files",
														"*.*")))
	return(filename)


# In[22]:


VARS=browseFiles2()


# In[23]:


df_dp=pd.DataFrame();


# In[24]:


def peak(df):
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
    frequent=max(set(p), key = p.count)
    nombre_peak=set(p)
    nombre_peak=list(nombre_peak)
    les_peaks=np.transpose(les_peak)
    #m_min = [np.nanmin(les_peak[pk]) for pk in range(len(les_peak.columns))]
    #m_max =[np.nanmax(les_peak[pk]) for pk in range(len(les_peak.columns))]
    return(nombre_peak,frequent)       
for VAR in VARS:
    delimit=find_delimiter(VAR)
    df=pd.read_csv(VAR,delimiter=delimit)
    for i in df.columns:
        if (df[i].isnull()[0]==True): # On elimine les colonnes vides
            del df[i];
    df=df.dropna(axis=0); # On elimine les lignes qui contiennent des na;
    df=df[1:];
    df=df.astype(float)
    p,f=peak(df)
    print("nombre de peak =  ",p," le plus frequent est : ",f)


# In[25]:


def interval(bor):
    if bor==[]:
            bounds_lower =[0,250,0,0,300,0,0,340,0,0,360,0]
            bounds_upper =[np.inf,300,np.inf,np.inf,340,np.inf,np.inf,360,np.inf,np.inf,400,np.inf]
            bounds = (bounds_lower, bounds_upper)
    else :
            born=bor[0]
            bounds_lower =[0,born[0][0],0,0,born[0][1],0,0,born[2][0],0,0,born[3][0],0]
            bounds_upper =[np.inf,born[0][1],np.inf,np.inf,born[1][1],np.inf,np.inf,born[2][1],
               np.inf,np.inf,born[3][1],np.inf]
            bounds = (bounds_lower, bounds_upper)  
    return(bounds)


# In[26]:


def calcul_para(VARS):
    df_dp=pd.DataFrame(columns = ['Fichier','Type','A1','M1','E1','C1','A2',
                                  'M2','E2','C2','A3','M3','E3','C3','A4','M4','E4','C4']);
    for VAR in VARS:
        delimit=find_delimiter(VAR)
        df=pd.read_csv(VAR,delimiter=delimit)
        for i in df.columns:
            if (df[i].isnull()[0]==True): # On elimine les colonnes vides
                del df[i];
        df=df.dropna(axis=0); # On elimine les lignes qui contiennent des na;
        df=df[1:];
        #df=df[1:list(np.where(df[df.columns[0]]=='400')[0])[0]]; # voir colonne ci-dessous pour les détails de cette ligne.
        df=df.astype(float)
        row=int(len(df.columns)/6)
        row2=int(len(df.columns)/2)
        p=[]
        les_peaks=[]
        borne=[]
        for i in range(row2):
            bor=[]
            x1=df[df.columns[0]]
            y=df[df.columns[2*i+1]];
            y_hat=savgol_filter(y, 11, 2);
            x = y_hat
            peaks, properties = find_peaks(x, prominence=1, width=1)
            p.append(len(peaks))
            for j in range(len(properties['left_ips'])):
                bor.append(list([x1[np.around(properties['left_ips'][j])],x1[np.around(properties['right_ips'][j])]]))
            les_peaks.append(list(x1[peaks]))
            les_peak=pd.DataFrame(les_peaks)
            if (len(bor)==4):
                borne.append(bor)
        bounds=interval(borne)
        for i in range(int(len(df.columns)/2)):
                x=df[df.columns[0]]
                y=df[df.columns[2*i+1]]/max(df[df.columns[2*i+1]]); # 
                #y = (y - np.min(y)) / (np.max(y) - np.min(y)) # pour normaliser les intensités
                y_hat=savgol_filter(y, 11, 2);
                pop1,pcov1=curve_fit(expT2,x,y,bounds=bounds)
                c1=spi.simps(expS(x,*pop1[0:3]),x)/spi.simps(expT2(x,*pop1),x);
                c2=spi.simps(expS(x,*pop1[3:6]),x)/spi.simps(expT2(x,*pop1),x);
                c3=spi.simps(expS(x,*pop1[6:9]),x)/spi.simps(expT2(x,*pop1),x);
                c4=spi.simps(expS(x,*pop1[9:12]),x)/spi.simps(expT2(x,*pop1),x);
                df_dp=df_dp.append({'Fichier':VAR.split('/')[-1],'Type':df.columns[2*i], 
                                      'A1':pop1[0],'M1':pop1[1],'E1':pop1[2],'C1':c1,'A2':pop1[3],
                                      'M2':pop1[4],'E2':pop1[5],'C2':c2,'A3':pop1[6],'M3':pop1[7],
                                      'E3':pop1[8],'C3':c3,'A4':pop1[9],'M4':pop1[10],'E4':pop1[11],
                                      'C4':c4},ignore_index=True)
                              
                               
    return(df_dp)


# In[27]:


dp=calcul_para(VARS)


# ## Calcul Ratio 

# In[29]:


dp


# In[ ]:





# In[108]:


import re
T=[]
for i in range(len(dp)):
    t=dp['Fichier'][i]
    if t.find('AMPA')!=-1:
        dp['Fichier'][i]="A-"+dp['Fichier'][i]
    if t.find('DTPA')!=-1:
        dp['Fichier'][i]="D-"+dp['Fichier'][i]
    if t.find('EDTA')!=-1:
        dp['Fichier'][i]="E-"+dp['Fichier'][i]
    if t.find('GLY')!=-1:
        dp['Fichier'][i]="G-"+dp['Fichier'][i]
    liste=re.split('; |-', t)
    liste2=re.findall(r'\b\d+\b', t)
    T.append(t)
    print(t)
    


# In[109]:


dp['Fichier']=T


# In[110]:


dp['A']=0
dp['D']=0
dp['E']=0
dp['G']=0


# In[ ]:





# In[112]:


for j in range(len(dp['Fichier'])):
    if dp['Fichier'][j][0]=='A':
        dp['A'][j]=1
    elif dp['Fichier'][j][0]=='D':
        dp['D'][j]=1
    elif dp['Fichier'][j][0]=='E':
        dp['E'][j]=1
    else :
        dp['G'][j]=1


# In[113]:


dp['clf']=0


# In[114]:


for j in range(len(dp['Fichier'])):
    if dp['Fichier'][j][0]=='A':
        dp['clf'][j]='A'
    elif dp['Fichier'][j][0]=='D':
        dp['clf'][j]='D'
    elif dp['Fichier'][j][0]=='E':
        dp['clf'][j]='E'
    else :
        dp['clf'][j]='G'


# In[ ]:





# In[ ]:


dp=dp.drop(columns=['A','D','E','G']


# In[116]:


dp


# In[160]:


Base1.to_csv('Base.csv', index=False)


# In[ ]:





# ## Ratio 2 polluants 

# In[95]:


VARS2=browseFiles2()


# In[ ]:





# In[96]:


dp2=calcul_para(VARS2)


# In[94]:


dp2


# In[15]:





# ## Calcul concentration pour chaque ajouts 

# In[16]:


# Serie 1
v_total=3.3
unk=2.8
es=9
std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1]
std=np.array(std)
ss=100
purity=50
chemicals=573.2
npol1=(unk*es+std*ss)*purity/(chemicals*100)
npol2=[(unk*es)*purity/(chemicals*100)]*len(std)
npol2=np.array(npol2)


# Serie 2
unk=2.8
es=1
std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1]
std=np.array(std)
ss=100
purity=50
chemicals=299.1
npol21=(unk*es+std*ss)*purity/(chemicals*100)
npol22=[(unk*es)*purity/(chemicals*100)]*len(std)

# Serie 3
unk=2.8
es=9
std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1]
std=np.array(std)
ss=100
purity=50
chemicals=573.2
npol3=(unk*es+std*ss)*purity/(chemicals*100)

# Serie 4
unk=2.8
es=1
std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1]
std=np.array(std)
ss=100
purity=50
chemicals=299.1
npol4=(unk*es+std*ss)*purity/(chemicals*100)


# In[21]:


c1=npol1/3.3*0.001
c2=npol2/3.3*0.001


# In[25]:


c1,c2


# In[46]:


for i in range(4):
    print(dp2['Fichier'][i].find('S{}'.format(i)))


# In[45]:


['S{}'.format(i) for i in range(1, 5)]

'''
    st.code(code,language="python") 
    
    st.markdown("# code Identification")  
    code=''' import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt;
import seaborn as sns
from tkinter import filedialog
from tkinter import *
import time
import pickle
from pycaret.classification import *
from scipy.stats import expon, poisson, gamma, lognorm, weibull_min, kstest,norm
import scipy
import scipy.stats
from mlxtend.plotting import plot_pca_correlation_graph
import scipy.stats as stats
#from pycaret.regression import *
import csv
from sklearn.preprocessing import LabelEncoder # nous permet de faire l'encodage , avec ordinalencoder fait la même mais avec plusieurs variable encoder


## fonction qui nous permet de charher des fichiers 

def browseFiles():
	filename = filedialog.askopenfilename(initialdir = "Z:\1_Data\1_Experiments\1_FENNEC\2_Stagiaires\2022_Alvin\7 Samples\ATMP_DTPMP",
										title = "Select a File",
										filetypes = (("Csv files",
														"*.csv*"),
													("all files",
														"*.*")))
	return(filename)

## fonction qui nous permet de recupérer le delimiter du fichier

def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter

VAR=browseFiles()

## charger les données 

concatenated_df=pd.read_csv(VAR,index_col=0,sep=',')
df=concatenated_df

n=len(df.columns)
df=df[df[df.columns[:(n-1)]].duplicated()==False]
df.index=range(0,len(df))

## donner le target et les variables numériques

X=df[df.columns[:(n-1)]]# on prend les variables numériques 
y=df[df.columns[-1]] # le target

# Etude des variables 

print(X.shape,y.shape) # la dimension
print("taille des données : ", df.shape,"\n taille de X:" ,X.shape,"\n taille de y : ",y.shape ) 
pour_A=np.sum(y=='A')/len(y)
pour_D=np.sum(y=='D')/len(y)
pour_G=np.sum(y=='G')/len(y)
pour_A_G=np.sum(y=='A+G')/len(y)
pour_D_G=np.sum(y=='D+G')/len(y)
pour_A_D=np.sum(y=='A+D')/len(y)
print(" A :",np.sum(y=='A'),"D :",np.sum(y=='D'),"G :",np.sum(y=='G'))
print("pourcentage d'exemple A :",pour_A*100,"%")
print("pourcentage d'exemple D :",pour_D*100,"%")
print("pourcentage d'exemple G :",pour_G*100,"%")
print("pourcentage d'exemple A+G :",pour_A_G*100,"%")
print("pourcentage d'exemple D+G :",pour_D_G*100,"%")
print("pourcentage d'exemple A+D :",pour_A_D*100,"%")

colors = ['#06344d', '#00b2ff']
sns.set(palette = colors, font = 'Serif', style = 'white', 
        rc = {'axes.facecolor':'#f1f1f1', 'figure.facecolor':'#f1f1f1'})

fig = plt.figure(figsize = (10, 6))
ax = sns.countplot(x = 'clf', data = df)
for i in ax.patches:
    ax.text(x = i.get_x() + i.get_width()/2, y = i.get_height()/7, 
            s = f"{np.round(i.get_height()/len(df)*100, 0)}%", 
            ha = 'center', size = 50, weight = 'bold', rotation = 90, color = 'white')
plt.title("histogramme des polluants", size = 20, weight = 'bold')
plt.xlabel('classes', weight = 'bold')
plt.ylabel('observation', weight = 'bold');

## histogramme des variables numériques

numerical = [var for var in df.columns ]
features = numerical
colors = ['blue']
df[features[:(n-1)]].hist(figsize=(10, 6), color=colors, alpha=0.7)
plt.show()

# matrice de corrélation 
plt.figure(figsize=(12,6))
corr_matrix = df[features[:(n-1)]].corr()
sns.heatmap(corr_matrix,annot=True)
plt.show()

# grâce à sklearn on peut importer standardscaler .
ss=StandardScaler()# enleve la moyenne et divise par l'ecartype
ss.fit(X)
X_norm=ss.transform(X)# tranform X 
## PCA
figure, correlation_matrix = plot_pca_correlation_graph(X_norm, features[:(n-1)], dimensions=(1, 2),figure_axis_size=8)
figure, correlation_matrix = plot_pca_correlation_graph(X_norm, features[:(n-1)],dimensions=(1, 3),figure_axis_size=8)
figure, correlation_matrix = plot_pca_correlation_graph(X_norm, features[:(n-1)],dimensions=(1, 4),figure_axis_size=8)

## Pycaret

# Ici on spécifit les données utilisés , si on doit les normaliser ou pas , utiliser acp ou pas , donner le pourcentage train ...
# ici on test avec données sans normalisés 
colonne=features[:(n-1)]+['clf']
d=0.75
setup_data = setup(data = df[colonne],target = 'clf',
                   train_size =d,categorical_features =None,
                   normalize = False,normalize_method = None,remove_multicollinearity =False
                   ,multicollinearity_threshold =None,pca =False, pca_method =None,
                   pca_components = None,numeric_features =features[:(n-1)], session_id=123, log_experiment=False, experiment_name='wine_q1')
r=compare_models()
final_model1 = create_model(r)
plot_model(final_model1,plot='auc')
plot_model(final_model1,plot='class_report')
plot_model(final_model1,plot='confusion_matrix')
tuned_model = tune_model(final_model1);# optimiser le modéle
plot_model(final_model1 , plot='boundary')
plot_model(estimator = tuned_model, plot = 'feature')
plot_model(final_model1, plot='pr')
plot_model(estimator = final_model1, plot = 'learning')
#predict=predict_model(final_model1)



# normaliser 
colonne=features[:(n-1)]+['clf']
setup_data = setup(data = df[colonne],target = 'clf',
                   train_size =0.75,categorical_features =None,
                   normalize = True,normalize_method = 'zscore',remove_multicollinearity = True
                   ,multicollinearity_threshold = 0.8,pca =False, pca_method =None,
                   pca_components =None,numeric_features =features[:(n-1)], session_id=123, log_experiment=False, experiment_name='wine_q1')
r2=compare_models()
final_model2 = create_model(r2)
plot_model(final_model2,plot='auc')
plot_model(final_model2,plot='class_report')
plot_model(final_model2,plot='confusion_matrix')
tuned_model = tune_model(final_model2);# optimiser le modéle
plot_model(final_model2 , plot='boundary')
plot_model(estimator = tuned_model, plot = 'feature')
plot_model(final_model2, plot='pr')
plot_model(estimator = final_model2, plot = 'learning')


colonne=features[:(n-1)]+['clf']
setup_data = setup(data = df[colonne],target = 'clf',
                   train_size =0.75,categorical_features =None,
                   normalize = True,normalize_method = 'zscore',remove_multicollinearity = True
                   ,multicollinearity_threshold = 0.8,pca =True, pca_method ='linear',
                   pca_components = 0.90,numeric_features =features[:(n-1)])


# les méthodes pour pca:
# "linear" : utilise la décomposition en valeurs singulières.
#"kernel" : réduction de la dimensionnalité grâce à l’utilisation du noyau RBF.
#"incremental" : similaire à « linéaire », mais plus efficace pour les grands ensembles de données
r3=compare_models()
final_model = create_model(r3)
plot_model(final_model,plot='auc')
plot_model(final_model,plot='class_report')
plot_model(final_model,plot='confusion_matrix')
tuned_model = tune_model(final_model);
plot_model(tuned_model , plot='boundary')

evaluate_model(final_model1)#Cette fonction affiche une interface utilisateur pour analyser les performances
final_predictions = predict_model(final_model1)
save_model(final_model1, model_name = 'extra_tree_model')
loaded_bestmodel = load_model('extra_tree_model')
from sklearn import set_config
set_config(display='diagram')
X_train = get_config('X_train')
X_train.head()
 '''
    st.code(code , language="python") 
    
    st.markdown("# code Image 3D")         
    code=''' 
from tkinter import filedialog
from tkinter import *
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
from scipy.interpolate import griddata
import tensorflow as tf
import keras.preprocessing.image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
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

# Charger les fichiers csv

def browseFiles2():
	filename = filedialog.askopenfilenames(initialdir = "http://localhost:8888/tree/Stage",
										title = "Select a File",
										filetypes = (("Csv files",
														"*.csv*"),
													("all files",
														"*.*")))
	return(filename)
VAR=browseFiles2()

## fonction délimiter
import csv
def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter
AD=[browseFiles2() for i in range(7)] # les csv A+D
Am=browseFiles2() ## les csv AMPA
D=browseFiles2() # les csv DTPMA
n=4
E=[browseFiles2() for i in range(n)] # Les csv EDTA
E=E[0]+E[1]+E[2]+E[3]
VARS=[AD,D,E,Am]

## Tracer les images 3D

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
        fig1.savefig('Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/'+V.split('/')[-1]+'.png')
        f=r'Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/'+V.split('/')[-1]+'.png'
        fichier.append(f)
        labels.append(col[VAR])
    plt.show()
    
## Augmenter les données 
from PIL import Image
labels1=[]
fichier1=[]
for j in range(len(fichier)):
    im = Image.open(fichier[j])
    im = im.rotate(180)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+labels[j]+"rotation"+fichier[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+labels[j]+"rotation"+fichier[j].split('/')[-1]
    fichier1.append(f)
    labels1.append(labels[j])
    plt.imshow(im)
    plt.show()
labels2=[]
fichier2=[]
for j in range(len(fichier)):
    im = Image.open(fichier[j])
    im = im.convert("L")  # Grayscale
    plt.imshow(im)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+labels[j]+"couleur"+fichier[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+labels[j]+"couleur"+fichier[j].split('/')[-1]
    fichier2.append(f)
    labels2.append(labels[j])
base=np.transpose(pd.DataFrame([fichier,labels]))
base1=np.transpose(pd.DataFrame([fichier1,labels1]))
base2=np.transpose(pd.DataFrame([fichier2,labels2]))
base=pd.concat([base,base1,base2],axis=0)
base.columns=['image','labels']
from PIL import Image, ImageOps
from PIL import Image, ImageFilter
labels3=[]
fichier3=[]
t=base[base['labels']=='D']

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(30)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_30"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_30"+t['image'].values[j].split('/')[-1]
    fichier3.append(f)
    labels3.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(60)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_60"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_60"+t['image'].values[j].split('/')[-1]
    fichier3.append(f)
    labels3.append(t['labels'].values[j])
    plt.show()
for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(130)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_130"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_130"+t['image'].values[j].split('/')[-1]
    fichier3.append(f)
    labels3.append(t['labels'].values[j])
    plt.show()


for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im.putalpha(128)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_60_col"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_60_col"+t['image'].values[j].split('/')[-1]
    fichier3.append(f)
    labels3.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im= im.convert("L")
    im= im.filter(ImageFilter.FIND_EDGES)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_bruit"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_D_bruit"+t['image'].values[j].split('/')[-1]
    fichier3.append(f)
    labels3.append(t['labels'].values[j])
    plt.show()
    
t=base[base['labels']=='A']
fichier4=[]
labels4=[]
for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(30)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_30"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_30"+t['image'].values[j].split('/')[-1]
    fichier4.append(f)
    labels4.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(60)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_60"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_60"+t['image'].values[j].split('/')[-1]
    fichier4.append(f)
    labels4.append(t['labels'].values[j])
    plt.show()
for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(130)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_130"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_130"+t['image'].values[j].split('/')[-1]
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
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_bruit"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_A_bruit"+t['image'].values[j].split('/')[-1]
    fichier4.append(f)
    labels4.append(t['labels'].values[j])
    plt.show()
    
    
t=base[base['labels']=='E']
fichier5=[]
labels5=[]
for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(30)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_30"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_30"+t['image'].values[j].split('/')[-1]
    fichier5.append(f)
    labels5.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(60)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_60"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_60"+t['image'].values[j].split('/')[-1]
    fichier5.append(f)
    labels5.append(t['labels'].values[j])
    plt.show()
    
for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im = im.rotate(130)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_130"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_130"+t['image'].values[j].split('/')[-1]
    fichier5.append(f)
    labels5.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im.putalpha(128)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_60_col"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_60_col"+t['image'].values[j].split('/')[-1]
    fichier5.append(f)
    labels5.append(t['labels'].values[j])
    plt.show()

for j in range(len(t)):
    im = Image.open(t['image'].values[j])
    im= im.convert("L")
    im= im.filter(ImageFilter.FIND_EDGES)
    im.save("Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_bruit"+t['image'].values[j].split('/')[-1])
    f="Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/image_3D/base/"+t['labels'].values[j]+"couleur_E_bruit"+t['image'].values[j].split('/')[-1]
    fichier5.append(f)
    labels5.append(t['labels'].values[j])
    plt.show()
## Creation de la base de données
base=np.transpose(pd.DataFrame([fichier,labels]))
base1=np.transpose(pd.DataFrame([fichier1,labels1]))
base2=np.transpose(pd.DataFrame([fichier2,labels2]))
base3=np.transpose(pd.DataFrame([fichier3,labels3]))
base4=np.transpose(pd.DataFrame([fichier4,labels4]))
base5=np.transpose(pd.DataFrame([fichier5,labels5]))
base=pd.concat([base,base1,base2,base3,base4,base5],axis=0)
base.columns=['image','labels']
base['labels'].value_counts().plot.bar()
base=base[base.duplicated()!=True]

## Séparer la base train et de validation

train_df, validate_df = train_test_split(base, test_size=0.3)
train_df = train_df.reset_index()
validate_df = validate_df.reset_index()
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

## Creation modéle pré_entrainer 

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

## graphe des performances
def plot_model_history(model_history, acc ='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(range(1,len(model_history.history[acc])+1),model_history.history[acc])
    axs[0].plot(range(1,len(model_history.history[val_acc])+1),model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    #axs[0].set_xticks(np.arange(1,len(model_history.history[acc])+1),len(model_history.history[acc])/10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    #axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
plot_model_history(history)

## Prédiction des images 
from tensorflow.keras.preprocessing import image
from PIL import Image
food_list=base['labels'].unique()
def predict_class(model, images, show = True):
    for img in images:
        img = image.load_img(img, target_size=(224, 224))
        img = image.img_to_array(img)                    
        img = np.expand_dims(img, axis=0)         
        img /= 255.                                      

        pred = model.predict(img)
        index = np.argmax(pred)
        food_list.sort()
        pred_value = food_list[index]
        if show:
            plt.imshow(img[0])                           
            plt.axis('off')
            plt.title(pred_value)
            plt.show()
        print(pred,index,pred_value)


## Enregistrer le model 
model.save('Z:/1_Data/1_Experiments/1_FENNEC/2_Stagiaires/2023_Faty_M2/csv 3D/model_final3.pkl')'''
    st.code(code,language="python")
    
    st.markdown("# code Quantification")
    code='''#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

#  - $C_{HD}$ :  la concentration obtenue dans la serie 1
#  - $C_{HA}$ :  concentration obtenue dans la serie 2
#  - $C_{DA}$ :  concentration obtenue dans la serie 3
# -  $C_{AD}$ :  concentration obtenue dans la serie 4
#  - $C_D^0$ : concentration initiale du polluant 1 
#  - $C_A^0$ : concentration initiale du polluant 2

# - $ C_{HD} = C_D + K_{A-D}C_A $     =>  Serie 1 : mélange dans standard 1 (D)
# - $ C_{HA}= C_A + K_{D-A}C_D $      =>  Serie 2 : mélange dans standard 2 (A)
# - $ C_{DA} = K_{D-A}C_D^0 $         =>  Serie 3 : polluant 1 dans standard 2 
# - $C_{AD} = K_{A-D}C_A^0$            =>  Serie 4 : polluant 2 dans la standard 1

# In[3]:


def cal_conc1(x,y,z,h,Ca,Cd):
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


# 
# 
# # **<center><font color='blue'> méthode  monoexponentielle </font></center>**

# # f_decay $(x,a,tau)$ = $ \epsilon + a\exp (\frac{-x}{tau} )  $

# In[4]:


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
        y=y/max(y)
        popt,pcov=curve_fit(f_decay,x,y,bounds=(0, np.inf));
        df1=df1.append({'A'+VAR.split('/')[-1] :popt[0] , 'Tau'+VAR.split('/')[-1] :popt[1]} , ignore_index=True)
        f=f_decay(x,*popt)
        ax.plot(x,y,label="Intensité réelle");
        ax.plot(x,f,label="Intensité estimée");
        ax.set_title(" solution "+df.columns[2*i]);
        ax.set_xlabel('Temps(ms)');
        ax.set_ylabel('Intensité(p.d.u)');
        plt.legend();
    plt.show();
    return(df1)   
    


# # **<center><font color='blue'> méthode  double exponentielle </font></center>**

# # f_decay $(x,a1,t1,a2,t2)$ = $ \epsilon + a1\exp (\frac{-x}{t1} ) +a2\exp (\frac{-x}{t2})  $
# ## tau = $ \frac{a1t1^2 + a2t2^2}{ a1t1 + a2t2} $

# In[5]:


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
    


# 
# 
# # **<center><font color='blue'>   méthode  gaussiennes  </font></center>**

# # f_decay $(x,a1,t1,c)$ = $ \epsilon + a1\exp (\frac{-x}{t1} )  +\frac{a2}{2}\exp (\frac{-x}{t1+1.177c} ) +\frac{a2}{2}\exp (\frac{-x}{t1-1.177c})  $

# In[6]:


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
        y=np.log(y)
        f=np.log(f_decay(x,*popt))
        ax.plot(x,y,label="log  Intensité réelle");
        ax.plot(x,f,label="log Intensité estimée");
        ax.set_title(" solution "+df.columns[2*i]);
        ax.set_xlabel('Temps(ms)');
        ax.set_ylabel('log(Intensité(p.d.u))');
        ax.legend();
    plt.show();
    
    return(df2)


# 
# 
# 
# # **<center><font color='blue'> Fonction pour  regression linéaire </font></center>**

# ## Calcule concentration en fonction de durée de vie 
# Nous avons utilisé trois fonction pour la regression linéaire : 
#   - LinearRegression()
#   - RANSACRegressor()
#   - np.polyfit()

# In[7]:


## regression avec linearregression
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


# In[65]:


def regression6(result, std, unk, ss, sum_kchel):
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

        def func(x, a, b, c): # x-shifted log
              return a*np.log(x + b)/2+c
        initialParameters = np.array([1.0, 1.0, 1.0])
        log_params, _= curve_fit(func, x, kchel, initialParameters,maxfev=50000)
        log_r2 = r2_score(kchel,func(x, *log_params))
        
        best_model = func(x, *log_params)
        plt.scatter(x, kchel)
        plt.plot(x, best_model, 'm')
        plt.show()

       
        y_intercept = func(0, *log_params)
        print("y_intercept:", y_intercept)
        x_inter = np.exp(-2*log_params[2]/log_params[0]) - log_params[1]
        x_inter=np.array([x_inter])
        print("x_intercepts:", x_inter)
        slope = -log_params[1] *func(x_inter, *log_params)
        con_poly3.append(x_inter)
        con2.append(x_inter)
    return con_poly3



# In[ ]:





# In[ ]:





# In[287]:





# In[ ]:





# In[ ]:





# ## calcul concentration en fonction du nombre d'ion chélaté 

# In[9]:


def regression3(result,std,unk,ss,sum_kchel):
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
        mymodel = np.poly1d(np.polyfit(x, kchel, 1))
        print('\n',f"\033[031m {result.columns[2*i+1][4:]} \033[0m",'\n')
        plt.scatter(x, kchel)
        plt.plot(x, mymodel(x),'m')
        plt.xlabel('Concentration solution standard(ppm)');
        plt.ylabel('nombre d\'ion chélaté ' );
        plt.title('Courbe de calibration'+'du polluant '+result.columns[2*i+1][4:])
        plt.legend();
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


# # **<center><font color='blue'> Fonction pour  regression non linéaire </font></center>**

#  - Polynôme de degré 3 

# In[10]:


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
        print('\n',f"\033[031m {result.columns[2*i+1][4:]} \033[0m",'\n')
        plt.scatter(x, kchel)
        plt.plot(x, mymodel(x),'m')
        plt.xlabel('Concentration solution standard(ppm)');
        plt.ylabel('nombre d\'ion chélaté ' );
        plt.title('Courbe de calibration'+'du polluant '+result.columns[2*i+1][4:])
        plt.legend();
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
        print("slope", slope)
        xinter=y_intercept/slope
        con2.append(xinter)
    return(con_poly3)


# # **<center><font color='blue'>  Séléctionner les 4 Séries  </font></center>**

# In[11]:


def browseFiles2():
	filename = filedialog.askopenfilenames(initialdir = "http://localhost:8888/tree/Stage",
										title = "Select a File",
										filetypes = (("Csv files",
														"*.csv*"),
													("all files",
														"*.*")))
	return(filename)


# In[48]:


VARS=browseFiles2()


# 
# # **<center><font color='blue'>  Entrer les valeurs   </font></center>**

# In[49]:


unk=2.8 # volume inconnue
unk=2.6 # le 19-06
unk=2.6
std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2] # 06-06
#std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2,1] 
#std=[0,0,0.025,0.05,0.075,0.1,0.125,0.2,1.7]
#std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1] # Volume standard 07-06 , 12-06
#std=[0,0,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.5,1] # Volume standard 08-06 
#std=[0,0,0.05,0.075,0.1,0.125,0.15,0.175,0.2,1] # 09-06
std=[0,0,0.025,0.075,0.125,0.2,0.5,0.7,1,1.5,4] # Volume standard 20-06 , 21-06
#std=[0,0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,3]  # 15-06  , 16-06 
#std=[0,0,0.025,0.075,0.125,0.2,0.5,0.7,1,1.5,3] 
ss1=100 # solution standard serie 1
ss2=100 # standard serie 2
ss3=100 # standard serie 3
ss4 =100 # standard serie 4
rev=0.4 # volume reveratrice
Ca=10 # concentration initiale du polluant A dans la serie 4
Cd=10 # concentration initiale du polluant D dans la serie 3


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
# ### **<center><font>    I )  On fit l'intensité avec une mono_exponentielle puis on calcul les concentrations en utilisant une regressions linéaire ensuite non lineaire ( degré 3)    </font></center>**

# In[50]:


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

# In[41]:


result4.style.background_gradient(cmap="Greens")


# 
# ## **<center><font> I-1) Calcul des concentrations en fonction durée de vie par une regression linéaire  </font></center>**

# In[21]:


ss=[ss1,ss2,ss3,ss4]
concentration4=regression1(result4,std,unk,ss) 


# ## **<center><font>  resultats des concentrations obtenuent dans chaque serie </font></center>**

# In[22]:


concentration4
serie=['s1','s2','s3','s4']
concentration4.index=serie
concentration4.style.background_gradient(cmap="Greens")


# ### Les concentrations finales pour chaque polluant 

# In[23]:


polyfit=concentration4[concentration4.columns[0]]
stats_lingress=concentration4[concentration4.columns[1]]
ransac=concentration4[concentration4.columns[2]]
r2=cal_conc(*polyfit,Ca,Cd)
r3=cal_conc(*stats_lingress,Ca,Cd)
r4=cal_conc(*ransac,Ca,Cd)
r5=pd.concat([r2,r3,r4],axis=1)
r5.columns=['polyfit','stats_lingress','ransac']
r5.style.background_gradient(cmap="Greens")


# In[ ]:





# 
# ## **<center><font> I-1) Calcul des concentrations en fonction nombre d'ion chélaté  par une regression linéaire  </font></center>**

# ### Calcul kchel et sum_k pour chaque serie 

# In[18]:


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

# In[19]:


sum_kchel3.style.background_gradient(cmap="Greens")


# In[26]:


ss=[ss1,ss2,ss3,ss4]
concentrationCC4=regression3(result4,std,unk,ss,sum_kchel3) 


# In[27]:


concentrationCC4
cc=pd.DataFrame(concentrationCC4)
cc.style.background_gradient(cmap="Greens")


# In[28]:


r2=cal_conc(*concentrationCC4,Ca,Cd)
r2.style.background_gradient(cmap="Greens")


# 
# ## **<center><font>   I-2 ) Calcul des concentrations en utilisant une regression non lineaire ( degré 3) </font></center>**

# In[66]:


ss=[ss1,ss2,ss3,ss4]
c=regression6(result4,std,unk,ss,sum_kchel3) 


# In[76]:


r2=cal_conc(*c,Ca,Cd)
r2.style.background_gradient(cmap="Greens")


# In[29]:


ss=[ss1,ss2,ss3,ss4]
concentrationC4=regression2(result4,std,unk,ss,sum_kchel3)


# ## Resultast des concentrations obtenuent dasn chaque serie 

# In[93]:


concen =pd.DataFrame(concentrationC4)
serie=['s1','s2','s3','s4']
concen.index=serie
concen.style.background_gradient(cmap="Greens")


# ### Resultats des concentrations de chaque polluant

# In[94]:


r1=cal_conc(*concentrationC4,Ca,Cd)
r1.style.background_gradient(cmap="Greens")


# In[ ]:





# 
# # **<center><font>  méthode double_exponentielle   </font></center>**

# In[133]:


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

# In[96]:


result2.style.background_gradient(cmap="Blues")


# 
# # **<center><font>  calcule de la concentration en fonction durée de vie  pour la  regression linéaire  </font></center>**

# In[97]:


ss=[ss1,ss2,ss3,ss4]
concentrationC3=regression1(result2,std,unk,ss) 


# In[98]:


serie=['s1','s2','s3','s4']
concentrationC3.index=serie
concentrationC3.style.background_gradient(cmap="Blues")


# ## Resultats des concentrations pour chaque série 

# In[99]:


polyfit=concentrationC3[concentrationC3.columns[0]]
stats_lingress=concentrationC3[concentrationC3.columns[1]]
ransac=concentrationC3[concentrationC3.columns[2]]
r2=cal_conc(*polyfit,Ca,Cd)
r3=cal_conc(*stats_lingress,Ca,Cd)
r4=cal_conc(*ransac,Ca,Cd)
r5=pd.concat([r2,r3,r4],axis=1)
r5.columns=['polyfit','stats_lingress','ransac']
r5.style.background_gradient(cmap="Blues")


# 
# # **<center><font>  calcule de la concentration en fonction de nombre d'ion chélaté  pour la  regression linéaire  </font></center>**

# In[100]:


for j in range(4):
    tt2=result2[result2.columns[2*j+1]]
    s_k=pd.DataFrame(fun(tt2))
    s_k=s_k.T
    s_k.columns=['sum_k'+result2.columns[2*j+1].split('_')[-1],'kchel'+result2.columns[2*j+1].split('_')[-1]]
    sum_kchel2=pd.concat([sum_kchel2,s_k],axis=1)


# In[101]:


result10=pd.DataFrame()
j=[1,1,2,2,3,3,4,4]
for i in j:
    for k in range(len(sum_kchel2.columns)) : 
        if sum_kchel2.columns[k].find('S'+str(i))!=-1:
            result10=pd.concat([result10,sum_kchel2[sum_kchel2.columns[k]]],axis=1)
result10 = result10.loc[:,~result10.columns.duplicated()]    


# In[102]:


sum_kchel2=result10
result10.style.background_gradient(cmap="Blues")


# In[103]:


ss=[ss1,ss2,ss3,ss4]
concentrationCC3=regression3(result2,std,unk,ss,sum_kchel2) 


# In[104]:


concentrationCC3
cc=pd.DataFrame(concentrationCC3)
cc.style.background_gradient(cmap="Blues")


# In[105]:


r2=cal_conc(*concentrationCC3,Ca,Cd)
r2.style.background_gradient(cmap="Blues")


# 
# # **<center><font>  calcule de la concentration pour la  regression non linéaire  </font></center>**

# In[106]:


ss=[ss1,ss2,ss3,ss4]
concentration3=regression2(result2,std,unk,ss,sum_kchel2) 


# In[107]:


concen =pd.DataFrame(concentration3)
serie=['s1','s2','s3','s4']
concen.index=serie
concen.style.background_gradient(cmap="Blues")


# ## Calcul concentration final 

# In[109]:


r1=cal_conc(*concentration3,Ca,Cd)
r1.style.background_gradient(cmap="Blues")


# 
# # **<center><font>  méthode gaussiennes    </font></center>**

# In[143]:


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

# In[111]:


result.style.background_gradient(cmap="Purples") 


# 
# # **<center><font>  resultats de la concentration en fonction durée de vie par  regression  linéaire  </font></center>**

# In[112]:


ss=[ss1,ss2,ss3,ss4]
concentrationC1=regression1(result,std,unk,ss) 


# In[113]:


concentrationC1
serie=['s1','s2','s3','s4']
concentrationC1.index=serie
concentrationC1.style.background_gradient(cmap="Purples")


# ## Calcul concentration final 

# In[114]:


polyfit=concentrationC1[concentrationC1.columns[0]]
stats_lingress=concentrationC1[concentrationC1.columns[1]]
ransac=concentrationC1[concentrationC1.columns[2]]
r2=cal_conc(*polyfit,Ca,Cd)
r3=cal_conc(*stats_lingress,Ca,Cd)
r4=cal_conc(*ransac,Ca,Cd)
r5=pd.concat([r2,r3,r4],axis=1)
r5.columns=['polyfit','stats_lingress','ransac']
r5.style.background_gradient(cmap="Purples")


# 
# # **<center><font>  resultats de la concentration en fonction de nombre d'ion chélaté par  regression  linéaire  </font></center>**

# In[115]:


for j in range(4):
    tt1=result[result.columns[2*j+1]]
    s_k=pd.DataFrame(fun(tt1))
    s_k=s_k.T
    s_k.columns=['sum_k'+result.columns[2*j+1].split('_')[-1],'kchel'+result.columns[2*j+1].split('_')[-1]]
    sum_kchel1=pd.concat([sum_kchel1,s_k],axis=1)


# In[116]:


sum_kchel1.style.background_gradient(cmap="Purples")


# In[117]:


ss=[ss1,ss2,ss3,ss4]
concentrationCC1=regression3(result,std,unk,ss,sum_kchel1) 


# In[118]:


cc=pd.DataFrame(concentrationCC1)
cc.style.background_gradient(cmap="Purples")


# In[119]:


r1=cal_conc(*concentrationCC1,Ca,Cd)
r1.style.background_gradient(cmap="Purples")


# 
# # **<center><font>  resultats regression non  linéaire  </font></center>**

# In[120]:


ss=[ss1,ss2,ss3,ss4]
concentration1=regression2(result,std,unk,ss,sum_kchel1) 


# In[121]:


concen =pd.DataFrame(concentration1)
serie=['s1','s2','s3','s4']
concen.index=serie
concen.style.background_gradient(cmap="Purples")


# 
# # **<center><font>  Concentration final   </font></center>**

# In[122]:


r2=cal_conc(*concentration1,Ca,Cd)
r2.style.background_gradient(cmap="Purples")


'''
    st.code(code,language="python")

   


page_names_to_funcs = {
    "identification": identification,
    "image": image,
    "Quantification": Quantification,
    "code python":code_python ,
    "lissage et déconvolution":liss_deconv ,
}

selected_page = st.sidebar.selectbox("Selectionner ", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


