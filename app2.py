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
	
def home():
    st.markdown('<h1 style="text-align: center;"> VAI-TRF </h1>', unsafe_allow_html=True)
    st.markdown(' <h2 style="text-align: center;"> Validation of Artificial Intelligence Time-Resolved Fluorescence method for the real-time monitoring of critical pollutants in industrial and municipal effluents </h2>', unsafe_allow_html=True)
    
    st.image("https://github.com/dKosarevsky/AI-Talks/raw/main/ai_talks/assets/img/ai_face.png")
    st.sidebar.image("https://th.bing.com/th/id/OIP.yP8OOsRFeygNvQFJr7SYJQHaB5?w=328&h=89&c=7&r=0&o=5&dpr=1.5&pid=1.7", use_column_width=True)
    st.sidebar.image("https://th.bing.com/th/id/OIP.CKq0LG-laXKo7RW19OIAcwHaCi?w=345&h=119&c=7&r=0&o=5&dpr=1.5&pid=1.7", use_column_width=True)
    st.sidebar.image("https://ilm.univ-lyon1.fr/templates/mojito/images/logo.jpg", use_column_width=True)
    st.sidebar.image("https://th.bing.com/th/id/OIP.hxQSgpDEPe0xWlxVsLqxFQHaCm?w=346&h=123&c=7&r=0&o=5&dpr=1.5&pid=1.7", use_column_width=True)
    
    st.write('          ')
    st.sidebar.write("<p style='text-align: center;'> Alioune Gaye: Stagiaire ILM %s</p>" % url, unsafe_allow_html=True)
    col3,col4=st.columns(2)
    col3.image("t1.png")
    col4.image("t2.png")
    col5,col6=st.columns(2)
    col5.image("t3.png")
    col6.image("t4.png")
    st.image("opentrons.png")


def identification():
    st.sidebar.image("https://th.bing.com/th/id/OIP.NztfNu6p_efe7yI8BXI4iAHaEK?w=330&h=181&c=7&r=0&o=5&dpr=1.5&pid=1.7", use_column_width=True)
    #st.sidebar.markdown('<h1 style="text-align: center;">Identification du polluants 🎈</h1>', unsafe_allow_html=True)
    def main():
        st.markdown('<h1 style="text-align: center;">Identification du polluants</h1>', unsafe_allow_html=True)
        st.markdown('<h1 style="text-align: center;">Base de donnée</h1>',unsafe_allow_html=True)
        col3,col4=st.sidebar.columns(2)
        col3.image("https://ilm.univ-lyon1.fr/templates/mojito/images/logo.jpg", use_column_width=True)
        col4.image("https://formation-professionnelle.universite-lyon.fr/var/site/storage/images/3/3/5/0/533-17-fre-FR/Lyon-1-Claude-Bernard.png", use_column_width=True)
        st.sidebar.write("<p style='text-align: center;'> Alioune Gaye: Stagiaire ILM %s</p>" % url, unsafe_allow_html=True)
        st.sidebar.write("<p style='text-align: center;'>Fait un choix: Apprentissage supervisé par régression ou par classification.</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>Nous allons procéder comme suit :</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>1 - Chargement de votre base de données</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>2 - Analyse exploratoire des données</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>3 - Sélection de la variable cible et de la méthode d'apprentissage</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>4 - Construction du modèle</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>5 - Vous pouvez maintenant telecharger votre modèle puis procéder au depoiement de ce dernier</p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center;'>6 - Aprés avoir créer notre pipeline nous procédons à la prédiction automatique de notre modéle</p>", unsafe_allow_html=True)
        file = st.file_uploader("entrer les données sous format csv ", type=['csv'])
        if file is not None:
            df=load_data(file)
            #type=st.selectbox("selectionner le target",["Homogene","Heterogene"])
            df=df[df.columns[0:]]
            n=len(df.columns)
            df=df[df[df.columns[:(n-1)]].duplicated()!=True]
            n=len(df.columns)
            X=df[df.columns[:(n-1)]]# on prend les variables numériques 
            y=df[df.columns[-1]] # le target
            st.dataframe(df.head())
            pr = df.profile_report()
            if st.button('statistique descriptive'):
                 st_profile_report(pr)
            target=st.selectbox("selectionner le target",df.columns)
            methode=st.selectbox("selectionner la méthode ",["Regression","Classification"])
            df=df.dropna(subset=target)
            if methode=="Classification":
                if st.button(" Machine learning "):
                     setup_data = setup(data=df,target = target,
                        train_size =0.75,categorical_features =None,
                        normalize = False,normalize_method = None,fold=5)
                     conf_df = pd.DataFrame(setup_data._display_container[0])
                     st.write('Configuration')
                     st.write(conf_df)
                     #r=compare_models(round=2)
                     #save_model(r,"best_model")
                     st.success("youpiiiii classification fonctionne \U0001F600")
                     st.write(" Les meilleurs algorithmes ")
                     #results = pull()
                     #st.write(results)
                     st.write(" Performances du modèle :")
                     
                     #final_model1 = create_model(r,fold=5,round=2)
                     col5,col6=st.columns(2)
                     col5.write('AUC')
                     #plot_model(final_model1,plot='auc',save=True)
                     col5.image("AUC.png")
                     col6.write("class_report")
                     #plot_model(final_model1,plot='class_report',save=True)
                     col6.image("Class Report.png")
                  
                     col7,col8=st.columns(2)
                     col7.write("confusion_matrix")
                     #plot_model(final_model1,plot='confusion_matrix',save=True)
                     col7.image("Confusion Matrix.png")
                     #tuned_model = tune_model(final_model1,optimize='AUC',round=2,n_iter=10);# optimiser le modéle
                     col8.write("boundary")
                     #plot_model(final_model1 , plot='boundary',save=True)
                     col8.image("Decision Boundary.png")
                    
                     col9,col10=st.columns(2)
                     col9.write("feature")
                     #plot_model(estimator = tuned_model, plot = 'feature',save=True)
                     col9.image("Feature Importance.png")
                     col10.write("learning")
                     #plot_model(estimator = final_model1, plot = 'learning',save=True)
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
    st.sidebar.image("RNN.png")
    st.markdown("# image 3D ❄️")
    #st.sidebar.markdown('<h1 style="text-align: center;">Identification d\'images ❄️ </h1>', unsafe_allow_html=True)
    st.sidebar.write("<p style='text-align: center;'> Alioune Gaye : Stagiaire ILM %s</p>" % url, unsafe_allow_html=True)
    # Création du modèle Keras
    st.sidebar.markdown("<p style='text-align: center;'> Création du modèle Keras </p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'> Voici les principales couches du modèle : </p>", unsafe_allow_html=True)

    # Couches du modèle
    model_layers_info = [
    "Couche de Convolution (32 filtres, taille 3x3) avec fonction d'activation ReLU, prenant une image en entrée de taille (224, 224, 3)",
    "Couche de MaxPooling (taille 2x2) pour réduire la dimensionnalité",
    "Couche de Convolution (32 filtres, taille 3x3) avec fonction d'activation ReLU",
    "Nouvelle couche de MaxPooling (taille 2x2)",
    "Couche d'aplatissement (Flatten) pour convertir les données en vecteur",
    "Couche Dense (128 neurones) avec fonction d'activation ReLU",
    "Couche Dense de sortie (4 neurones) avec fonction d'activation Softmax pour la classification multi-classes"
    ]

    for i in range(len(model_layers_info)):
        st.sidebar.markdown( "<p style='text-align: center;'>" + "{} - ".format(i) + model_layers_info[i] +" </p>", unsafe_allow_html=True)

    # Compilation du modèle
    st.sidebar.markdown("<p style='text-align: center;'> Une fois les couches définies, nous compilons le modèle en spécifiant l'optimiseur et la fonction de perte :</p>", unsafe_allow_html=True)
    #st.sidebar.code("model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])")
    st.sidebar.markdown("<p style='text-align: center;'> Nous utilisons l'optimiseur Adam et la fonction de perte 'sparse_categorical_crossentropy' car les étiquettes sont encodées sous forme d'entiers.</p>", unsafe_allow_html=True)    
    from keras.models import load_model
    st.markdown('<h1 style="text-align: center;">Prédiction image 3D </h1>', unsafe_allow_html=True)
    model = load_model('model_final2.pkl')
    f=np.array(['ATMP+DTPMP', 'DTPMP', 'DTPMP+DTPA', 'EDTA'])
    from PIL import Image
    from tensorflow.keras.preprocessing import image
    from PIL import Image
    f=np.array(['ATMP+DTPMP', 'DTPMP', 'EDTA', 'DTPMP+DTPA'])
    def get_image_path(img):
        # Create a directory and save the uploaded image.
        file_path = f"data/uploadedImages/{img.name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as img_file:
             img_file.write(img.getbuffer())
        return file_path
    def predict_class(model,image):
        img2 = cv2.imread(image)
        if img2 is not None:
             img2 = cv2.resize(img2, (224, 224))
             img2=[img2]
        else:
             print(f"Erreur lors du chargement de l'image : {img2}")
        # Convertir en numpy array et normaliser les valeurs des pixels
        imag = np.array(img2) / 255.0
        pred=model.predict(imag)
        index = np.argmax(pred,axis=1)
        f.sort()
        pred_value = f[index]
        score=np.max(pred)
        return(pred_value,score)
    file = st.file_uploader("Entrer l'image", type=["jpg", "png"])
    if file is None:
        st.text("entrer l'image à prédire")
    else:
        file=get_image_path(file)
        label,score = predict_class(model, file)
        st.image(file, use_column_width=True)
        
        st.markdown("##  🤖  Il s'agit du polluant ")
        st.markdown('<h1 style="text-align: center;">'+'{}'.format(label[0])+'</h1>', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center;">'+'Avec un score =  {}'.format(score)+'</h2>', unsafe_allow_html=True)
    st.image("https://ilm.univ-lyon1.fr//images/slides/SLIDER7.png")
    
   
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
        st.sidebar.write("<p style='text-align: center;'> Alioune Gaye : Stagiaire ILM %s</p>" % url, unsafe_allow_html=True)
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
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit;
import math
import scipy.integrate as spi
import csv
def liss_deconv():
    def main():
        st.sidebar.write("<p style='text-align: center;'> Lissage avec algorithme de Savitz-golay, la fonction savgol_filter prend en paramètre: y ou x : il s\'agit de la donnée à filtrer, la longueur de la fenetre de lissage, le degré du polynome de lissage. Elle renvoie: La donnée filtrée.</p>",unsafe_allow_html=True)
        st.sidebar.image("savot.png")
        st.sidebar.write("Déconvolution formule : ")
        st.sidebar.latex(r''' \fcolorbox{black}{brown}{$I(\lambda) =  \sum_{i=1}^{4}I_i \frac{\exp\left(-\frac{(\lambda - \lambda_i)^2}{2\sigma_i^2}\right)}{\sigma_i \sqrt{2\pi}}  $}''')
        st.sidebar.image("deconv.png")
        st.markdown('# <h1 style="text-align: center;"> Lissage et Déconvolution spectre  :</h1>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(" Choisir un fichier ", type=['csv'])
        st.image("https://ilm.univ-lyon1.fr//images/slides/CARROUSSEL-15.png")
        def find_delimiter(filename):
            sniffer = csv.Sniffer()
            with open(filename) as fp:
                delimiter = sniffer.sniff(fp.read(5000)).delimiter
            return delimiter
        def get_image_path(img):
            # Create a directory and save the uploaded image.
            file_path = f"data/uploadedImages/{img.name}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as img_file:
                 img_file.write(img.getbuffer())
            return file_path
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
           file=get_image_path(uploaded_files)
           delim=find_delimiter(file)
           df = pd.read_csv(uploaded_files, delimiter=delim) 
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
        st.markdown('# <h1 style="text-align: center;"> Construction de la base de donnée  :</h1>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Choisir les fichiers csv ", accept_multiple_files=True)
        st.image("https://ilm.univ-lyon1.fr//images/slides/CARROUSSEL-15.png")
        def find_delimiter(filename):
            sniffer = csv.Sniffer()
            with open(filename) as fp:
                delimiter = sniffer.sniff(fp.read(5000)).delimiter
            return delimiter
        def get_image_path(img):
            # Create a directory and save the uploaded image.
            file_path = f"data/uploadedImages/{img.name}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as img_file:
                 img_file.write(img.getbuffer())
            return file_path
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
        Base=pd.DataFrame()
        for uploaded_file in uploaded_files:
            file = get_image_path(uploaded_file)
            delim=find_delimiter(file)
            df = pd.read_csv(uploaded_file, delimiter=delim) 
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
            Base=pd.concat([df_dp,Base])
        st.write("Création de la base de donnée")
        st.write(Base)
	           
    

page_names_to_funcs = {
    "Home":home,
    "identification": identification,
    "image": image,
    "Quantification": Quantification,
    "lissage et déconvolution":liss_deconv 
}

selected_page = st.sidebar.selectbox("Selectionner ", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


