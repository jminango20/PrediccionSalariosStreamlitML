#Created by Juan Minango
import streamlit as st
#EDA Pakgs
import pandas as pd
import numpy as np
#Data Viz Pkgs
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

#ML Pkgs
import joblib
import os
from PIL import Image
import sqlite3
import datetime

#Funciones
#Get Value
def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value
#Get the Keys
def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return key

#Load ML Models
def load_prediction_models(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

#Class DB(Base de datos)
class Monitor(object):
	"""docstring for Monitor"""
	conn = sqlite3.connect('data.db')
	c = conn.cursor()
	def __init__(self, age=None, workclass=None, fnlwgt=None, education=None, education_num=None, marital_status=None, occupation=None, relationship=None, race=None, sex=None, capital_gain=None, capital_loss=None, hours_per_week=None, native_country=None, predicted_class=None, model_class=None):
		super(Monitor, self).__init__()
		self.age = age
		self.workclass = workclass
		self.fnlwgt = fnlwgt
		self.education = education
		self.education_num = education_num
		self.marital_status = marital_status
		self.occupation = occupation
		self.relationship = relationship
		self.race = race
		self.sex = sex
		self.capital_gain = capital_gain
		self.capital_loss = capital_loss
		self.hours_per_week = hours_per_week
		self.native_country = native_country
		self.predicted_class = predicted_class
		self.model_class = model_class
        #self.time_of_prediction = time_of_prediction
      

	def __repr__(self):
		# return "Monitor(age ={self.age},workclass ={self.workclass},fnlwgt ={self.fnlwgt},education ={self.education},education_num ={self.education_num},marital_status ={self.marital_status},occupation ={self.occupation},relationship ={self.relationship},race ={self.race},sex ={self.sex},capital_gain ={self.capital_gain},capital_loss ={self.capital_loss},hours_per_week ={self.hours_per_week},native_country ={self.native_country},predicted_class ={self.predicted_class},model_class ={self.model_class})".format(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class)
		"Monitor(age = {self.age},workclass = {self.workclass},fnlwgt = {self.fnlwgt},education = {self.education},education_num = {self.education_num},marital_status = {self.marital_status},occupation = {self.occupation},relationship = {self.relationship},race = {self.race},sex = {self.sex},capital_gain = {self.capital_gain},capital_loss = {self.capital_loss},hours_per_week = {self.hours_per_week},native_country = {self.native_country},predicted_class = {self.predicted_class},model_class = {self.model_class})".format(self=self)

	def create_table(self):
		self.c.execute('CREATE TABLE IF NOT EXISTS predictiontable(age NUMERIC,workclass NUMERIC,fnlwgt NUMERIC,education NUMERIC,education_num NUMERIC,marital_status NUMERIC,occupation NUMERIC,relationship NUMERIC,race NUMERIC,sex NUMERIC,capital_gain NUMERIC,capital_loss NUMERIC,hours_per_week NUMERIC,native_country NUMERIC,predicted_class NUMERIC,model_class TEXT, time_of_prediction TEXT)')

	def add_data(self):
		self.c.execute('INSERT INTO predictiontable(age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,predicted_class,model_class) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class))
		self.conn.commit()

	def view_all_data(self):
		self.c.execute('SELECT * FROM predictiontable')
		data = self.c.fetchall()
		# for row in data:
		# 	print(row)
		return data



#Programa Principal
def main():
    """Prediccion Salario con ML"""
    st.title("Prediccion de Salarios")
    activity = ["Analisis Exploratorio Datos","Prediccion","Metrica","Paises"]
    choice = st.sidebar.selectbox("Escojer una actividad: ", activity)
    st.sidebar.info("Desarrollado por: Juan Minango y Pablo Minango")
    st.sidebar.markdown("[Juan Minango](https://www.linkedin.com/in/juan-carlos-minango-negrete-4b6106197/)")
    st.sidebar.markdown("[Pablo Minango](https://www.linkedin.com/in/pablo-david-minango-negrete-3b275b141/)")
    st.sidebar.markdown("[JD-Techn](https://jdtechn.com/)")
    #Load File
    df = pd.read_csv("data/adult_salary.csv")

    #ExploringDataAnalize
    if choice == "Analisis Exploratorio Datos":
        st.subheader("Planteamiento del Problema")
        st.write("Ha sido proporcionado un conjunto de datos de salarios que tiene 15 columnas y 48842 filas. Nuestra tarea es analizar el conjunto de datos y predecir si el ingreso de un adulto excederá de 50k ($USD 50.000) por año o no mediante el desarrollo de un modelo supervisado de aprendizaje automático (Machine Learning). Para lo cuál se ha considerado 3 modelos de Machine Learning los cuáles son: Logistic Regression, Naive Bayes y Random Forest.")
        st.markdown("[UCI: Base de Datos](https://archive.ics.uci.edu/ml/datasets/Adult)")
        st.markdown("[Teoria: Logistic Regression, Naive Bayes y Random Forest](https://www.amazon.com.br/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291)")       
        st.subheader("Seccion Analisis Exploratorio de Datos")
        #Previews
        if st.checkbox("Vista Previa Dataset"):
            number = st.number_input("Numero de Filas a Mostrar: ", value=0)
            st.dataframe(df.head(number))
        #Show Columns/Rows
        if st.button("Nombre de las Columnas"):
            st.write(df.columns)
        #Description
        if st.checkbox("Mostrar Descripcion del DataSet"):
            st.write(df.describe())
        #Shape
        if st.checkbox("Mostrar Dimensiones del DataSet"):
            st.write(df.shape)
            data_dim = st.radio("Mostrar Dimensiones por: ",("Filas", "Columnas"))
            if data_dim == "Filas":
                st.write("Numero de Filas: ")
                st.write(df.shape[0])
            elif data_dim == "Columnas":
                st.write("Numero de Columnas: ")
                st.write(df.shape[1])
        #Selection Columns
        if st.checkbox("Seleccionar las Columnas a Mostrar"):
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect("Seleccionar las Columnas: ", all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)
        #Selection Filas
        if st.checkbox("Seleccionar las Filas a Mostrar"):
            selected_indices = st.multiselect("Seleccionar las Filas: ", df.head(11).index)
            selected_rows = df.loc[selected_indices]
            st.dataframe(selected_rows)
        #Value Counts
        if st.button("Conteo Valores"):
            st.text("Conteo de Valores del Target")
            st.write(df.iloc[:,-1].value_counts())
        #Plot
        if st.checkbox("Mostrar Grafico de Correlacion [MatplotLib]"):
            plt.matshow(df.corr())
            st.pyplot()
        if st.checkbox("Mostrar Grafico de Correlacion [SeaBorn]"):
            st.write(sns.heatmap(df.corr(),annot=True))
            st.pyplot()
    
    #Prediction
    elif choice == "Prediccion":
        st.subheader("Seccion Prediccion")
        #Diccionario
        d_workclass = {"Never-worked": 0, "Private": 1, "Federal-gov": 2, "?": 3, "Self-emp-inc": 4, "State-gov": 5, "Local-gov": 6, "Without-pay": 7, "Self-emp-not-inc": 8}
        d_education = {"Some-college": 0, "10th": 1, "Doctorate": 2, "1st-4th": 3, "12th": 4, "Masters": 5, "5th-6th": 6, "9th": 7, "Preschool": 8, "HS-grad": 9, "Assoc-acdm": 10, "Bachelors": 11, "Prof-school": 12, "Assoc-voc": 13, "11th": 14, "7th-8th": 15}
        d_marital_status = {"Separated": 0, "Married-spouse-absent": 1, "Married-AF-spouse": 2, "Married-civ-spouse": 3, "Never-married": 4, "Widowed": 5, "Divorced": 6}
        d_occupation = {"Tech-support": 0, "Farming-fishing": 1, "Prof-specialty": 2, "Sales": 3, "?": 4, "Transport-moving": 5, "Armed-Forces": 6, "Other-service": 7, "Handlers-cleaners": 8, "Exec-managerial": 9, "Adm-clerical": 10, "Craft-repair": 11, "Machine-op-inspct": 12, "Protective-serv": 13, "Priv-house-serv": 14}
        d_relationship = {"Other-relative": 0, "Not-in-family": 1, "Own-child": 2, "Wife": 3, "Husband": 4, "Unmarried": 5}
        d_race = {"Amer-Indian-Eskimo": 0, "Black": 1, "White": 2, "Asian-Pac-Islander": 3, "Other": 4}
        d_sex = {"Female": 0, "Male": 1}
        d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}
        d_class = {">50K": 0, "<=50K": 1}
        #ML Aspect User Input
        age = st.slider("Seleccionar Edad", 17, 90)
        workclass = st.selectbox("Seleccionar el tipo de Trabajo",tuple(d_workclass.keys()))
        fnlwgt = st.number_input("Enter FNLWGT",1.228500e+04,1.484705e+06)
        education = st.selectbox("Seleccionar tu Educacion",tuple(d_education.keys()))
        education_num = st.slider("Selecciona tu Nivel de Educacion",1,16)
        marital_status = st.selectbox("Selecciona Estado Civil",tuple(d_marital_status.keys()))
        occupation = st.selectbox("Selecciona Ocupacion",tuple(d_occupation.keys()))
        relationship = st.selectbox("Selecciona Relacionamiento",tuple(d_relationship.keys()))
        race = st.selectbox("Selecciona Raza",tuple(d_race.keys()))
        sex = st.radio("Selecciona Sexo",tuple(d_sex.keys()))
        capital_gain = st.number_input("Ganancia Capital",0,9999,value=0)
        capital_loss = st.number_input("Perdida Capital",0,4356,value=0)
        hours_per_week = st.number_input("Horas por Semana",1,99,value=1)
        native_country = st.selectbox("Selecciona Pais de Origen",tuple(d_native_country.keys()))
        #User Input
        k_workclass = get_value(workclass,d_workclass)
        k_education = get_value(education,d_education)
        k_marital_status = get_value(marital_status,d_marital_status)
        k_occupation = get_value(occupation,d_occupation)
        k_relationship = get_value(relationship,d_relationship)
        k_race = get_value(race,d_race)
        k_sex = get_value(sex,d_sex)
        k_native_country = get_value(native_country,d_native_country)
        #Result of User Input		
        selected_options = [age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country]
        vectorized_result = [age ,k_workclass ,fnlwgt ,k_education ,education_num ,k_marital_status ,k_occupation ,k_relationship ,k_race ,k_sex ,capital_gain ,capital_loss ,hours_per_week ,k_native_country]
        st.success(vectorized_result)
        st.subheader("Informacion Ingresada en Formato Lista")
        #Formato Lista
        st.info(selected_options)
        sample_data = np.array(vectorized_result).reshape(1, -1)
        #Formato JSON
        st.subheader("Informacion Ingresada en Formato JSON")
        prettified_result = {"age":age,
		"workclass":workclass,
		"fnlwgt":fnlwgt,
		"education":education,
		"education_num":education_num,
		"marital_status":marital_status,
		"occupation":occupation,
		"relationship":relationship,
		"race":race,
		"sex":sex,
		"capital_gain":capital_gain,
		"capital_loss":capital_loss,
		"hours_per_week":hours_per_week,
		"native_country":native_country}
        st.json(prettified_result)
        #Formato Numerico
        st.subheader("Informacion Ingresada en Formato Numerico")
        st.write(vectorized_result)
        #MAke Prediction
        st.header("Seccion ML para Prediccion")
        if st.checkbox("Realizar Prediccion"):
            all_ml_lists = ["LOGISTIC REGRESSION","RFOREST","NAIVE BAYES"]
            #Seleccion Modelo ML
            model_choice = st.selectbox("Escoje el Modelo ML: ",all_ml_lists)
            st.text("Usando la siguiente codificacion para prediccion: ")
            prediction_label = {">50k":0,"<=50k":1}
            st.write(prediction_label)
            if st.button("Predecir"):
                if model_choice == "LOGISTIC REGRESSION":
                    model_predictor = load_prediction_models("models/salary_logit_model_juan.pkl")
                    prediction = model_predictor.predict(sample_data)
                    st.text("Prediccion con ML Logistic Regression")
                    st.write(prediction)
                    if prediction == 1:
                        st.success("Salario es: <=50k ")
                    elif prediction == 0:
                        st.success("Salario es: >50k ")
                elif model_choice == "RFOREST":
                    model_predictor = load_prediction_models("models/salary_rf_model_juan.pkl")
                    prediction = model_predictor.predict(sample_data)
                    st.text("Prediccion con ML Random Forest")
                    st.write(prediction)
                    if prediction == 1:
                        st.success("Salario es: <=50k ")
                    elif prediction == 0:
                        st.success("Salario es: >50k ")
                elif model_choice == "NAIVE BAYES":
                    model_predictor = load_prediction_models("models/salary_nv_model_juan.pkl")
                    prediction = model_predictor.predict(sample_data)
                    st.text("Prediccion con ML Naive Bayes")
                    st.write(prediction)
                    if prediction == 1:
                        st.success("Salario es: <=50k ")
                    elif prediction == 0:
                        st.success("Salario es: >50k ")
                final_result = get_keys(prediction,prediction_label)
                model_class = model_choice
                time_prediction = datetime.datetime.now()
                monitor = Monitor(age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,final_result,model_class)
                monitor.create_table()
                monitor.add_data()
                #st.success("Salario Predecido como :: {}".format(final_result))
    
    # Countries
    elif choice == "Paises":
        st.subheader("Seccion Paises")
        #List of Countries
        d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}
        selected_countries =  st.selectbox("Escojer un Pais",tuple(d_native_country.keys()))
        #Selection Countries
        st.text(selected_countries)
        df2 = pd.read_csv("data/adult_salary_data.csv")
        result_df = df2[df2['native-country'].str.contains(selected_countries)]
        st.dataframe(result_df.head(10))
        countries_images = {'af': 'Afghanistan','al': 'Albania','dz': 'Algeria','as': 'American Samoa','ad': 'Andorra','ao': 'Angola','ai': 'Anguilla','aq': 'Antarctica','ag': 'Antigua And Barbuda','ar': 'Argentina','am': 'Armenia','aw': 'Aruba','au': 'Australia','at': 'Austria','az': 'Azerbaijan','bs': 'Bahamas','bh': 'Bahrain','bd': 'Bangladesh','bb': 'Barbados','by': 'Belarus','be': 'Belgium','bz': 'Belize','bj': 'Benin','bm': 'Bermuda','bt': 'Bhutan','bo': 'Olivia','ba': 'Bosnia And Herzegovina','bw': 'Botswana','bv': 'Bouvet Island','br': 'Brazil','io': 'British Indian Ocean Territory','bn': 'Brunei Darussalam','bg': 'Bulgaria','bf': 'Burkina Faso','bi': 'Burundi','kh': 'Cambodia','cm': 'Cameroon','ca': 'Canada','cv': 'Cape Verde','ky': 'Cayman Islands','cf': 'Central African Republic','td': 'Chad','cl': 'Chile','cn': "People'S Republic Of China",'cx': 'Hristmas Island','cc': 'Cocos (Keeling) Islands','co': 'Colombia','km': 'Comoros','cg': 'Congo','cd': 'Congo, The Democratic Republic Of','ck': 'Cook Islands','cr': 'Costa Rica','ci': "Côte D'Ivoire",'hr': 'Croatia','cu': 'Cuba','cy': 'Cyprus','cz': 'Czech Republic','dk': 'Denmark','dj': 'Djibouti','dm': 'Dominica','do': 'Dominican Republic','ec': 'Ecuador','eg': 'Egypt','eh': 'Western Sahara','sv': 'El Salvador','gq': 'Equatorial Guinea','er': 'Eritrea','ee': 'Estonia','et': 'Ethiopia','fk': 'Falkland Islands (Malvinas)','fo': 'Aroe Islands','fj': 'Fiji','fi': 'Finland','fr': 'France','gf': 'French Guiana','pf': 'French Polynesia','tf': 'French Southern Territories','ga': 'Gabon','gm': 'Gambia','ge': 'Georgia','de': 'Germany','gh': 'Ghana','gi': 'Gibraltar','gr': 'Greece','gl': 'Greenland','gd': 'Grenada','gp': 'Guadeloupe','gu': 'Guam','gt': 'Guatemala','gn': 'Guinea','gw': 'Guinea-Bissau','gy': 'Guyana','ht': 'Haiti','hm': 'Heard Island And Mcdonald Islands','hn': 'Honduras','hk': 'Hong Kong','hu': 'Hungary','is': 'Iceland','in': 'India','id': 'Indonesia','ir': 'Iran, Islamic Republic Of','iq': 'Iraq','ie': 'Ireland','il': 'Israel','it': 'Italy','jm': 'Jamaica','jp': 'Japan','jo': 'Jordan','kz': 'Kazakhstan','ke': 'Kenya','ki': 'Kiribati','kp': "Korea, Democratic People'S Republic Of",'kr': 'Korea, Republic Of','kw': 'Kuwait','kg': 'Kyrgyzstan','la': "Lao People'S Democratic Republic",'lv': 'Latvia','lb': 'Lebanon','ls': 'Lesotho','lr': 'Liberia','ly': 'Libyan Arab Jamahiriya','li': 'Liechtenstein','lt': 'Lithuania','lu': 'Luxembourg','mo': 'Macao','mk': 'Macedonia, The Former Yugoslav Republic Of','mg': 'Madagascar','mw': 'Malawi','my': 'Malaysia','mv': 'Maldives','ml': 'Mali','mt': 'Malta','mh': 'Marshall Islands','mq': 'Martinique','mr': 'Mauritania','mu': 'Mauritius','yt': 'Mayotte','mx': 'Mexico','fm': 'Micronesia, Federated States Of','md': 'Moldova, Republic Of','mc': 'Monaco','mn': 'Mongolia','ms': 'Montserrat','ma': 'Morocco','mz': 'Mozambique','mm': 'Myanmar','na': 'Namibia','nr': 'Nauru','np': 'Nepal','nl': 'Netherlands','an': 'Netherlands Antilles','nc': 'New Caledonia','nz': 'New Zealand','ni': 'Nicaragua','ne': 'Niger','ng': 'Nigeria','nu': 'Niue','nf': 'Norfolk Island','mp': 'Northern Mariana Islands','no': 'Norway','om': 'Oman','pk': 'Pakistan','pw': 'Palau','ps': 'Palestinian Territory, Occupied','pa': 'Panama','pg': 'Papua New Guinea','py': 'Paraguay','pe': 'Peru','ph': 'Philippines','pn': 'Pitcairn','pl': 'Poland','pt': 'Portugal','pr': 'Puerto Rico','qa': 'Qatar','re': 'Réunion','ro': 'Romania','ru': 'Russian Federation','rw': 'Rwanda','sh': 'Saint Helena','kn': 'Saint Kitts And Nevis','lc': 'Saint Lucia','pm': 'Saint Pierre And Miquelon','vc': 'Saint Vincent And The Grenadines','ws': 'Samoa','sm': 'San Marino','st': 'Sao Tome And Principe','sa': 'Saudi Arabia','sn': 'Senegal','cs': 'Serbia And Montenegro','sc': 'Seychelles','sl': 'Sierra Leone','sg': 'Singapore','sk': 'Slovakia','si': 'Slovenia','sb': 'Solomon Islands','so': 'Somalia','za': 'South Africa','gs': 'South Georgia And South Sandwich Islands','es': 'Spain','lk': 'Sri Lanka','sd': 'Sudan','sr': 'Suriname','sj': 'Svalbard And Jan Mayen','sz': 'Swaziland','se': 'Sweden','ch': 'Switzerland','sy': 'Syrian Arab Republic','tw': 'Taiwan, Republic Of China','tj': 'Tajikistan','tz': 'Tanzania, United Republic Of','th': 'Thailand','tl': 'Timor-Leste','tg': 'Togo','tk': 'Tokelau','to': 'Tonga','tt': 'Trinidad And Tobago','tn': 'Tunisia','tr': 'Turkey','tm': 'Turkmenistan','tc': 'Turks And Caicos Islands','tv': 'Tuvalu','ug': 'Uganda','ua': 'Ukraine','ae': 'United Arab Emirates','gb': 'United Kingdom','us': 'United States','um': 'United States Minor Outlying Islands','uy': 'Uruguay','uz': 'Uzbekistan','ve': 'Venezuela','vu': 'Vanuatu','vn': 'Viet Nam','vg': 'British Virgin Islands','vi': 'U.S. Virgin Islands','wf': 'Wallis And Futuna','ye': 'Yemen','zw': 'Zimbabwe'}

        for k,v in countries_images.items():
            if v == selected_countries:
                temp_images = 'cflags/{}.png'.format(k)
                st.text(temp_images)
                img = Image.open(os.path.join(temp_images)).convert('RGB')
                st.image(img)
        
        if st.checkbox("Selecciona las Columnas a Mostrar"):
            result_df_columns_list = result_df.columns.tolist()
            selected_columns_countries = st.multiselect("Selecciona las columnas", result_df_columns_list)
            new_df2 = df[selected_columns_countries]
            st.dataframe(new_df2.head(10))

            if st.button("Grafico del Pais"):
                st.area_chart(new_df2)
                st.pyplot()

    #Metrics
    elif  choice == "Metrica":
        st.subheader("Seccion Metrica")
        cnx = sqlite3.connect('data.db')
        mdf = pd.read_sql_query("SELECT * FROM predictiontable",cnx)
        st.dataframe(mdf)


if __name__ == "__main__":
    main()



