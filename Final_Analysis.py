from ctypes import alignment
from secrets import choice
from turtle import color, position
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
from streamlit_option_menu import option_menu
from  PIL import Image
import streamlit.components.v1 as html
from sklearn.neighbors import NearestNeighbors
import io 
import os.path
import altair as alt
import plotly.graph_objects as go
sns.set()
import ipywidgets as widgets
from ipywidgets import FloatSlider,IntSlider,interact
from sklearn.preprocessing import LabelEncoder


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


df = pd.read_csv(r"cars_engage_2022.csv")

#side navigation bar
with st.sidebar:
    selected = option_menu(
    menu_title = "Analysis",
    options= ["Customer Segment", "Car Specification", "Price Analysis", "Other Specifications","Recommendation Engine","Frequently asked questions"],
    default_index = 0,
    styles = {
        "container": {"background-color": "bisque"},
    }
    )

#replacing null value in Make with first name of model from a column named splitted model name
df.insert(loc=1,
          column='splitted_model',
          value= df["Model"].apply(lambda x: x.split()[0]))
df["Make"].fillna(df["splitted_model"], inplace = True)
#dropping null values of useful columns 
df = df.dropna(subset=['Width', 'Height', 'Wheelbase', 'Fuel_Tank_Capacity', 'Seating_Capacity', 'Torque', 'Drivetrain', 'Doors', 'Seating_Capacity', 'Number_of_Airbags','Cylinders', 'Displacement'])
#cleaning data
df['Ex-Showroom_Price'] = df["Ex-Showroom_Price"].apply(lambda x:int( x[4:].replace(',','')))
df["Height"] = df["Height"].astype(str).apply(lambda x: x.replace(' mm','')).astype(float)
df["Length"] = df["Length"].astype(str).apply(lambda x: x.replace(' mm','')).astype(float)
df["Width"] = df["Width"].astype(str).apply(lambda x: x.replace(' mm','')).astype(float)
df["Wheelbase"] = df["Wheelbase"].astype(str).apply(lambda x: x.replace(' mm','')).astype(float)
df['Fuel_Tank_Capacity'] = df['Fuel_Tank_Capacity'].astype(str).apply(lambda x: x.replace(' litres','')).astype(float)
df["Displacement"] = df["Displacement"].astype(str).apply(lambda x: x.replace(' cc','')).astype(float)
df['Number_of_Airbags'] = df['Number_of_Airbags'].fillna(0)
#dropping null values and converting it to integer
df['Doors'] = df['Doors'].astype(int)
df['Seating_Capacity'] = df['Seating_Capacity'].astype(int)
df['Number_of_Airbags'] = df['Number_of_Airbags'].astype(int)
df['Cylinders'] = df['Cylinders'].astype(int)
df['Displacement'] = df['Displacement'].astype(int)

df.loc[df.ARAI_Certified_Mileage == '9.8-10.0 km/litre','ARAI_Certified_Mileage'] = '10'
df.loc[df.ARAI_Certified_Mileage == '10kmpl km/litre','ARAI_Certified_Mileage'] = '10'
df.loc[df.ARAI_Certified_Mileage == '22.4-21.9 km/litre', 'ARAI_Certified_Mileage'] = '22'
df['ARAI_Certified_Mileage'] = df['ARAI_Certified_Mileage'].dropna().astype(str).apply(lambda x: x.replace(' km/litre','')) .astype(float) 
HP = df.Power.str.extract(r'(\d{1,4}).*').astype(int) * 0.98632
HP = HP.apply(lambda x: round(x,2))
TQ = df.Torque.dropna().str.extract(r'(\d{1,4}).*').astype(int)
TQ = TQ.apply(lambda x: round(x,2))
df.Torque = TQ
df.Power = HP
#cleaning data

#plotting graphs
if selected == "Customer Segment":
    col1, col2 = st.columns( [0.9, 0.1])
    with col1:
        #plotting graph between car count and company by dictionary
        dic = {}
        for make in df['Make'].unique():
            dic[make] = sum(df['Make']==make)
            car_statistics = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:20]
        fig = plt.figure(figsize=(20,12))
        plt.bar(range(len(car_statistics)), [val[1] for val in car_statistics], align='center')
        plt.xticks(range(len(car_statistics)), [val[0] for val in car_statistics])
        plt.xticks(fontsize =20,rotation=70)
        plt.yticks(fontsize =20)
        plt.ylabel('Number_of_cars', fontsize =20)
        plt.xlabel('Company_Name', fontsize =20)
        plt.title('Car count vs company', fontsize=40)
        plt.grid()
        st.pyplot(fig)
        st.write('The above graph represents the number of cars each companies make. It can be intepreted that Maruti Suzuki manufactures most number of cars followed by Hyundai and so on.', fontsize = 30)

        #plotting graph between car count vs price
        fig = plt.figure(figsize=(20,12))
        sns.histplot(data=df, x='Ex-Showroom_Price',bins=1000,alpha=.5, color='darkblue')
        plt.title('Histogram of cars price data',fontsize=40)
        plt.xticks(fontsize =20)
        plt.yticks(fontsize =20)
        plt.ylabel('Count', fontsize =20)
        plt.xlabel('Price', fontsize =20)
        plt.xlim(0, 5000000)
        plt.grid()
        st.pyplot(fig)
        st.write('The above graph shows car price data i.e., number of cars in the market with respect to price.', fontsize=30)


if selected == "Car Specification":
    st.header('Engine_Type')
    col1, col2 = st.columns( [0.9, 0.1])
    with col1:
        dic1 = {}
        for displacement in df['Displacement'].unique():
            dic1[displacement] = sum(df['Displacement']==displacement)
        displacement = sorted(dic1.items(), key=lambda x: x[1], reverse=True)[:15]
        #plotting bar graph for number of cars vs engine type (displacement)
        fig = plt.figure(figsize=(16,12))
        plt.bar(range(len(displacement)),[val[1] for val in displacement], align='center', color= 'green', edgecolor='black')
        plt.xticks(range(len(displacement)),[val[0] for val in displacement])
        plt.yticks(fontsize= 18)
        plt.xticks(fontsize= 18,rotation=70)
        plt.ylabel('Number_of_cars', fontsize=25)
        plt.xlabel('Displacement(in cc)', fontsize=25)
        st.pyplot(fig)
    
        st.header('Fuel Type')
        fig = plt.figure(figsize=(14,8))
        fuel_type = df['Fuel_Type'].unique()
        car_count = df['Fuel_Type'].value_counts()
        plt.plot(fuel_type, car_count, marker='o', color='red')
        plt.ylabel("Car_Count", fontsize=25)
        plt.yticks(fontsize= 18)
        plt.xticks(fontsize= 18)
        for xy in zip(fuel_type, car_count):                               
            plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') 
        plt.grid()
        st.pyplot(fig)

        fig = plt.figure(figsize=(14,8))
        tank_capacity= df['Fuel_Tank_Capacity']
        fuel_type = df['Fuel_Type']
        plt.ylabel('tank_capacity', fontsize=20)
        plt.xlabel('fuel_type', fontsize=20)
        plt.xticks(fontsize=18, rotation=300)
        plt.bar( fuel_type, tank_capacity, color = 'salmon')
        st.pyplot(fig)

        st.header('Mileage')
        fig= plt.figure(figsize=(14,8))
        car_count = df['ARAI_Certified_Mileage'].value_counts()
        plt.xlim(0 , 35)
        plt.ylim(0 , 20)
        plt.ylabel('Mileage', fontsize=20)
        plt.xlabel('car_count', fontsize=20)
        plt.yticks(fontsize= 18)
        plt.xticks(fontsize= 18)
        plt.plot(car_count,  linestyle='none', marker='o')
        st.pyplot(fig)

        st.header('Drivetrain')
        fig = plt.figure(figsize=(14,8))
        drivetrain = df["Drivetrain"].unique()
        number_of_cars = df['Drivetrain'].value_counts()
        fig = px.pie(drivetrain, values =number_of_cars, names= drivetrain)
        fig.update_layout(
        title="<b>Drivetrain relation with car counts</b>")
        st.plotly_chart(fig)

        st.header('Body Type')
        fig = plt.figure(figsize=(14,8))
        sns.countplot(data=df, y='Body_Type',alpha=.6,color='purple')
        plt.title('Cars vs car body type',fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        st.pyplot(fig)
    st.write('The following graphs gives the detailed analysis about cars. The factors which are mainly responsible for decision of users are represented in these graphs. Starting with engine type, fuel type, Mileage, users can easily analyse to know about different cars.')


if selected == 'Price Analysis':
    col1, col2 = st.columns( [0.9, 0.1])
    with col1:
        st.header('Engine Type')
        fig = plt.figure(figsize=(14,8))
        displacement = df['Displacement']
        price = df['Ex-Showroom_Price']
        plt.scatter(displacement, price, color='peru')
        plt.ylabel('Ex-Showroom_Price', fontsize=20)
        plt.xlabel('Displacement', fontsize=20)
        plt.yticks(fontsize= 18)
        plt.xticks(fontsize= 18)
        plt.grid()
        st.pyplot(fig)

        st.header('Mileage')
        fig = plt.figure(figsize=(14,8))
        price = df['Ex-Showroom_Price']
        Mileage = df['ARAI_Certified_Mileage']
        plt.ylabel('price', fontsize=20)
        plt.xlabel('Mileage', fontsize=20)
        plt.xlim(0 , 35)
        plt.yticks(fontsize= 18)
        plt.xticks(fontsize= 18)
        plt.scatter( Mileage,price, color= 'green')
        st.pyplot(fig)

        st.header('Body Type')
        fig = plt.figure(figsize=(14,8))
        price = df['Ex-Showroom_Price']
        body_type = df['Body_Type']
        plt.ylabel('price', fontsize =20)
        plt.xlabel('body_type', fontsize=20)
        plt.xticks(fontsize=18, rotation=300)
        plt.yticks(fontsize=18)
        plt.bar( body_type,price)
        st.pyplot(fig)

if selected == 'Other Specifications':
    col1, col2 = st.columns( [0.9, 0.1])
    with col1:
        #relation between components of company cars
        st.header('3D plots for some parameters')
        st.write('Between Airbags, Cylinders and Seating capacity')
        plt.figure()
        fig= px.scatter_3d(df, x='Number_of_Airbags', z='Seating_Capacity', y='Cylinders',color='Make',width=800,height=750)
        fig.update_layout(showlegend=True)
        plt.savefig('graph1.png')
        st.plotly_chart(fig)

        st.write('Between dimension of car body')
        plt.figure()
        fig= px.scatter_3d(df, x='Length', z='Width', y='Height',color='Make',width=800,height=750)
        fig.update_layout(scene = dict(
        xaxis = dict(range=[3000,6000],),
                     yaxis = dict(range=[1000,2000],),
                     zaxis = dict(range=[1200,2000],),),showlegend=True)
        plt.savefig("graph.png")
        st.plotly_chart(fig)

        st.header('Coorelation between different parameters')
        fig = plt.figure(figsize=(22,14))
        sns.heatmap(df.corr(), annot=True, fmt='.2%', cmap= "PiYG")
        plt.title('Correlation between differet variable',fontsize=20)
        plt.xticks(fontsize=18, rotation=300)
        plt.yticks(fontsize=18)
        st.pyplot(fig)

data = pd.read_csv("cleaneddata.csv")
df1=data

def RE(CSL,P,M,SC,FTC,BS,X):
	print(P,M,X)
	test = df1[["Price","ARAI_Certified_Mileage","Seating_Capacity","Fuel_Tank_Capacity","Boot_Space","Child_Safety_Locks","Type"]]
	test.fillna(0, inplace = True)
	# Create K-Nearest Neighbors
	print(test.head())
	nn = NearestNeighbors(n_neighbors=3).fit(test.values)
	print(nn.kneighbors([[P,M,SC,FTC,BS,CSL,X]]))
	res=df1.iloc[nn.kneighbors([[P,M,SC,FTC,BS,CSL,X]])[1][0]]
	print(res)
	return res

if selected=="Recommendation Engine":
	st.title("Car Recommendation & Analysis")
	form1=st.form(key = "my_form")
	form1.write("What kind of Car do you want?")
	p=form1.slider("Price",min_value=100000,max_value=1000000)
	m=form1.slider("Mileage",10,50)
	sc=form1.number_input("Seating Capacity",2,7)
	ftc=form1.slider("Fuel Tank Capacity",10,80)
	bs=form1.slider("Boot Space",20,100)
	tt=form1.selectbox('Transmission Type',
												options=["Manual","Automatic","DCT","CVT","AMT"])
	csl=form1.checkbox("Child Safety Locks", False)									
	submitted = form1.form_submit_button("Recommend")
			
	if submitted:
		x = {
			"Manual":0,"Automatic":1,"DCT":2,"CVT":3,"AMT":4
			}
		t=RE((1 if csl else 0),p,m,sc,ftc,bs,x[tt])
		print(t)
		with st.expander(label="Here is your Recommendation"):
			i=1
			for _,x in t.iterrows():
				with st.container():
					st.subheader(str(i)+") "+str(x.Make)+" "+str(x.Model))
					st.text("Body Type: "+str(x["Body_Type"]))
					st.text("Mileage: "+str(x["ARAI_Certified_Mileage"]))
					st.text("Fuel Tank Capacity: "+str(x.Fuel_Tank_Capacity) +" litres")
					st.text("Seating capacity: "+str(x.Seating_Capacity))
					i+=1

if selected=="Frequently asked questions":
    st.title('Frequenty Asked Questions'':')
    # check = st.checkbox('Yes')
    # if check:
    nested_btn=st.button('Which Fuel Type is used in most of the Automobiles?')
    if nested_btn:
        fig,ax = plt.subplots()
        sns.histplot(data=df, x='Fuel_Type',bins=5, alpha=0.6, color='#f54242')
        ax.set_title('Histogram of Fuel Type data')
        ax.set_xlabel('Fuel Type')
        st.pyplot(fig)
        st.write('* **600** Automobiles use petrol as their Fuel Type, and as we know, there are 1201 cars in the data. As a result, nearly **half** of all cars run on **_Petrol_**.')
        st.write('* The second most used Fuel Type is **Diesel**. Therefore, it can be deduced that most cars run on either **_Gas_** or **_Diesel_**.')
    nested_btn_1 = st.button('Show the range of Ex-Showroom Price')
    if nested_btn_1:
        fig,ax = plt.subplots()
        sns.histplot(data=df, x='Ex-Showroom_Price',bins=15, alpha=0.6, color='#f54242')
        ax.set_title('Histogram of Ex-Showroom Price data')
        ax.set_xlabel('Ex-Showroom Price')
        st.pyplot(fig)
        st.write('* According to the graph, the maximum car price ranges from **10 lakh** to **16 lakh**.')
    nested_btn_2 = st.button('Number of Doors in most of the Automobiles')
    if nested_btn_2:
        fig,ax = plt.subplots()
        sns.histplot(data=df, x='Doors',bins=20, alpha=0.6, color='#f54242')
        ax.set_title('Histogram of Number of Doors data')
        ax.set_xlabel('Number of Doors')
        st.pyplot(fig)
        st.write('* **800** Automobiles has **five** doors, and as we know, there are 1201 cars in the data. As a result, nearly 67% of all cars has **_Five Doors_**.')
    nested_btn_3 = st.button('Which Car Body Type is most popular?')
    if nested_btn_3:
        fig,ax = plt.subplots()
        sns.countplot(data=df, y='Body_Type', alpha=0.6, color='#f54242')
        ax.set_title('Frequency of Car Body Type data')
        ax.set_ylabel('Body Type')
        st.pyplot(fig)
        st.write('* **SUVs** are the most common car body type, followed by **Hatchbacks** and **Sedans**.')
    nested_btn_4=st.button('Relation between Company and Car Ex-Showroom Price')
    if nested_btn_4:
        df3 = df[(df['Make'].isin(['Audi', 'Aston Martin', 'Bentley', 'BMW', 'Bulgatti', 'Ferrari', 'Jaguar', 'Lamborghini', 
                                                           'Land Rover Rover', 'Lexus', 'Maserati', 'Porsche', 'Volvo']))]
        fig,ax = plt.subplots()
        sns.boxplot(data=df3, y='Make', x='Ex-Showroom_Price');
        ax.set_title('')
        ax.set_ylabel('Company')
        ax.set_xlabel('Ex-Showroom Price')
        st.pyplot(fig)
        st.write('* According to the graph, **Lamborghini, Bentley, and Ferrari** produce the most expensive cars, with prices exceeding **3 crore**.')
    nested_btn_5=st.button('What is the Basic Warranty applied to Automobiles?')
    if nested_btn_5:
        fig,ax = plt.subplots()
        sns.countplot(data=df, y='Basic_Warranty', alpha=0.6, color='#f54242')
        ax.set_title('Frequency of Warranty data')
        ax.set_ylabel('Warranty')
        st.pyplot(fig)
        st.write('* According to the graph, **2 Years/Unlimited Kms** is the most common Basic Warranty provided by the Companies')
    nested_btn_6=st.button('Correlation between the different features of Automobile')
    if nested_btn_6:
        df_r=df.copy()
        c1=['Ex-Showroom_Price', 'Displacement', 'Cylinders', 'Fuel_Tank_Capacity', 'Doors', 'ARAI_Certified_Mileage', 
            'Seating_Capacity', 'Number_of_Airbags']
        df_r=df_r[c1]
        fig,ax = plt.subplots(figsize=(15,10))
        sns.heatmap(df_r.corr(), annot=True, fmt=".2f")
        st.pyplot(fig)
        st.write('* According to the plot, the **Ex-Showroom Price, Displacement, and Cylinders** is negatively correlated with **No. of Doors, Mileage and Seating Capacity**')
        st.write('* The **Fuel Tank Capacity** is negatively correlated with **No. of Doors and Mileage**')
        st.write('* The **No. of Doors** is also negatively correlated with **No. of Airbags**')
        st.write('* The **Mileage** is also negatively correlated with the **Seating Capacity and No. of Airbags**')
        st.write('* The **Seating Capacity** is also negatively correlated with **No. of Airbags**')
    nested_btn_7=st.button('Pie Chart on the car production by different Companies')
    if nested_btn_7:
        fig= plt.figure(figsize=(15,10))
        ax = fig.subplots()
        df.Make.value_counts().plot(ax=ax, kind='pie')
        st.pyplot(fig)
        st.write('* **Maruti Suzuki** manufactures the most cars, followed by **Hyundai**, **Mahindra** and **Tata**.')