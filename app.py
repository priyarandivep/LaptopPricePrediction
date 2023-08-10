import streamlit as st
import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")
import pickle

with open("D:\\endtoend\\Laptop Price Prediction\\fle\\ct.pkl", "rb") as file:
    ct= pickle.load(file)

with open("D:\\endtoend\\Laptop Price Prediction\\fle\\model.pkl", "rb") as md:
    model= pickle.load(md)
with open("D:\\endtoend\\Laptop Price Prediction\\fle\\df.pkl", "rb") as d:
    df= pickle.load(d)


def main():
    st.title("Laptop Price Prediction")
    left, right= st.columns(2)
    company = left.selectbox('Brand',df['Company'].unique())

    # type of laptop
    type = right.selectbox('Type',df['TypeName'].unique())

    #ram
    Ram= left.selectbox("Ram (in GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])

    # operating system
    os = right.selectbox('OS',df['OpSys'].unique())

    # Weight
    Weight= left.number_input("Weight of the Laptop")
    # Touchscreen
    Touchscreen =right.selectbox('touchscreen',["Yes", "No"])
    if Touchscreen =="Yes":
        Touchscreen= 1
    else:
        Touchscreen= 0


    # IPS
    ips = left.selectbox('IPS',['No','Yes'])
    if ips =="Yes":
        ips= 1
    else:
        ips= 0

    # screen size
    screen_size = right.number_input('Screen Size')

    #resolution
    resolution = left.selectbox('Screen Resolution',['1920x1080','1366x768',
                                               '1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
    #cpu
    cpu = right.selectbox('CPU',df['cpu Brand'].unique())
    #hdd
    hdd = left.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
    #ssd
    ssd = right.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
    #gpu
    gpu = left.selectbox('GPU',df['Gpu Brand'].unique())
    
    Predict_Button= st.button("Predict")
    if Predict_Button:
        ppi =None
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
        
        a=pd.DataFrame([{"Company":company, "TypeName":type, "Ram":Ram,"OpSys": os, "Weight":Weight, "Touchscreen":Touchscreen,
                        "Ips":ips,
                    "ppi":ppi,"cpu Brand": cpu,
        "HDD": hdd, "SSD": ssd, "Gpu Brand": gpu}])
        a= ct.transform(a)
        result=model.predict(a)
        st.title("The predicted price of this configuration is " + str(int(np.exp(result[0]))))




if __name__== "__main__":
    main()
