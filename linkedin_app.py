#### LOCAL APP
#### Predicting a LinkedIn user application

#### Import packages
import streamlit as st
import pandas as pd
import numpy as np
import plotly
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import plotly.graph_objects as go


#Code for machine learning model

s = pd.read_csv("social_media_usage.csv") 

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return(x)

ss = pd.DataFrame({
    "income": np.where(s["income"] <= 9,s["income"],np.nan),
    "education": np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    "parent": np.where(s["par"]==1,1,0),
    "married": np.where(s["marital"]==1,1,0),
    "female": np.where(s["gender"] ==2,1,0),
    "age": np.where(s["age"] <= 98, s["age"], np.nan),
    "sm_li": clean_sm(s["web1h"])  
})

ss = ss.dropna()

y = ss["sm_li"] #Target vector
x = ss[["income", "education", "parent", "married","female","age"]] #Feature set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   stratify = y,
                                                   test_size = 0.2,
                                                   random_state = 987)

lr = LogisticRegression(class_weight="balanced") #Initialize the algorithm

lr.fit(x_train, y_train) #Fit algorithm to training data

y_pred = lr.predict(x_test) #Make predictions using model on test data



#Add header to describe app
st.title("Using Machine Learning to Predict LinkedIn Users")
st.caption("Created by: Katelyn Gromala for MSBA OPIM-607")

#Import image
image = Image.open("photo.jpg")

st.image(image, use_column_width= True, caption = "Image sourced from Google")

#Set subtitle
st.header("Are you a LinkedIn User? Let's Find Out!")

#Title to select buttons
st.subheader("Please answer the following questions:")

#Income button
page_names = ["Less than $10,000", "10,000 to under $20,000", "20,000 to under $30,000", "30,000 to under $40,000", "40,000 to under $50,000", "50,000 to under $75,000", "75,000 to under $100,000", "100,000 to under $150,000", "$150,000 or more"]

page = st.radio("Select your income level (household in USD)", page_names)
st.write("**You selected:**", page)

#Create conversion for model to understand
if page == "Less than $10,000":
    a = 1
elif page == "$10,000 to under $20,000":
    a = 2
elif page == "$20,000 to under $30,000":
    a = 3
elif page == "$30,000 to under $40,000":
    a = 4
elif page == "$40,000 to under $50,000":
    a = 5
elif page == "$50,000 to under $75,000":
    a = 6
elif page == "$75,000 to under $100,000":
    a = 7
elif page == "$100,000 to under $150,000":
    a = 8
else:
    a = 9

st.write("#")

#Education button
page_names2 = ["Less than high school", "High school incomplete", "High school graduate", "Some college, no degree", "Two-year associate degree from college or university", "Bachelor's degree (e.g., BS, BA, AB)", "Some graduate school", "Post graduate/professional degree (e.g., MA, MS, MBA, PhD, MD, JD"]

page2 = st.radio("Select your highest level of education", page_names2)
st.write("**You selected:**", page2)

#Create conversion for model to understand
if page2 == "Less than high school":
    b = 1
elif page2 == "High school incomplete":
    b = 2
elif page2 == "High school graduate":
    b = 3
elif page2 == "Some college, no degree":
    b = 4
elif page2 == "Two-year associate degree from college or university":
    b = 5
elif page2 == "Bachelor's degree (e.g., BS, BA, AB)":
    b = 6
elif page2 == "Some graduate school":
    b = 7
else:
    b = 8

st.write("#")

#Parent status select
page_names3 = ["Yes", "No"]

page3 = st.radio("Are you a parent of a child under 18 living in your home?", page_names3)

st.write("**You selected:**", page3)

#Create conversion for model to understand
if page3 == "Yes":
    c = 1
else:
    c = 0

st.write("#")

#Marital status select
page_names4 = ["Married", "Not married or other"]

page4 = st.radio("What is your current marital status?", page_names4)

st.write("**You selected:**", page4)

#Create conversion for model to understand
if page4 == "Married":
    d = 1
else:
    d = 0

st.write("#")

#Gender status select
page_names5 = ["Male", "Female"]

page5 = st.radio("What is your gender?", page_names5)

st.write("**You selected:**", page5)

#Create conversion for model to understand
if page5 == "Female":
    e = 1
else:
    e = 0

st.write("#")

#Age selection
number = st.number_input("Please select your age", min_value = 16, max_value = 98)
st.write("Your age is", number)

st.write("#")

#Create button to run the model
if st.button("Click to predict!"):
    newdata1 = [a,b,c,d,e,number] #Create data for example
    predicted_class = lr.predict([newdata1]) #Predict the class
    probability = lr.predict_proba([newdata1]) #Find the probability

    st.write(f"**Predicted class: {predicted_class[0]} (1 is a user, 0 is not a user)**")
    st.write(f"**Probability you are a LinkedIn user: {probability[0][1]}**")
    

#Define probability
newdata1 = [a,b,c,d,e,number] #Create data for example
predicted_class = lr.predict([newdata1]) #Predict the class
probability = lr.predict_proba([newdata1]) #Find the probability

#### Create label (called sent) from TextBlob polarity score to use in summary below
if probability[0][1] > .50:
    label = "Yes"
else:
    label = "No"

#### Create sentiment gauge
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = probability[0][1],
    title = {'text': f"LinkedIn User? {label}"},
    gauge = {"axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, .50], "color":"red"},
                {"range": [.50, 1], "color":"lightgreen"}
            ],
            "bar":{"color":"lightgray"}}
))


st.plotly_chart(fig)