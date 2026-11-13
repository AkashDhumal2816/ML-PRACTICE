import streamlit as st
import pandas as pd
import pickle
st.title("Titanic Fare Predicator")
st.subheader("Enter Passenger detail below to predict Total Fare")

## App title and description
model = pickle.load(open('artifact/model.pkl' , 'rb'))
lbl_ticket = pickle.load(open('artifact/lbl_ticket.pkl' , 'rb'))
ohe_embark = pickle.load(open('artifact/ohe_embark.pkl' , 'rb'))
ohe_sex = pickle.load(open('artifact/ohe_sex.pkl' , 'rb'))
scalar = pickle.load(open('artifact/scalar.pkl' , 'rb'))


## Collecting inputs (APP UI)
survived = st.pills("0 = NO, 1 = YES", ["0", "1"])
pclass = st.pills("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", ["1", "2", "3"])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 100)
sipsp = st.select_slider("Siblings/Spouses Abroad (Sibsp)", ["1", "2", "3", "4", "5"])
parch = st.select_slider("Parents/Spouses Abroad (Parch)", ["1", "2", "3", "4", "5"])
ticket = st.selectbox("Ticket Number" , lbl_ticket.classes_)
embarked = st.radio("Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)", ["S", "C" , "Q"])


### Prediction Logic
if st.button("Calculate Expected Fare"):
    ticket_encoded = lbl_ticket.transform([ticket])[0]
    sex_encoded = ohe_sex.transform(pd.DataFrame({"Sex":[sex]}))
    embarked_encoded = ohe_embark.transform(pd.DataFrame({"Embarked":[embarked]}))
    ### converting encoded arrray in pandas dataframe
    
    df_sex = pd.DataFrame(sex_encoded,columns=ohe_sex.get_feature_names_out())
    df_embarked = pd.DataFrame(embarked_encoded,columns=ohe_embark.get_feature_names_out())
    ###creating main dataset for remaining features
    df_main = pd.DataFrame({
        "Survived":[survived],
        "Pclass":[pclass],
        "Age":[age],
        "SibSp":[sipsp],
        "Parch":[parch],
        "Ticket":[ticket_encoded]
        })
        ##Merge all datasets
    final_df = pd.concat([df_main,df_sex,df_embarked])
    ###Scaling the final dataset
    scaled_data = scalar.transform(final_df)
    ## predicting Fare
    y_pred = model.predict(scaled_data)

    ### Display Result
    st.success(f"Expected Ticket Fare: $ {y_pred[0]:.2f}")


