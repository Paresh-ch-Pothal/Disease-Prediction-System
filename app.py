import streamlit as st
import pandas as pd
import numpy as np
import pickle

# st.set_page_config(layout="wide")

pipe=pickle.load(open("DiseasePredictionModel.pkl","rb"))
df=pd.read_csv("cleaned.csv")

leng=df.shape[0]
re=[]
for i in range(leng):
    li=[word.replace('_', ' ') for word in df.iloc[i,2].split()]
    for j in li:
        if j not in re:
            re.append(j)





st.title("Disease Prediction System")
st.markdown("---")
opt=st.multiselect("Enter the Symptoms you have",options=re)
btn=st.button("Predict")
if btn:
    li = []
    for i in opt:
        li.append(i.replace(" ", "_"))
    y = " ".join(li)

    ypred = pipe.predict_proba([y])

    # Get the top 5 disease indices based on probabilities
    top_5_indices = np.argsort(ypred[0])[::-1][:1]

    # Create a list of top 5 diseases with their probabilities
    top_5_diseases = [(pipe.classes_[i], ypred[0][i]) for i in top_5_indices]
    print(top_5_diseases)
    table_data=[]
    for rank, (disease, prob) in enumerate(top_5_diseases, 1):
        disease_info = df[df['Disease'] == disease].iloc[0]  # Fetch the row corresponding to the disease
        description = disease_info['Description']
        precautions = disease_info['Precautions']
        st.subheader("Disease")
        st.text(disease)
        st.subheader("Description")
        st.text(description)
        st.subheader("Precautions")
        st.text(precautions)
        # table_data.append([disease , description , precautions])
    # table_df=pd.DataFrame(table_data,columns=["Disease","Description","Precaution"])
    # st.table(table_df)



