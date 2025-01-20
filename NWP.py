import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# Set page configuration to full width
st.set_page_config(page_title="Next word Preediction", page_icon="💬")

st.title(":rainbow[Next word prediction!]")
X = st.text_input("Enter text below to see the Next words.")
if st.button("predict"):
    Mdl1 =pickle.load(open(r"C:\Users\madas\Data Science 255 - Batch\Deep_Learning\RNN\model.pkl",'rb'))
    Mdl2 =pickle.load(open(r"C:\Users\madas\Data Science 255 - Batch\Deep_Learning\RNN\tk.pkl",'rb'))


    for y in range(20):
        word = Mdl2.index_word[np.argmax(Mdl1.predict(pad_sequences(Mdl2.texts_to_sequences([X]),maxlen=23)))]
        X = X+" "+word
        st.write(X)
        time.sleep(0.1)