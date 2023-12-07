import streamlit as st
from PIL import Image
from clf import predict
import io

# Suppress file uploader encoding warning


st.title("Image Classification App")
st.write("")

file_up = st.file_uploader("Upload an image", type="jpg")
text_io = io.TextIOWrapper(file_up)

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
