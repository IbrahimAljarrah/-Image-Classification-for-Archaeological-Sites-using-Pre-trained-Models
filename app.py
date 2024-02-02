import streamlit as st
from keras import models
from keras.preprocessing.image import *
import numpy as np
from PIL import Image
import streamlit.components.v1 as components



st.set_page_config(page_title="Halo Jordan ", page_icon=":🗺️:", layout="wide")
img = st.file_uploader('Upload the image', type=["jpg", "png"])
if img:
    st.markdown('Upload complete!')
    st.markdown('''
        <style>
            .uploadedFile {display: none}
        <style>''',
        unsafe_allow_html=True)







class_labels = ["Ajloun", "Petra", "Roman_Theater", "Wadi_Rum", "jarash", "um_qais"]

def modl(mod):



    path=(f'{mod}.h5').format(mod)
    model = models.load_model(path)

    image = load_img(img, target_size=(256, 256))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float') / 255.0

   
    predictions = model.predict(image)


    top_class_index = predictions.argmax(axis=1)[0]
    top_label = class_labels[top_class_index]
    top_probability = predictions[0, top_class_index]

    return top_label,top_probability,mod

    
    
   

modlels=['VGG16',"EfficientNet","InceptionV3","ResNet"]
col_1 = st.sidebar.columns(1)

st.sidebar.title("Model Selector")

selected_model = st.sidebar.selectbox("Select model:", modlels,index=None,placeholder="Select the model...",)
lockupTable = {'Petra':'https://en.wikipedia.org/wiki/Petra',"Ajloun":"https://en.wikipedia.org/wiki/Ajloun",
               'Roman_Theater':'https://en.wikipedia.org/wiki/Roman_Theatre_(Amman)','Wadi_Rum':'https://en.wikipedia.org/wiki/Wadi_Rum',
               'jarash':'https://en.wikipedia.org/wiki/Jerash','um_qais':'https://en.wikipedia.org/wiki/Umm_Qais'}

label=None
probability=None
mod=None
cl1,cl2,cl3=st.columns(3)

if selected_model and img :
        label,probability,mod=modl(selected_model)


with cl1:

    if (label and probability) is not None:

        st.write(f"Top Prediction: {label} with Probability: {probability * 100:.3f}%")
        st.write(f"the model name is : {mod}")
        st.markdown("<br>"*3, unsafe_allow_html=True)
        but=st.button('read more',type="primary")
        if but:

            components.iframe(lockupTable[label],scrolling=True,width=600,height=1000,)

    
with cl3:

    if img is not None:
        image = Image.open(img)
        new_image = image.resize((600, 400))
        st.image(new_image)













