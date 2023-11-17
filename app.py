import streamlit as st
from fastai.vision.all import *

def is_cat(x):
    return x[0].isupper() 

model = load_learner('model.pkl') 

categories = ('Dog', 'Cat')

def clasify_image(img):
    img = PILImage.create(img)
    pred,idx,probs = model.predict(img)
    return dict(zip(categories, map(float,probs)))


def main():
    
    st.title("Dog or Cat predictor")
    
    #The input is an image
    image = st.file_uploader("Upload an image", "jpg")
    # Display the image once it has been uploaded
    if image:
        disp = Image.open(image)
        st.image(disp, width=150)
    # Make the prediction
    if st.button("Predict", use_container_width=True):
        result = clasify_image(image)
        for key, value in result.items():
            st.progress(value, f"Probabiity that its a {key} is {value:.15f}%")
    
    #There should be examples you can pick from to put into the input interface
    images = [
        Image.open('photos/cat/001.jpg'),
        Image.open('photos/dog/006.jpg'),
        Image.open('photos/dog/007.jpg'),
        Image.open('photos/cat/003.jpg')
        
    ]
      
    col1, col2, col3, col4 = st.columns(4)
    
    col1.image(images[0], width=150)
    col2.image(images[1], width=150)
    col3.image(images[2], width=150)
    col4.image(images[3], width=150)
        
if __name__ == "__main__":
    main()