import streamlit as st
import torch
from PIL import Image

from helper import return_model, process_input

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    user_image = Image.open(uploaded_file).convert('RGB')

    # Display the uploaded image
    st.image(user_image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Processing....")
#user_image=Image.open('').convert('RGB') # line used for testing
    user_input = process_input(image=user_image)
    loaded_model = return_model()  # model_name()

    MODEL_SAVE_PATH = 'model_state_dict'

    loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH,map_location=torch.device('cpu'))) # map_location can change depending on where the model is running
    loaded_model.eval()
    with torch.inference_mode():
        pred_logit = loaded_model(user_input.unsqueeze(dim=0))
        pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
        #print(pred_prob.shape)
        pred_classes = pred_prob.argmax(dim=0)
        if pred_classes == 1:
            st.write('It is a dog')
        else:
            st.write('It is a cat')


