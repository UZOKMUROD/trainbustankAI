import streamlit as st
from fastai.vision.all import *
import pathlib
import numpy as np
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

plt=platform.system()
if plt == "Linux" : pathlib.WindowsPath=pathlib.PosixPath

from st_pages import Page, Section, show_pages, add_page_title
import streamlit as st



# Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be

show_pages(
    [
        Page("main.py", "Home", "🏠"),
        Page("page2.py", "Dastur kodi", ),
        Page("rezume.py ","Rezume", icon="💪"),
        
        
    ]
)

st.title("Assalomu alaykum siz bu yerda Bus, Tank, Train rasmlarini ajrata oladigan AI sinab ko'rishingiz mumkin va rasm yuklashda pastdagi 3 turgagi rasmlarni yuklang")

col1, col2, col3 = st.columns(3)

with col1:
   st.header("Train")
   st.image('111.webp', caption='')

with col2:
   st.header("Bus")
   st.image("4.jfif")

with col3:
   st.header("Tank")
   st.image("1.jfif")

img=st.file_uploader("Rasm yuklash uchun joy", type=["png","jpg","jpeg","gif", "svg", "webp","jfif"])
if img:
    st.image(img)

    imgs=PILImage.create(img)

    model=load_learner("Transport.pkl")

    pred, pred_id,prods=model.predict(imgs)

    st.success(f"Aniqlangan rasm nomi: {pred}")
    st.info(f"Rasm aniqliyligi : {prods[pred_id]*100:.1f}%")
    

    
