import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

DIR = './'

st.set_page_config(page_title='SPINtelligent Vehicle Recommendation', layout='centered')

st.title('SPINtelligent Vehicle Recommendation')

cosine = pd.read_csv(DIR + 'cosine_df.csv', index_col='Unnamed: 0')

st.sidebar.image(DIR + 'spincar-logo.png')
image = st.sidebar.file_uploader('Upload an image:', type='jpg')

if image:
    st.sidebar.image(image)

    top_k = tf.math.top_k(cosine[image.name], 5)

    imgs = [DIR + 'Data/' + file for file in cosine.iloc[top_k.indices].index if file != image.name]

    if st.sidebar.button('Recommend'):

        idx = 0
        for _ in range(len(imgs)-1):
            cols = st.columns(2)
            if idx < len(imgs):
                cols[0].image(imgs[idx])
                idx += 1

            if idx < len(imgs):
                cols[1].image(imgs[idx])
                idx += 1

            # if idx < len(imgs)-1:
            #     cols[2].image(imgs[idx])
            #     idx += 1
            else:
                break
