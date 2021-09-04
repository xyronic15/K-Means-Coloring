# web app package
import streamlit as st
# image processing packages
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import argparse
import os
import cv2
from PIL import Image
import numpy as np
import base64
from cluster import quantize, output_quantized, centroid_histogram, plot_bar
import time

st.set_page_config("wide")

# sidebar
# st.sidebar.header("")
# st.sidebar.text("(Description of application)")
# st.sidebar.subheader("Original image")
# st.sidebar.image('images/sunflower.jpg')
# st.sidebar.subheader("Valued Sketch")
# st.sidebar.image('output/quantized_sunflower.jpg.jpg')
# st.sidebar.subheader("Histogram of dominant colors")
# st.sidebar.image('output/sunflower.jpg_bar_chart.jpg')

about = st.container()
app_con = st.container()

def main():

    nav = st.sidebar.radio("", ['About', 'App'])

    if nav == 'About':
        # about.write("This is the about page")
        about.header("Welcome to my value sketching app")
        about.text("(Description of application)")
        about.subheader("Original image")
        about.image('images/sunflower.jpg')
        # about.subheader("Valued Sketch")
        # about.image('output/quantized_sunflower.jpg.jpg')
        # about.subheader("Histogram of dominant colors")
        # about.image('output/sunflower.jpg_bar_chart.jpg')

        about.col1, about.col2 = st.columns(2)

        about.col1.subheader("Valued Sketch")
        about.col1.image('output/quantized_sunflower.jpg.jpg')

        about.col2.subheader("Histogram of dominant colors")
        about.col2.image('output/sunflower.jpg_bar_chart.jpg')

    elif nav == 'App':
        app_con.write("This is the application")

        image_file = app_con.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        if image_file is not None:
            
            img = Image.open(image_file)
            app_con.image(img,width=250)
            filename = os.path.splitext(image_file.name)[0]
            file_details = {"Filename":filename,"FileType":image_file.type,"FileSize":image_file.size}
            app_con.write(file_details)

            # num clusters input
            n_clusters = app_con.slider("How many colors would you like?", min_value=2, max_value=10, value=2)

            # run process once button clicked
            if app_con.button("Get the value sketch"):

                progress = app_con.progress(0)

                # process the image with number of clusters
                (quant, bar) = process_img(img, n_clusters, progress)

                app_con.dispcol1, app_con.dispcol2 = st.columns(2)
                app_con.downloadcol1, app_con.downloadcol2 = st.columns(2)

                final_quant = cv2.cvtColor(quant, cv2.COLOR_BGR2RGB)
                final_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)

                progress.progress(80)

                app_con.dispcol1.image(quant)
                app_con.downloadcol1.markdown(download_img(final_quant, filename, 'quant'), unsafe_allow_html=True)
                app_con.dispcol2.image(bar)
                app_con.downloadcol2.markdown(download_img(final_bar, filename, 'color_bar'), unsafe_allow_html=True)

                progress.progress(100)
                progress.empty()




# helper functions
def process_img(img, n_clusters, progress):

    image = np.array(img)
    (h, w) = image.shape[:2]

    # Convert image from RGB space to LAB
    # Do this to apply euclidean distance which s better understood
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Reshape the image into a feature vector so that kmeans can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    progress.progress(10)

    # Apply kmeans with the number of clusters then create quantized image
    print("Fitting K-Means...")
    clt = MiniBatchKMeans(n_clusters=n_clusters)
    clt.fit(image)
    labels = clt.fit_predict(image)
    quant = quantize(clt, labels)

    progress.progress(40)

    # # Reshape the vectors of the images
    # # Convert to RGB
    # # Save to directory and show images
    (image, quant) = output_quantized(image, quant, h, w)

    progress.progress(60)

    # Build the histogram
    # Plot the colors of each cluster
    print("Building color histogram...")
    hist = centroid_histogram(clt, labels)
    print("Making bar chart...")
    bar = plot_bar(hist, clt.cluster_centers_)
    print("done")

    progress.progress(70)

    return (quant, bar)

def download_img(img, filename, piece):
    b64 = base64.b64encode(cv2.imencode('.jpeg', img)[1]).decode()
    new_filename = "{}_{}.{}".format(filename, piece, 'jpeg')
    href = f'<a href="data:file/jpeg;base64,{b64}" id="{piece}" download="{new_filename}">Download</a>'
    # buffered = BytesIO()
    # img.save(buffered, format="JPEG")
    # img_str = base64.b64encode(buffered.getvalue()).decode()
    # href =  f'<a href="data:file/jpeg;base64,{img_str}" download="{filename}">Download</a>'
    custom_css = f""" 
        <style>
            #{piece} {{
                background-color: rgb(0, 0, 0);
                color: rgb(255, 255, 255);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;

            }} 
            #{piece}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{piece}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """
    return custom_css + href

    # st.markdown(href, unsafe_allow_html=True)




if __name__ == '__main__':
    main()