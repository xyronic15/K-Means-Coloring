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
from cluster import quantize, output_quantized, centroid_histogram, plot_bar

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

            # num clusters input
            n_clusters = app_con.slider("How many colors would you like?", min_value=2, max_value=10, value=2)

            # run process once button clicked
            if app_con.button("Get the value sketch"):
                # process the image with number of clusters
                (quant, bar) = process_img(img, n_clusters)

                app_con.dispcol1, app_con.dispcol2 = st.columns(2)

                app_con.dispcol1.image(quant)
                app_con.dispcol2.image(bar)


# helper functions
def process_img(img, n_clusters):
    image = np.array(img)
    (h, w) = image.shape[:2]

    # Convert image from RGB space to LAB
    # Do this to apply euclidean distance which s better understood
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Reshape the image into a feature vector so that kmeans can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # Apply kmeans with the number of clusters then create quantized image
    print("Fitting K-Means...")
    clt = MiniBatchKMeans(n_clusters=n_clusters)
    clt.fit(image)
    labels = clt.fit_predict(image)
    quant = quantize(clt, labels)

    # # Reshape the vectors of the images
    # # Convert to RGB
    # # Save to directory and show images
    (image, quant) = output_quantized(image, quant, h, w)

    # Build the histogram
    # Plot the colors of each cluster
    print("Building color histogram...")
    hist = centroid_histogram(clt, labels)
    print("Making bar chart...")
    bar = plot_bar(hist, clt.cluster_centers_)
    print("done")

    return (quant, bar)




if __name__ == '__main__':
    main()