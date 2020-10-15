# import all the packages
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import sys #might not be needed
import argparse
import os
import cv2
import numpy as np

# Arguments parser
ap = argparse.ArgumentParser()
# Parse the arguments using argument parser
# Parse it into image file name and the number of cluster for quantization
ap.add_argument("-i", "--image", required=True, help="Image file name in images path")
ap.add_argument("-c", "--clusters", required=True, type=int, help="# of clusters for quantization")
args = vars(ap.parse_args())

def main():

    # Load the image then grab its width and height
    img_path = os.path.join(os.getcwd(), "images", args["image"])
    image = cv2.imread(img_path)
    (h, w) = image.shape[:2]

    # Convert image from RGB space to LAB
    # Do this to apply euclidean distance which s better understood
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Reshape the image into a feature vector so that kmeans can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # Apply kmeans with the number of clusters then create quantized image
    print("Fitting K-Means...")
    clt = MiniBatchKMeans(n_clusters=args["clusters"])
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

    # Save the image and bar plot to the output folder
    save(quant, bar)

    # display the image
    # Display the bar chart
    # wait for keypress to close 
    print("Displaying original and quantized images...")
    cv2.imshow("Quantized and Original Image", np.hstack([image,quant]))
    print("Displaying bar chart")
    cv2.imshow("Bar chart", bar)
    cv2.waitKey(0)

    

####### Quantization helper functions ##########

def quantize(clt, labels):
    # Get the labels and ten make the quantized image
    quant = clt.cluster_centers_.astype("uint8")[labels]

    return quant

def output_quantized(image, quant, height, width):
    # Reshape the vectors of the quant and original to images
    print("Reshaping images...")
    quant = quant.reshape((height, width, 3))
    image = image.reshape((height, width, 3))

    # Convert from LAB back to RGB
    print("Converting to RGB...") 
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

    return (image, quant)

####### Quantization helper functions ##########

####### Color clustering helper functions ##########

def centroid_histogram(clt, labels):
    # Get the # of clusters and make a histogram
    # Based on the amount of pixels that are assigned to each cluster
    num_labels = np.arange(0, len(np.unique(labels))+1)
    (hist, _) = np.histogram(labels, num_labels) # Assign bin_edges to temp variable so hist is not tuple

    # normalize the histogram so that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_bar(hist, centroids):
    # initilize the bar chart representing relative frequeny of each color
    # 50 is the height of bar, 300 is length of bar
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0 # initilize the starting x coordinate for the first color

    # Iterate through each color and the percentageof each color
    for (percent, color) in zip(hist, centroids):
        # Plot the rectangle for each color with length based on relative frequency
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX #build the next rectangle after the end of the last

    # Convert the bar from LAB to BGR
    bar = cv2.cvtColor(bar, cv2.COLOR_LAB2BGR)

    return bar

####### Color clustering helper functions ##########

####### Save function

def save(quant, bar):

    # Get the directory of the output folder
    # Make the filename for the quantized image and the bar chart
    # Change the directory
    # Save the quantized image and bar chart in the output folder
    output_dir = os.path.join(os.getcwd(), "output")
    img_name = args["image"]
    quant_file = f"quantized_{img_name}.jpg"
    bar_file = f"{img_name}_bar_chart.jpg"
    os.chdir(output_dir)
    print("Saving image...")
    if not cv2.imwrite(quant_file, quant):
        raise Exception("Could not write image")
    print("Saving bar chart...")
    if not cv2.imwrite(bar_file, bar):
        raise Exception("Could not write chart")
    

if __name__ == '__main__':
    main()
    