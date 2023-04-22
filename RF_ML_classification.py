# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 21:01:49 2023

@author: giannisps
"""



                         
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt

# Set up file paths for the input images and output image
#image_path = r"C:\Users\GNR\Desktop\ΕΙΚΟΝΕΣ\2\EIKONA_2.tif"

image_path = r"C:\Users\GNR\Desktop\shp\data\image.tif"
class1_path = r"C:\Users\GNR\Desktop\shp\data\kal1_data.tif"
class2_path = r"C:\Users\GNR\Desktop\shp\data\kal2_data.tif"
class3_path = r"C:\Users\GNR\Desktop\shp\data\snow_data.tif"
class4_path = r"C:\Users\GNR\Desktop\shp\data\vegetation_data.tif"
output_path = os.path.expanduser(r"C:\Users\GNR\Desktop\image_result.tif")



# Load Landsat-8 multiband image as array
with rasterio.open(image_path) as src:
    image = src.read([1, 2, 3, 4, 5, 6, 7])
    profile = src.profile

# Load training and validation data from the 4 classification images
with rasterio.open(class1_path) as src:
    class1 = src.read([1, 2, 3, 4, 5, 6, 7])
    class1_samples = class1.reshape((class1.shape[0], -1)).T
    class1_labels = np.zeros(class1_samples.shape[0])
with rasterio.open(class2_path) as src:
    class2 = src.read([1, 2, 3, 4, 5, 6, 7])
    class2_samples = class2.reshape((class2.shape[0], -1)).T
    class2_labels = np.ones(class2_samples.shape[0])
with rasterio.open(class3_path) as src:
    class3 = src.read([1, 2, 3, 4, 5, 6, 7])
    class3_samples = class3.reshape((class3.shape[0], -1)).T
    class3_labels = np.ones(class3_samples.shape[0]) * 2
with rasterio.open(class4_path) as src:
    class4 = src.read([1, 2, 3, 4, 5, 6, 7]) 
    class4_samples = class4.reshape((class4.shape[0], -1)).T
    class4_labels = np.ones(class4_samples.shape[0]) * 3

# Combine samples and labels from all classes into training and validation data
samples = np.concatenate((class1_samples, class2_samples, class3_samples, class4_samples))
labels = np.concatenate((class1_labels, class2_labels, class3_labels, class4_labels))
validation_indices = np.random.choice(labels.shape[0], size=int(labels.shape[0] * 0.3), replace=False)
training_indices = np.setdiff1d(np.arange(labels.shape[0]), validation_indices)

# Train random forest classifier using training data
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
clf.fit(samples[training_indices], labels[training_indices])

# Apply classifier to Landsat-8 image and save result
result = clf.predict(image.reshape((image.shape[0], -1)).T)
result = result.reshape(image.shape[1], image.shape[2])
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(result.astype(rasterio.uint32), 1)

# Calculate and display confusion matrix and overall accuracy
validation_labels = labels[validation_indices]
validation_samples = samples[validation_indices]
predicted_labels = clf.predict(validation_samples)
confusion_mat = confusion_matrix(validation_labels, predicted_labels)
accuracy = accuracy_score(validation_labels, predicted_labels)
print("Confusion matrix:")
print(confusion_mat)
print("Overall accuracy: {:.2f}%".format(accuracy * 100))

#display the result
plt.imshow(result)
plt.show()




