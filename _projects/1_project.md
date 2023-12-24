---
layout: page
title: Action Recognition
description: Mediapipe, Tensorflow, OpenCV
img: assets/img/12.jpg
importance: 1
category: computer vision
related_publications: 
---
For this project, Ryan

Mapped a set of landmarks consisting of 30 frames of movement for eight different actions. Used mediapipe for the tracking motions of palm joints,
the landmarks were converted into NumPy arrays which will be used for training the model.

Built a model using Tensorflow, based on Long short-term memory (LSTM) layers with ReLU activation functions and dense layers with a softmax
activation function on the last layer for mapping the last 30 frames of data gathered to one of eight actions.

Accuracy tests using confusion matrix stated that the model had 95% accuracy when trained on and given 30 frames of data for every prediction.

Integrated the model with OpenCV and PyAutoGUI which enables enabling/disabling mouse cursor movement controls using action recognition and
landmark tracking which translates the index finger joint to mouse cursor movements

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The libraries used are Pandas, NumPy, Mediapipe, Tensorflow, and OpenCV
</div>


You can also put regular text between your rows of images.
Say you wanted to write a little bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, *bled* for your project, and then... you reveal its glory in the next row of images.


<div class="row justify-content-sm-center">
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>
