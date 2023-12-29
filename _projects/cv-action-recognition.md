---
layout: page
title: Action Recognition
description: Mediapipe, Tensorflow, OpenCV, PyAutoGUI
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

Integrated the model with OpenCV and PyAutoGUI which enables enabling/disabling mouse cursor movement controls using action recognition and landmark tracking which translates the index finger joint to mouse cursor movements. 

The libraries used are Pandas, NumPy, Mediapipe, Tensorflow, and OpenCV

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.html path="https://youtu.be/zUQLOD_fYF0" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Play this video to see the demonstration of the application! The model is predicting in real-time and translating the movements into actions based on the classification being done: The movement, clicking, and scrolling of the mouse is being controlled with the actions being predicted by the model.
</div>



<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cv-action-recognition-network.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This is the neural network's architecture that is being trained on the data, it consists of four LSTM layers, three fully connected layers, and a softmax function at the end to map the data into the predicted action.
</div>
