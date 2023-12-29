---
layout: page
title: Emotion Detection
description: TensorFlow, OpenCV
img: assets/img/cv-emotion-detection-architecture.jpg
importance: 3
category: computer vision
---

The first Computer Vision project that Ryan worked on is a simple emotion detection model that can detect faces by making a bounding-box surrounding the face, while making a prediction towards the emotion that is being conveyed by the face. 

This project used a method called transfer learning, which is unfreezing a neural network that has been trained on a different data beforehand, of the VGG 16 model.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cv-emotion-detection-architecture.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    VGG-16 Network Architecture ()
</div>