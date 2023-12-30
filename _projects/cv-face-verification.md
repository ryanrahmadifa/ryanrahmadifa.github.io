---
layout: page
title: Face Verification
description: PyTorch, OpenCV
img: assets/img/cv-face-verification-network.png
importance: 2
category: computer vision
giscus_comments: false
---

Using the LFW dataset containing 13000 face photographs, built a face verification system with Tensorflow based on the model proposed in the paper "FaceNet: A Unified Embedding for Face Recognition and Clustering" by researchers from Google Inc. 

Implemented a sequence of inception layers consisting of three 2D convolutional layers each, all of which are based on the model proposed from the paper. 

Implemented a model trained on personal face photographs to a camera using OpenCV and tested real-time face verification with 100% precision and 80% recall scores.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cv-face-verification-network.png" title="FaceNet Architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    FaceNet Architecture Illustrated by Analytics Vidhya
</div>
This is the Architecture for the FaceNet network. Don't worry if you're feeling overwhelmed, I, too, was overwhelmed at first. But looking at it closely, the idea of the Inception network is the use of inception modules. 

Which consist of multiple convolutional filters of different sizes applied to the input simultaneously. This allows the network to capture features at different spatial scales, helping it recognize both fine and coarse details in the input.