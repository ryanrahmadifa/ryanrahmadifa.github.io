---
layout: page
title: Image Generation
description: TensorFlow, TensorBoard
img: assets/img/gan-image-generation-thumbnail.jpg
importance: 3
category: general adversarial networks
---

Paper implementation of deep learning through a Deep Convolutional General Adversarial Network for image generation: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. 

Trained and implemented on the LFW Dataset, a dataset containing hundreds of thousands of celebrity faces, the agent will then recreate fake pictures similar to the real photographs.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/gan-image-generation-thumbnail.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Results of the image generation. (Top: Fake results; compared to Bottom: Real results.)
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/gan-image-generation-architecture.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Network Architecture (Racford, Alec et al.)
</div>
