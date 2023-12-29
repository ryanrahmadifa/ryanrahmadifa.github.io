---
layout: page
title: Inventory Optimization
description: PyTorch, Deep Q-Network
img: assets/img/rl-inventory-optimization-thumbnail.jpg
importance: 2
category: reinforcement learning
related_publications: 
---
Major inventory control policies adopted in the supply chain industry nowadays are classic static policies. A dynamic policy that can adaptively adjust the decisions of when and how much to order based on the inventory position and future demand information would be advantageous.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/rl-inventory-optimization-thumbnail.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The Deep Q-Network agent performing better than the classic (s, S) policy.
</div>

In this project, an environment simulating an inventory problem for a product in a single retail store is initiated. A reinforcement learning agent will be trained on one year worth of data with the ultimate goal of achieving a higher total profit value when compared to one of the classic inventory control policies, the (s, S) policy. 

The agent will be using a Deep Q-Network, an algorithm developed by Google DeepMind in 2015 to find the optimal policy for maximizing the total profit.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/rl-inventory-optimization-results.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Tensorboard log of each iteration of the model done by Ryan.
</div>

As one of the first deep learning projects being done which can directly be compared to a subject in Industrial Engineering (Production Planning & Control, Operations Research), Ryan was eager to do multiple experimentations towards the hyperparameters were done to ensure the best decision-making that could done by the agent. 

A post on this website will also be made sequentially after the completion of this project, the post will discuss some actions taken by the agent at given states and finding out the reasoning behind said actions.
