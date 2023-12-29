---
layout: page
title: Inventory Optimization
description: PyTorch, Deep Q-Network
img: assets/img/12.jpg
importance: 2
category: reinforcement learning
related_publications: 
---
Major inventory control policies adopted in the supply chain industry nowadays are classic static policies. A dynamic policy that can adaptively adjust the decisions of when and how much to order based on the inventory position and future demand information would be advantageous.

In this project, an environment simulating an inventory problem for a product in a single retail store is initiated. A reinforcement learning agent will be trained on one year worth of data with the ultimate goal of achieving a higher total profit value when compared to one of the classic inventory control policies, the (s, S) policy. 

The agent will be using a Deep Q-Network, an algorithm developed by Google DeepMind in 2015 to find the optimal policy for maximizing the total profit.