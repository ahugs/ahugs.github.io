---
layout: post
title: Lane-Following with Simple ICRL
date: 2024-12-17 11:12:00-0400
description: A real-world implementation of simple constraint inference
author: Adriana Hugessen and Florence Cloutier
tags: irl icrl neurips duckietown
categories: posts
related_posts: false

layout: distill
title: Lane-Following with Simple ICRL
description: A real-world implementation of simple constraint inference for autonomous driving
tags: irl icrl neurips duckietown
giscus_comments: true
date: 2024-12-17
featured: false
mermaid:
  enabled: true
  zoomable: true
code_diff: false
map: false
chart:
  chartjs: false
  echarts: false
  vega_lite: false
tikzjax: true
typograms: true

authors:
  - name: Adriana Hugessen
    url: "https://ahugs.github.io"
    affiliations:
      name: University of Montreal, Mila
  - name: Florence Cloutier
    affiliations:
      name: Universiy of Montreal, Mila

bibliography: 2024-12-22-lane_following.bib
---

## Simple Constraint Inference for Lane-Following

Reinforcement learning often struggles with complex tasks like autonomous driving, in part due to the difficulty of designing reward functions which properly balance objectives and constraints. A common solution is imitation learning, where the agent is trained in a supervised manner to simply mimic expert behavior. However, imitation learning can be brittle and is prone to compounding errors. A potentially more robust solution lies in IRL, which combines offline expert trajectories with real-world rollouts to learn a reward function. Traditional IRL methods, however, are challenging to train due to their reliance on bi-level adversarial objectives. Does an intermediate solution exist?

<div style="display: flex; justify-content: space-around; align-items: center; gap: 20px;">

  <div style="text-align: center;">
    <img src="../../../assets/img/diagram.jpg?raw=true" alt="Diagram" title="Diagram" style="width: 700px; height: auto;">
    <p><strong style="font-size: 14px;">Architecture Diagram</strong></p>
  </div>
</div>


In fact, in many real-world scenarios, there may exist some simple to define primary goal. For example, in autonomous driving, we can define this goal as driving speed or time-to-destination. This goal alone, however, does not explain the expert behavior, which must additionally obey the rules or "constraints" of the road. In this setting, we can consider the potentially easier inverse task of inferring _constraints_ under a _known_ reward function. Precisely, this is the domain of inverse constrained reinforcement learning (ICRL).  In their seminal paper on inverse constraint learning, Malik et al <d-cite key="malik2021inverse"></d-cite>, propose an algorithm for inferring constraints from expert data which involves solving a constrained MDP in the inner loop of a maximum-entropy IRL algorithm. This introduces a tri-level optimization, however, which may in fact be more complex to solve than the original IRL problem. In our NeurIPS 2024 paper <d-cite key="hugessen2024simplifying"></d-cite>, we propose a simple reduction of the ICRL problem to an IRL problem, which is equivalent under a broad class of constraint functions, and propose some simple-to-implement practical modifications for the constraint inference case. We show that this can perform as well or better than ICRL methods that explicitly solve a constrained MDP on simulated MuJoCo environments <d-cite key="liu2022benchmarking"></d-cite>. But does it work in the real world?

In this blog post, we will try it out on the autonomous driving task in the Duckietown environment. In our simplified setting for autonomous driving, we consider the single constraint of lane following, which means that the driver is required to stay in the right-hand lane at all times and incurs a cost penalty when outside the lane (either off-road or in the opposite lane). 

### Background

As shown in prior work <d-cite key="swamy2023inverse"></d-cite>, the inverse RL problem can be cast as a two-player zero-sum game
\begin{equation}
    \newcommand{\expectation}{\mathbb{E}}
    \min_{\pi \in \Pi} \sup_{f \in \mathcal{F}} J(\pi_{E},f) - J(\pi, f)
\end{equation}
where $$\mathcal{F}$$ is convex and compact and $$J(\pi, f) = \expectation_{\pi} \left[\sum_{t=0}^T f(s_t,a_t)\right]$$.

Similarly, the constrained RL problem with a single constraint can also be cast as a two-player zero-sum game
\begin{equation}
    \min_{\pi \in \Pi}\max_{\lambda \ge 0} -J(\pi, r) + \lambda (J(\pi,c) - \delta).
\end{equation}

Finally, <d-cite key="kim2023learning"></d-cite> show that for inverse constrained RL, these two games can be combined into a three-player game of the following form (where $$r$$ is a given reward function in a class $$\mathcal{F}_r$$)

\begin{equation}
    \sup_{c \in \mathcal{F_c}}\max_{\lambda > 0} \min_{\pi \in \Pi}  J(\pi_{E},r - \lambda c) - J(\pi, r - \lambda c)
    \label{eq:tri}
\end{equation}

Practically speaking, solving this game involves running IRL where the inner loop optimization solves a constrained MDP using a Lagrangian version of RL.


### Method

In our paper, we demonstrate that Equation \ref{eq:tri} can be reduced to a bi-level optimization, equivalent to performing IRL on a reward "correction" term. 

\begin{equation}\label{eq:icrl-bilevel}
   \max_{c\in\mathcal{F_c}}\min_{\pi\in\Pi} J(\pi_E, r - c) - J(\pi, r - c).
\end{equation}

where $$r$$ is the known reward and $$c$$ is the learned constraint.


### Results

The following graphs compare 4 models trained differently. Partial IRL is our model, where we start by using RL with a base reward function and learn the residual of the reward function with IRL. Full IRL is a model trained from scratch using IRL. RL velocity is a model trained with RL using a reward function based on velocity. The last model, RL human is a model trained with RL with a human designed reward function.

<div style="display: flex; justify-content: space-around; align-items: center; gap: 20px;">

  <div style="text-align: center;">
    <img src="../../../assets/img/deviation_heading.png?raw=true" alt="Image 1" title="Image 1 Title" style="width: 220px; height: auto;">
    <p><strong style="font-size: 14px;">Figure 1: Deviation Heading &darr;</strong></p>
  </div>

  <div style="text-align: center;">
    <img src="../../../assets/img/distance_traveled.png?raw=true" alt="Image 2" title="Image 2 Title" style="width: 220px; height: auto;">
    <p><strong style="font-size: 14px;">Figure 2: Distance Traveled &uarr;</strong></p>
  </div>

  <div style="text-align: center;">
    <img src="../../../assets/img/time_in_lane.png?raw=true" alt="Image 3" title="Image 3 Title" style="width: 220px; height: auto;">
    <p><strong style="font-size: 14px;">Figure 3: % Time in Lane &uarr;</strong></p>
  </div>

</div>

Figure 1 measures how much the vehicle deviates from the intended heading direction (lower values are better). The results show that the model RL Velocity performs the best with the smallest deviation and the model Partial IRL is close second. Full IRL and RL Human show significant deviation, indicating weaker performance.

Figure 2 represents how far the model travels (higher values are better, implying better control). Partial IRL achieves the greatest distance traveled, followed closely by RL Velocity. Full IRL and RL Human perform poorly, with minimal distance covered, indicating less effective control.

Figure 3 shows the percentage of time the model stays within the lane (higher values are better). RL Human performs the best, staying in the lane the most. Partial IRL also achieves high performance, slightly below RL Human. Full IRL and RL Velocity perform worse, spending less time in the lane.

Overall, the Partial IRL is the most balanced model, performing well across all metrics. The RL Velocity model has good performance as well in these metrics.

Next, we show simulation experiments of all 4 models in the same environment.

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; text-align: center;">

  <div>
    <video controls style="width: 100%; height: auto;">
      <source src="../../../assets/video/partial_irl_sim_trim.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><strong style="font-size: 14px;">Partial IRL</strong></p>
  </div>

  <div>
    <video controls style="width: 100%; height: auto;">
      <source src="../../../assets/video/full_irl_sim.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><strong style="font-size: 14px;">Full IRL</strong></p>
  </div>

  <div>
    <video controls style="width: 100%; height: auto;">
      <source src="../../../assets/video/rl_velocity_sim_trim.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><strong style="font-size: 14px;">RL Velocity</strong></p>
  </div>

  <div>
    <video controls style="width: 100%; height: auto;">
      <source src="../../../assets/video/rl_human_sim_trim.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><strong style="font-size: 14px;">RL Human</strong></p>
  </div>

</div>

As expected, the videos of the Full IRL and the RL Human model perform poorly, which is coherent with the metrics evaluated in the Figures above. If we compare the videos from Partial IRL and RL Velocity, we can see that both perform well, but Partial IRL stays in the right lane whereas RL Velocity changes lanes at the beginning of the simulation. This behavior can be explained by the fact that Partial IRL captures more complex behavior from the expert trajectories it is trained on to learn the residual of the reward function, justifying its usefulness.

In the following videos, we show more examples of Partial IRL in simulation.

<div style="display: flex; justify-content: space-around; align-items: center; gap: 20px;">

  <div style="text-align: center;">
    <img src="../../../assets/video/partial_irl_sim_1.gif" alt="GIF 1" style="width: 200px; height: auto;">
    <p><strong style="font-size: 14px;">GIF 1 Title</strong></p>
  </div>

  <div style="text-align: center;">
    <img src="../../../assets/video/partial_irl_sim_2.gif" alt="GIF 2" style="width: 200px; height: auto;">
    <p><strong style="font-size: 14px;">GIF 2 Title</strong></p>
  </div>

  <div style="text-align: center;">
    <img src="../../../assets/video/partial_irl_sim_3.gif" alt="GIF 3" style="width: 200px; height: auto;">
    <p><strong style="font-size: 14px;">GIF 3 Title</strong></p>
  </div>

</div>

From all 3 videos, we see good performance of the model in the different lanes.

Next, we show the model running on the real duckiebot in a similar environment as in simulation.

<div style="display: flex; justify-content: space-around; align-items: center; gap: 20px;">

  <div style="text-align: center;">
    <video controls style="width: 100%; max-width: 400px; height: auto;">
      <source src="../../../assets/video/partial_irl_real_1.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><strong style="font-size: 14px;">Video 1 Title</strong></p>
  </div>

  <div style="text-align: center;">
    <video controls style="width: 100%; max-width: 400px; height: auto;">
      <source src="../../../assets/video/partial_irl_real_2.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><strong style="font-size: 14px;">Video 2 Title</strong></p>
  </div>

</div>


From the performance in the videos, we can see that the sim2real transfer is pretty smooth. The only adjustment made on the real robot was changing the velocity of the robot. 

### Conclusion

In this work, we addressed the lane-following problem in autonomous driving using a simplified inverse reinforcement learning (IRL) approach combined with reinforcement learning (RL). By learning the residual component of the reward function using expert trajectories, Partial IRL demonstrated a clear advantage over traditional RL and full IRL models.

Our results showed that Partial IRL achieves a balance across key metrics—deviation heading, distance traveled, and time spent in lane—outperforming RL models that rely on predefined or simplistic reward functions. The model not only generalizes well in simulations but also transfers smoothly to a real Duckiebot, with only minor velocity adjustments.

Notably, Partial IRL captures more complex behaviors, such as lane adherence, which plain RL methods struggle to learn. This highlights the usefulness of incorporating expert knowledge to refine reward functions, offering a practical and computationally efficient solution to challenges in autonomous driving.




