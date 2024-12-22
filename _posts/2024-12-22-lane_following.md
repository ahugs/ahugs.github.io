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

In this blog post, we will try it out on an autonomous driving task in the <a href="https://duckietown.com/">Duckietown</a> environment.  

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

In our paper <d-cite key="hugessen2024simplifying"></d-cite>, we demonstrate that Equation \ref{eq:tri} can be reduced to a bi-level optimization, equivalent to performing IRL on a reward "correction" term. 

\begin{equation}\label{eq:icrl-bilevel}
   \max_{c\in\mathcal{F_c}}\min_{\pi\in\Pi} J(\pi_E, r - c) - J(\pi, r - c).
\end{equation}

where $$r$$ is the known reward and $$c$$ is the learned constraint. This simplification holds as long as the constraint class $$\mathcal{F_c}$$ is closed under scalar positive multiplication.

Practically speaking, this can be implemented by adapting any adversarial IRL algorithm such that the policy learner in the inner loop receives the true reward from the environment adjusted additively by the learnt correction term. The main adjustment from IRL is that the constraint class must be closed to positive scalar multiplicatoin, which excludes the bounded activation functions like sigmoid employed in previous ICRL methods <d-cite key="malik2021inverse"></d-cite><d-cite key="liu2022benchmarking"></d-cite>. We instead use a linear activation function clipped to the positive range. We propose using soft clipping with a leaky ReLU activation to reduce issues with disappearing gradients and adding an L2 regularizer on the expected cost of the expert trajectories in the discriminator loss in order to prevent the cost function values from exploding. Hence, the discriminator loss becomes:

\begin{equation}
    \mathcal{L}(\pi, c) = J(\pi_E, r-c) - J(\pi, r-c) + \expectation_{s,a \sim \tau_E}{\left[c(s,a)^2\right]},
\end{equation}


For more details, including additional possible modifications (not used in these experiments) for adapting IRL to ICRL, please see our paper <d-cite key="hugessen2024simplifying"></d-cite>. 

### Implementation Details for Lane-Following
In the autonomous driving experiments, we employ the method described above along with the following specific implementation details. All of our code and instructions for reproducing the experiments are available in our <a href="https://github.com/ahugsduckietown-irl">github</a>. 


#### Environment
All experiments are conducted within the <a href="https://duckietown.com/">Duckietown</a> ecosystem, a robotics learning environment environment for autonomous driving. In our simplified setting for autonomous driving, we consider the single constraint of lane following, which means that the driver is required to stay in the right-hand lane at all times and incurs a cost penalty when outside the lane.

* __Real-world environment__: The real-world setup consists of the Duckiebot and Duckietown. The Duckiebot is a small mobile-robot powered by differential drive. It is equipped with a front-facing camera and onboard NVIDIA Jetson Nano.  Duckietown is a modularly assembled track consisting of road tiles (straight, curved, intersections) that can be configured into different topologies. In our experiments, we use a small loop. For more details on the evironment, see <a href="https://duckietown.com/platform/">here</a>.
* __Simulated Environment__: <a href="https://github.com/duckietown/gym-duckietown/tree/master">Gym-Duckietown</a> is a simulation platform with a Gym interface built to simulate the dynamics of the Duckietown ecosystem. 


#### Expert Trajectories:
Expert trajecotories are generated by a pure pursuit policy <d-cite key="wallace1985first"></d-cite>, an implementation for which is provided in the Duckietown simulator. The pure pursuit policy selects a heading angle for the robot based on a circle that is tangent to the current heading and intersects with the robot's current position and a lookahead point at some distance $L$ on the center curve of the lane. It uses priveledged information available in the simulator, namely the global coordinates of the robot and the lanes of the road. 

#### Algorithm
* __RL Algorithm__: As the policy optimizer, we use DrQ-v2 <d-cite key="yarats2021mastering"></d-cite>, an RL-method developed for continuous control in image-based environments. DrQ-v2 is an extension of the Deep Deterministic Policy Gradient (DDPG) algorithm <d-cite key="lillicrap2015continuous"></d-cite> which adds data augmentation to images sampled from the replay buffer. DDPG is an off-policy actor-critic method that jointly learns a Q-function critic network $$Q_{\theta}$$ and a deterministic policy network $$\pi_{\phi}$$. In DrQ-v2, a CNN-encoder is used to pre-process the images which are then passed to separate actor and critic heads (gradients are not passed through to the encoder from the actor). In our implementation, we also re-use this encoder with a stop-gradient to pre-process the observations sent to the learned constraint function
* __Reward function__: We use a simple reward function of forward velocity as the primary reward. This simply rewards the agent for moving forward. Additionally, the episode ends in the simulator when the agent drives onto undrivable areas, so that the agent is incentivized to stay on the road in order to maximize episode returns. 
* __Sim2Real__: The policy is trained entirely within the simulator and then transferred to the real-world robot directly without additional fine-tuning. In the real-world experiments, inference is performed on a laptop and actions are then transfmitted to the robot over the network (no onboard computation in performed). In these experiments, we do not perform any specific Sim2Real methods such as domain randomization, though this would be an interesting avenue for future work. 

#### Baselines
We consider three baselines as comparison to our method

* __RL with human-designed reward__: Our first baseline is an RL-policy trained using a human-designed reward. Notably, since one of the primary advantages of IRL is the reduction in human-effort needed to define good reward functions, our goal is not to compare necessarily to the best-possible human-designed reward, but rather a simple reward that could be devised without signfiicant human effort. Hence, we do not use the best-designed reward functions, such as those used in previous work for lane-following with RL <d-cite key="kalapos2021vision"></d-cite>, which do produce good policies with RL alone but required significant engineering. Rather, we select the default reward function provided in the Duckietown simulator. While this reward function is known not to be optimal, this provides a good baseline for what a minimally-engineered reward function could produce.
* __RL with velocity reward__: The next baseline trains an RL policy on the simple primary goal reward of forward velocity. Hence, this baseline provides an estimate of performance if no IRL fine-tuning to the expert trajectories is performed. Under this reward function, the agent has no incentive to stay within the right lane.
* __Full IRL__: The final baseline trains IRL from scratch, without access to the simple forward velocity reward function. This baselne provides an estimate for the advantage gained by using a simple reward function as the primary goal, and learning only the remaining "correction" or constraint through IRL.

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




