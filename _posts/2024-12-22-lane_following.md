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

 ```mermaid 
 %%{init: { 'theme':'dark', 'securityLevel': 'loose'} }%%
flowchart LR
i("
<img src='assets/img/prof_pic.jpg' style='max-width:1000px;min-height:0'/>
") -..-> Minimize
 ```


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





