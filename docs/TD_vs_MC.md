# Temporal-Difference vs Monte Carlo

Both TD and Monte Carlo methods use experience to solve the prediction problem. Given some experience following a policy $\pi$ , both methods update their estimate $V$ of $v_\pi$ for the nonterminal states $S_t$ occurring in that experience.

Roughly speaking, Monte Carlo methods wait until the return following the visit is known, then use that return as a target for $V(S_t)$. A simple every-visit Monte Carlo method suitable for nonstationary environments is

$$
V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)],
$$

where $G_t$ is the actual return following time $t$ , and is $\alpha$ constant step-size parameter. Let us call this method *constant-MC*. Whereas Monte Carlo methods must wait until the end of the episode to determine the increment to $V(S_t)$ (only then is $G_t$ known), TD methods need to wait only until the next time step. At time $t+1$ they immediately form a target and make a useful update using the observed reward $R_{t+1}$  and the estimate $V(S_{t+1})$. The simplest TD method makes the update

$$
V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
$$

immediately on transition to $S_{t+1} $ and receiving $R_{t+1}$. In effect, the target for the Monte Carlo update is $G_t$, whereas the target for the TD update is $R_{t+1} + \gamma V(S_{t+1})$.

![Monte Carlo vs Temporal-Difference]()

In practice, TD methods have usually been found to converge faster than constant- ↵ MC methods on stochastic tasks, as illustrated in Example 6.2.

![Example 6.2  Random Walk]()

## Optimality of TD(0)

Suppose there is available only a finite amount of experience, say 10 episodes or 100 time steps. In this case, a common approach with incremental learning methods is to present the experience repeatedly until the method converges upon an answer. Given an approximate value function, $V$, the increments specified by (6.1) or (6.2) are computed for every time step t at which a nonterminal state is visited, but the value function is changed only once, by the sum of all the increments. Then all the available experience is processed again with the new value function to produce a new overall increment, and so on, until the value function converges. We call this *batch updating* because are made only after processing each complete *batch* of training data.

Under batch updating, TD(0) converges deterministically to a single answer independent of the step-size parameter, $\alpha$, as long as $\alpha$ is chosen to be sufficiently small.
