<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true},
      jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
      extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
      TeX: {
      extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
      equationNumbers: {
      autoNumber: "AMS"
      }
    }
  });
</script>

# Temporal-Difference Learning

If one had to identify one idea as central and novel to reinforcement learning, it would undoubtedly be *temporal-difference* (TD) learning. TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment’s dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).

As usual, we start by focusing on the policy evaluation or prediction problem, the problem of estimating the value function $v_\pi$ for a given policy $\pi$. For the control problem (finding an optimal policy), DP, TD, and Monte Carlo methods all use some variation of generalized policy iteration (GPI). The differences in the methods are primarily differences in their approaches to the prediction problem.

## TD Prediction

TD methods need to wait only until the next time step. At time $t+1$ they immediately form a target and make a useful update using the observed reward $R_{t+1}$  and the estimate $V(S_{t+1})$. The simplest TD method makes the update

$$
V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
$$

immediately on transition to $S_{t+1} $ and receiving $R_{t+1}$. This TD method is called *TD(0)*, or *one-step* TD, because it is a special case of the TD($\lambda$) and *n-step* TD methods developed in Chapter 12 and Chapter 7.

![TD(0) for estimating v_pi]()

Because TD(0) bases its update in part on an existing estimate, we say that it is a *bootstrapping* method, like DP.

![Backup diagram for TD(0)]()

Shown to the right is the backup diagram for tabular TD(0). The value estimate for the state node at the top of the backup diagram is updated on the basis of the one sample transition from it to the immediately following state. We refer to TD and Monte Carlo updates as *sample updates* because they involve looking ahead to a sample successor state (or state–action pair), using the value of the successor and the reward along the way to compute a backed-up value, and then updating the value of the original state (or state–
action pair) accordingly. *Sample* updates differ from the *expected* updates of DP methods in that they are based on a single sample successor rather than on a complete distribution of all possible successors.

Finally, note that the quantity in brackets in the TD(0) update is a sort of error, measuring the difference between the estimated value of $S_t$ and the better estimate $R_{t+1} + \gamma V(S_{t+1})$. This quantity, called the TD error, arises in various forms throughout reinforcement learning:

$$
    \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t).
$$

## Advantages of TD Prediction Models

* TD methods have an advantage over DP methods in that they do not
require a model of the environment, of its reward and next-state probability distributions.
* They are naturally implemented in an online, fully incremental fashion. With Monte Carlo methods one must wait until the end of an episode, because only then is the return known, whereas with TD methods one need wait only one time step.

## Batch Uptading

Under batch updating, TD(0) converges deterministically to a single answer independent of the step-size parameter, $\alpha$, as long as $\alpha$ is chosen to be sufficiently small. The constant-$\alpha$ MC method also converges deterministically under the same conditions, but to a different answer.

In batch form, TD(0) is faster than Monte Carlo methods because it computes the true certainty-equivalence estimate. Nonbatch TD(0) may be faster than constant-$\alpha$ MC because it is moving toward a better estimate, even though it is not getting all the way there. At the current time nothing more definite can be said about the relative eficiency of online TD and Monte Carlo methods.

**Example 6.3:** The Monte Carlo method is optimal only in a limited way, and that TD is optimal in a way that is more relevant to predicting returns.

On tasks with large state spaces, TD methods may be the only feasible way of approximating the certainty-equivalence solution.