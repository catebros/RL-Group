
**PART 1: Analyze, design, build, test/eval, compare and explain/justify** 


**For each Scenario: full analysis of ONE scenario end to end:**
- Already implemented some models. We need to use tensorboard.
- For scenario 3 and 4 a wrapper needs to be used for the rewards

1. **Setup & Framework**
   - Make sure TensorBoard is logging your training (reward curves, epsilon, loss) -> if you use SB3 SAC you need to implement a callback method (to watch training)
   - Try at least two different reward designs.
   - Try at least 2 models.

2. **Run**
   - Train your agents and save results (rewards, eval metrics)

3. **Evaluate & Interpret**
   - **Objective vs engineered reward**: did optimizing your custom reward actually improve the real goal (reaching the flag)? Show where they agree and where they diverge
   - **Numerical analysis**: report mean reward, standard deviation, and success rate over 100 evaluation episodes for each agent/config, episode length
   - **Statistical analysis**: comment on consistency, high std means the agent is unreliable even if the mean looks good. Distributions, Confidence intervals
   - **Policy analysis**: look at the policy heatmap and value function — what actions does the agent take in each region of the state space? Does the structure make sense? what does the policy actually do? Does it make sense?
   - **Topological/structural analysis**: use the phase portrait — does the agent show the oscillation pattern needed to build momentum? Is it efficient or wasteful?
   - **Physical interpretation**: explain *why* the agent behaves the way it does in terms of the actual physics of the Mountain Car. e.g. "the agent idles once it has enough velocity because no more thrust is needed to reach the goal"

4. **Visualize**
   - Training curves, policy heatmaps, value surface, phase portrait, visit frequency map..

5. **Write**: interpretation and implementation (in report and notebook)

**Comparative Analysis** — cross-scenario, done together as a group

The goal here is to compare all 4 scenarios against each other and extract conclusions about what the reward design and environment type actually change in the learned behaviour.

1. **Performance comparison across all 4 scenarios**
   - Build a single summary table: mean reward, std, success rate, mean steps to goal for every scenario and every agent
   - Compare S1 vs S3 (same env, different reward): did changing the reward from "minimize steps" to "minimize fuel" actually produce a more fuel-efficient policy? At what cost in steps?
   - Compare S2 vs S4 (same env, different reward): does the flat time penalty in S4 make the agent more aggressive (higher force magnitude) than S2?
   - Compare S1 vs S4 and S2 vs S3 (same objective, different env): how does moving from discrete to continuous actions change the difficulty and the quality of the solution?

2. **Policy structure comparison**
   - Plot all 4 policy maps side by side — do S1 and S3 policies differ? Where do they disagree (which states)?
   - For continuous (S2 vs S4): compare the action magnitude heatmaps — S4 should show higher forces across the state space
   - Compute the % of state cells where S1 and S3 policies disagree — quantify how much the reward actually changed the behaviour

3. **Efficiency trade-off analysis**
   - For discrete (S1 vs S3): is there a trade-off? Does the fuel-optimal agent (S3) take more steps than the time-optimal agent (S1)?
   - For continuous (S2 vs S4): does the step-optimal agent (S4) use more fuel (higher action²) than S2?
   - This is the key conceptual finding: reward shaping changes what is optimized, and the two objectives are in tension

4. **Physical interpretation of the differences**
   - Why does the discrete fuel-optimal agent idle more? What does that mean physically?
   - Why does the continuous step-optimal agent apply maximum force? Is that always the right strategy?
   - Which scenario produces the most "natural" or physically intuitive policy and why?

5. **Visualize**
   - Combined performance bar/box plots across all 4 scenarios
   - Side-by-side policy heatmaps (2×2 grid: S1, S2, S3, S4)
   - Trade-off scatter plot: mean steps vs mean fuel used, one point per scenario/agent

6. **Write**: interpretation and implementation (in report and notebook)

**Explanation Tools** — making the learned policies interpretable using basic ML

The goal here is to go beyond looking at the policy visually and actually quantify *what the agent is using to make decisions*. You use simple ML models to explain the RL policy, not to retrain it.

1. **Multinomial logistic regression on the policy**
   - Treat the policy as a classification problem: input = state (position, velocity), output = action chosen by the agent
   - Use engineered features: `[pos, vel, pos², vel², pos×vel, |vel|, sin(3×pos)]` — these capture the physics of the terrain
   - Fit the regression to the greedy policy grid of each scenario
   - Report accuracy: how well can a linear model explain the agent's decisions? Low accuracy = the policy is non-linear and complex

2. **Feature importance**
   - From the regression weights, compute mean |coefficient| per feature across all action classes
   - This tells you which state variables the agent relies on most:
     - Is position more important than velocity, or vice versa?
     - Does `sin(3×pos)` (the terrain shape) matter? If so, the agent has implicitly learned the hill geometry
     - Does `pos×vel` matter? That would suggest the agent reasons about momentum, not just position

3. **Compare explanations across scenarios**
   - Run the same regression on S1 and S3 policies — do they rely on the same features?
   - If S3 (min fuel) relies more on `|vel|` than S1 (min steps), it means the fuel agent is more sensitive to how fast it's already moving before deciding to thrust — which makes physical sense
   - This is the bridge between the statistical analysis and the physical interpretation

4. **Interpret the results**
   - Don't just report the numbers — explain what they mean
   - e.g. "velocity is the dominant feature, meaning the agent's main decision rule is: thrust when moving slowly, idle when moving fast"
   - e.g. "the regression achieves 85% accuracy, meaning most of the policy can be captured by a simple linear boundary in state space"

5. **Visualize**
   - Feature importance bar charts for each scenario (side by side)
   - Overlay the logistic regression decision boundary on the policy heatmap to show where it agrees and disagrees with the true policy

6. **Write**: interpretation and implementation (in report and notebook)

