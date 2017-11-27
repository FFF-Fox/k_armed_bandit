import numpy as np
import progress as prog

class Agent:
    """The agent class.
    Attributes
        k      : Int. The number of "arms" or actions that the agent can choose
                 from.
        q      : np.array[Float]. The actual values for each action. (Size = k)
        Q      : np.array[Float]. The agent's evaluations for each actual value.
                 (Size = k)
        N      : np.array[Int]. How many times an action has been chosen.
                 (Size = k)
        policy : () => Int. The agent's policy. Should return an Int that shows
                 the action chosen by the policy for a given state.
        reward : (Int) => Float. A function that given the selected action,
                 returns the agent's reward.

    Methods
        e_greedy        : (Int) => Int. ε-greedy policy based on the probability
                          of e.
        normal_around_q : (Int) => Float. Reward function that returns Rewards
                          based on a normal distribution around the actual values.
        run_session     : (Int, Boolean, Boolean) => dict(Str : np.array). Runs
                          a session for the given number of steps.
    """

    def __init__(self, k=10, q=None,
                 policy_info=["e_greedy", 0.1], reward="normal_around_q"):
        """Agent's constructor"""
        self.k = k
        self.Q = None
        self.N = None

        # Assigning the q.
        if q is not None:
            if len(q) != self.k:
                raise ValueError('len(q) != k')
            else:
                self.q = np.array(q)
        else:
            # Initialize q with random values from a Gausssian distribution.
            self.q = np.random.normal(1, 1, k)

        # Assigning the policy.
        if not isinstance(policy_info, list):
            raise ValueError("policy_info's value should be a list.")
        else:
            self.policy_info = policy_info
            if self.policy_info[0] == "e_greedy":
                self.policy = self.e_greedy

        # Assigning the reward function.
        if reward == "normal_around_q":
            self.reward = self.normal_around_q


    def e_greedy(self, e):
        """ε-greedy policy. Given the probability of choosing a
        random action, returns an action that is either greedy
        or random.
        Parameters:
            e : Float. The probability of choosing a random action.
        """

        if np.random.rand() < e:
            a = np.random.randint(0, self.k)
            return a
        else:
            a = np.argmax(self.Q)
            return a


    def normal_around_q(self, a):
        """Given a selected action a, returns a reward based on a normal
        distribution with mean = q[a].
        Parameters:
            a : Int. The action chosen by the agent.
        """
        return np.random.normal(self.q[a], 1)


    def run_session(self, t, verbose=False, new_session=True):
        """Run a session for t steps.
        Parameters:
            t           : Int. The number of iterations that the agent will act.
            verbose     : Boolean. Whether messages about the progress should be
                          displayed or not.
            new_session : Boolean. Choose whether the session is new or not,
                          so as the Q and N gets initialized to zeros or not.
        Notes: Actions A and rewards R are stored in arrays because they are
        needed for later calculations.
        """
        if new_session:
            self.Q = np.zeros(self.k)
            self.N = np.zeros(self.k)

        A = np.zeros(t, dtype=np.int)
        R = np.zeros(t,)
        total_reward = np.zeros(t,)

        for i in range(t):
            A[i] = self.policy(self.policy_info[1])
            R[i] = self.reward(A[i])

            if i == 0:
                total_reward[i] = R[i]
            else:
                total_reward[i] = total_reward[i-1] + R[i]

            self.N[A[i]] += 1
            self.Q[A[i]] = self.Q[A[i]] + (1/self.N[A[i]]) * (R[i] - self.Q[A[i]])

            if verbose:
                prog.print_progress(i+1, t, bar_length=50, prefix="Running session...")

        # The rewards taken and the actions chosen, for each individual step.
        session_metrics = {
            "R" : R,
            "A" : A,
            "total_reward" : total_reward,
            }

        return session_metrics


if __name__ == '__main__':
    np.random.seed(420)
    steps = 1500
    N = 200
    k = 10
    q = np.random.normal(0, 1, k)

    agents = [
        Agent(k, q=q, policy_info=["e_greedy", 0.0]),
        Agent(k, q=q, policy_info=["e_greedy", 0.1]),
        Agent(k, q=q, policy_info=["e_greedy", 0.01])
    ]

    # For each sample, we store the actions and rewards of its session.
    # Then we use that information to find the average reward and the
    # percentage of optimal actions taken by the samples of the current
    # agent.
    avg_rew = []
    opt_act = []
    avg_total_rew = []
    for i, agent in enumerate(agents):
        rewards = np.empty([N, steps])
        actions = np.empty([N, steps])
        total_rewards = np.empty([N, steps])

        # We sample the agent N times.
        for j in range(N):
            metrics = agent.run_session(steps)
            rewards[j] = metrics["R"]
            A = metrics["A"]
            actions[j] = A == np.argmax(agent.q)
            total_rewards[j] = metrics["total_reward"]

            prog.print_progress(j+1, N, bar_length=50, prefix="Agent "+str(i))

        # Let's calculate the metrics presented above.
        avg_rew.append(sum(rewards) / N)
        opt_act.append(sum(actions) * 100 / N)
        avg_total_rew.append(sum(total_rewards) / N)


    import matplotlib.pyplot as plt

    for A in avg_rew:
        plt.plot(A)
    plt.show()

    for O in opt_act:
        plt.plot(O)
    plt.show()

    for AT in avg_total_rew:
        plt.plot(AT)
    plt.show()
