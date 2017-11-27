from agent import Agent
import numpy as np


def test_Agent_init():
    """Tests for agents init function."""
    print("Testing Agent's __init__()")
    agent0 = Agent(5)
    agent1 = Agent()
    agent2 = Agent(4, np.array([1,2,3,5]))
    agent3 = Agent(3)
    try:
        agent4 = Agent(q=[55,22])
    except ValueError as detail:
        print("Caught exception:", detail)

    try:
        agent5 = Agent(policy_info="e_gr")
    except ValueError as detail:
        print("Caught exception:", detail)


    print("Agent3, q:", agent3.q)
    print("Agent3, reward:", agent3.reward(2))

    assert(agent0.k == 5)
    assert(agent1.k == 10)
    assert(np.array_equal([1,2,3,5], agent2.q))
    assert(np.array_equal(agent2.Q, None))
    assert(agent0.policy(agent0.policy_info[1]) < agent0.k)
    print("Agent's __init__() passed!")
    return True

def test_assertions(assertions):
    np.random.seed(420)
    if assertions:
        print("+"+"-"*50+"+")
        test_Agent_init()
        print("+"+"-"*50+"+")


if __name__ == '__main__':
    assertions = True

    test_assertions(assertions)
