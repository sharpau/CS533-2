# Austin Sharp
# CS533 Winter '14 Homework 2
# Infinite Horizon MDP Solver
import operator
from mdp import MDP

def bellman_backup(mdp, value_k, discount):
    """
    Given an MDP and its value function with time-to-go k, returns its value function with time-to-go k+1.
    """
    # computing a new value function, ie list over all states
    new_value_fn = []
    actions = []
    for s in range(mdp.num_states):
        # R(s) + max_a[sum_s'(T(s,a,s') dot Vk(s'))]
        action_results = []
        for a in range(mdp.num_actions):
            action_results.append(sum([mdp.transition(s, a, s_next) * value_k[s_next] for s_next in range(mdp.num_states)]))

        max_index, max_value = max(enumerate(action_results), key=operator.itemgetter(1))
        actions.append(max_index)
        new_value_fn.append(mdp.rewards[s] + discount * max_value)

    return new_value_fn, actions

def max_norm(new, old):
    return max([new[i] - old[i] for i in range(0, len(new))])

def plan(mdp, discount, epsilon):
    """
    Main algorithm. The input to your algorithm should be a description of an MDP and
    a time horizon H (positive integer). The output should be an optimal non-stationary value
    function and non-stationary policy for the MDP and time horizon .
    """
    bellman_error = []
    bellman_error.append(max(mdp.rewards))
    policy = []
    iterations = 0
    # initialize at iteration 0 with reward of each state
    value_function = mdp.rewards
    while bellman_error[len(bellman_error) - 1] > epsilon:
        new_values, policy = bellman_backup(mdp, value_function, discount)
        bellman_error.append(max_norm(new_values, value_function))
        value_function = new_values
        iterations += 1
    # at this point we have the k-th value and policy
    return value_function, policy, iterations

def part_ii_test():
    simple = MDP("simple_test.txt")

    val, pol, k = plan(simple, 0.99, 0.01)
    print "Value fn: " + str(val)
    print "Policy: " + str(pol)
    print "Iterations: " + str(k)


    val, pol, k = plan(simple, 0.9, 0.01)
    print "Value fn: " + str(val)
    print "Policy: " + str(pol)
    print "Iterations: " + str(k)


    val, pol, k = plan(simple, 0.5, 0.01)
    print "Value fn: " + str(val)
    print "Policy: " + str(pol)
    print "Iterations: " + str(k)


    val, pol, k = plan(simple, 0.1, 0.01)
    print "Value fn: " + str(val)
    print "Policy: " + str(pol)
    print "Iterations: " + str(k)


def generate_parking_mdp(n, distance_rewards, name):
    """
    Given a size n and and n-length list of reward by distance (starting from the nearest spot and ending with the farthest),
    generates an output file with the corresponding MDP.
    """
    assert(len(distance_rewards) == n)
    num_actions = 2  # drive, park
    rewards = []

    # state organization. always drive from state j to j + 1, clockwise
    for i in range(n - 1, -1, -1):  # n - 1, n - 2, ..., 0
        # A[i], unoccupied, unparked = state 4i
        rewards.append(-1)  # cost of time passing
        # A[i], occupied, unparked = state 4i + 1
        rewards.append(-1)  # cost of time passing
        # A[i], occupied, parked = state 4i + 2
        rewards.append(distance_rewards[i] - 101)  # cost of time passing + cost of crash
        # A[i], unoccupied, parked = state 4i + 3
        rewards.append(distance_rewards[i] - 1)  # reward for parking + cost of time passing
    for i in range(n):  # 0, 1, 2, ..., n - 1
        # B[i], unoccupied, unparked = state 4n + 4i
        rewards.append(-1)  # cost of time passing
        # B[i], occupied, unparked = state 4n + 4i + 1
        rewards.append(-1)  # cost of time passing
        # B[i], occupied, parked = state 4n + 4i + 2
        rewards.append(distance_rewards[i] - 101)  # cost of time passing + cost of crash
        # B[i], unoccupied, parked = state 4n + 4i + 3
        rewards.append(distance_rewards[i] - 1)  # reward for parking + cost of time passing
    rewards.append(0)  # sink state
    rewards[4 * n - 1] -= 50  # for parking in handicapped A[1]
    rewards[4 * n - 2] -= 50  # for parking in handicapped A[1]
    rewards[4 * n + 2] -= 50  # for parking in handicapped B[1]
    rewards[4 * n + 3] -= 50  # for parking in handicapped B[1]
    b_offset = 4 * n

    transitions = []  # transitions[actions][start state][end state]

    # initialize all transitions to empty - 8n + 1 states (2n spots * 2 for occupied/not * 2 for parked/not + 1 terminal)
    for a in range(num_actions):
        transitions.append([])
        for i in range((n * 8) + 1):
            transitions[len(transitions) - 1].append([0 for i in range(n * 8 + 1)])

    # assign actual values to the transition matrices
    # drive action
    # if state is parked, go to terminal state
    # otherwise, move to next state occupied with p, move to next state unoccupied with p-1 (swapped for B[])
    transitions[0][n * 8][n * 8] = 1.0  # if terminal, stay terminal
    for i in range(n):
        # probability of spot A[i+1] being open, or B[i+1] being taken
        if i == n - 1: # same probability repeated, both top spots are equal
            p = 1.0 / (i + 2.0)
        else:
            p = 1.0 / (i + 3.0)

        # row A, higher i = closer
        # transitions from unoccupied, unparked, state 4i
        transitions[0][4 * i][4 * (i + 1)] = p  # probability of next spot being open
        transitions[0][4 * i][4 * (i + 1) + 1] = 1.0 - p  # probably next spot is occupied
        # transitions from occupied, unparked, state 4i + 1... same as above
        transitions[0][4 * i + 1][4 * (i + 1)] = p  # probability of next spot being open
        transitions[0][4 * i + 1][4 * (i + 1) + 1] = 1.0 - p  # probably next spot is occupied
        # transitions from parked states always go to terminal state
        transitions[0][4 * i + 2][n * 8] = 1.0
        transitions[0][4 * i + 3][n * 8] = 1.0

        # row B, higher i = farther
        if i == n - 1:  # last spot, must wrap around
            p = 1.0 / (n - i + 1)
            # transitions from unoccupied, unparked, state 4i
            transitions[0][b_offset + 4 * i][0] = p  # probability of next spot being open
            transitions[0][b_offset + 4 * i][1] = 1.0 - p  # probably next spot is occupied
            # transitions from occupied, unparked, state 4i + 1... same as above
            transitions[0][b_offset + 4 * i + 1][0] = p  # probability of next spot being open
            transitions[0][b_offset + 4 * i + 1][1] = 1.0 - p  # probably next spot is occupied
        else:
            p = 1.0 / (n - i)
            # transitions from unoccupied, unparked, state 4i
            transitions[0][b_offset + 4 * i][b_offset + 4 * (i + 1)] = p  # probability of next spot being open
            transitions[0][b_offset + 4 * i][b_offset + 4 * (i + 1) + 1] = 1.0 - p  # probably next spot is occupied
            # transitions from occupied, unparked, state 4i + 1... same as above
            transitions[0][b_offset + 4 * i + 1][b_offset + 4 * (i + 1)] = p  # probability of next spot being open
            transitions[0][b_offset + 4 * i + 1][b_offset + 4 * (i + 1) + 1] = 1.0 - p  # probably next spot is occupied

        # transitions from parked states always go to terminal state
        transitions[0][b_offset + 4 * i + 2][n * 8] = 1.0
        transitions[0][b_offset + 4 * i + 3][n * 8] = 1.0

    # park action
    transitions[1][n * 8][n * 8] = 1.0  # if terminal, stay terminal
    for i in range(n):
        # row A, higher i = closer
        # transitions from unoccupied, unparked, state 4i
        transitions[1][4 * i][4 * i + 3] = 1.0  # parking in unoccupied
        # transitions from occupied, unparked, state 4i + 1... same as above
        transitions[1][4 * i + 1][4 * i + 2] = 1.0  # parking in occupied...ouch
        # transitions from parked states always go to terminal state
        transitions[1][4 * i + 2][n * 8] = 1.0
        transitions[1][4 * i + 3][n * 8] = 1.0

        # row B, higher i = farther
        # transitions from unoccupied, unparked, state 4i
        transitions[1][b_offset + 4 * i][b_offset + 4 * i + 3] = 1.0  # parking in unoccupied
        # transitions from occupied, unparked, state 4i + 1... same as above
        transitions[1][b_offset + 4 * i + 1][b_offset + 4 * i + 2] = 1.0  # parking in occupied...ouch
        # transitions from parked states always go to terminal state
        transitions[1][b_offset + 4 * i + 2][n * 8] = 1.0
        transitions[1][b_offset + 4 * i + 3][n * 8] = 1.0

    # write everything out to file
    with open(name + ".txt", "w") as out_file:
        out_file.write(str(8 * n + 1) + "\n")
        out_file.write(str(num_actions) + "\n")
        out_file.write(" ".join([str(x) for x in rewards]))
        out_file.write("\n")
        for a in range(num_actions):
            out_file.write("\n".join([" ".join([str(cell) for cell in line]) for line in transitions[a]]))
            out_file.write("\n")


def part_iii_test():
    generate_parking_mdp(10, [40, 36, 32, 28, 24, 20, 16, 12, 8, 4], "parking_mdp_linear_rewards_n_10")
    linear_10 = MDP("parking_mdp_linear_rewards_n_10.txt")

    val, pol, k = plan(linear_10, 0.99, 0.01)
    print "Rewards: " + str(linear_10.rewards)
    print "Value fn: " + str(val)
    print "Policy: " + str(pol)
    print "Iterations: " + str(k)
    with open("linear_10_results.csv", "w") as out_file:
        for i in range(linear_10.num_states):
            out_file.write("(" + str(linear_10.rewards[i]) + ";" + str(val[i]) + ";" + str(pol[i]) + "),")
            if (i + 1) % 4 == 0:
                out_file.write("\n")


    generate_parking_mdp(10, [50, 40, 32, 24, 18, 12, 8, 4, 2, 0], "parking_mdp_quad_rewards_n_10")
    quad_10 = MDP("parking_mdp_quad_rewards_n_10.txt")

    val, pol, k = plan(quad_10, 0.99, 0.01)
    print "Rewards: " + str(quad_10.rewards)
    print "Value fn: " + str(val)
    print "Policy: " + str(pol)
    print "Iterations: " + str(k)
    with open("quad_10_results.csv", "w") as out_file:
        for i in range(quad_10.num_states):
            out_file.write("(" + str(quad_10.rewards[i]) + ";" + str(val[i]) + ";" + str(pol[i]) + "),")
            if (i + 1) % 4 == 0:
                out_file.write("\n")

# Main program flow.
#part_ii_test()

part_iii_test()




