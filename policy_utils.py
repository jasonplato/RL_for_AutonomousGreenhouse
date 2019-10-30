import numpy as np
import tensorflow as tf
import threading
import queue
from collections import namedtuple
import scipy.signal


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(
        logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.ss = []
        self.gs = []
        self.features = []
        self.r = 0.0
        self.terminal = False

    def add(self, state, action, reward, value, g, s, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        self.gs += [g]
        self.ss += [s]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.gs.extend(other.gs)
        self.ss.extend(other.ss)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)


class RunnerThread(threading.Thread):
    """
    One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.
    """

    def __init__(self, env, policy, num_local_steps, visualise):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)


def env_runner(env, policy, num_local_steps, summary_writer, visualise):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    last_c_g, last_features = policy.get_initial_features()
    # print last_c_g
    length = 0
    rewards = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            # print last_c_g.shape
            fetched = policy.action(last_state, last_c_g, *last_features)
            action, value_, g, s, last_c_g, features = fetched[0], fetched[1], \
                                                       fetched[2], fetched[3], \
                                                       fetched[4], fetched[5:]
            action_to_take = action.argmax()
            # print action_to_take
            # print action
            # print g
            # print s
            # # exit(0)
            state, reward, terminal, info = env.step(action_to_take)

            # collect the experience
            rollout.add(last_state, action, reward, value_, g, s, terminal, last_features)
            length += 1
            rewards += reward

            last_state = state
            last_features = features

            # if info:
            #     summary = tf.Summary()
            #     for k, v in info.items():
            #         summary.value.add(tag=k, simple_value=float(v))
            #     summary_writer.add_summary(summary, policy.global_step.eval())
            #     summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_c_g, last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: %f. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, last_c_g, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout


def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)

    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])
    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]

    batch_s = np.asarray(rollout.ss)
    batch_g = np.asarray(rollout.gs)
    features = rollout.features
    return Batch(batch_si, batch_a, batch_r, rollout.terminal, batch_s, batch_g, features)


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


Batch = namedtuple("Batch", ["obs", "a", "returns", "terminal", "s", "g", "features"])
