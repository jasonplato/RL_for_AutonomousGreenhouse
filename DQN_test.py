import tensorflow as tf
import numpy as np
import collections
import random
# from GreenhouseControlPanel import MyENv
from greenhousecontrolpanel.GreenhouseControlPanel import GreenhouseControlPanel as MyEnv

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.3),
    'bias_initializer': tf.constant_initializer(0.1)
}


class Memory(object):
    def __init__(self, batch_size, max_size):
        self.batch_size = batch_size  # mini batch大小
        self.max_size = max_size
        self._transition_store = collections.deque()

    def store_transition(self, s, a, r, s_, done):
        if len(self._transition_store) == self.max_size:
            self._transition_store.popleft()

        self._transition_store.append((s, a, r, s_, done))

    def get_mini_batches(self):
        n_sample = self.batch_size if len(self._transition_store) >= self.batch_size else len(self._transition_store)
        t = random.sample(self._transition_store, k=n_sample)
        t = list(zip(*t))

        return tuple(np.array(e) for e in t)


class DQN(object):
    def __init__(self, sess, s_dim, a_dim, batch_size, gamma, lr, epsilon, replace_target_iter):
        self.sess = sess
        self.s_dim = s_dim  # 状态维度
        self.a_dim = a_dim  # one hot行为维度
        self.gamma = gamma
        self.lr = lr  # learning rate
        self.epsilon = epsilon  # epsilon-greedy
        self.replace_target_iter = replace_target_iter  # 经历C步后更新target参数

        self.memory = Memory(batch_size, 10000)
        self._learn_step_counter = 0
        self._generate_model()

    def choose_action(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.a_dim)
        else:
            q_eval_z = self.sess.run(self.q_eval_z, feed_dict={
                self.s: s[np.newaxis, :]
            })
            return q_eval_z.squeeze().argmax()

    def _generate_model(self):
        self.s = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s')
        self.a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a')
        self.r = tf.placeholder(tf.float32, shape=(None, 1), name='r')
        self.s_ = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_')
        self.done = tf.placeholder(tf.float32, shape=(None, 1), name='done')

        self.q_eval_z = self._build_net(self.s, 'eval_net', True)
        self.q_target_z = self._build_net(self.s_, 'target_net', False)

        # y = r + gamma * max(q^)
        q_target = self.r + self.gamma * tf.reduce_max(self.q_target_z, axis=1, keepdims=True) * (1 - self.done)

        q_eval = tf.reduce_sum(self.a * self.q_eval_z, axis=1, keepdims=True)
        # a_mask = tf.cast(self.a, tf.bool)
        # q_eval = tf.expand_dims(tf.boolean_mask(self.q_eval_z, a_mask), 1)

        self.loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval))

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        param_target = tf.global_variables(scope='target_net')
        param_eval = tf.global_variables(scope='eval_net')

        # 将eval网络参数复制给target网络
        self.target_replace_ops = [tf.assign(t, e) for t, e in zip(param_target, param_eval)]

        tf.summary.scalar('loss', self.loss)
        # tf.summary.scalar('r',self.r)

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(s, 20, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
            q_z = tf.layers.dense(l, self.a_dim, trainable=trainable, **initializer_helper)

        return q_z

    def store_transition_and_learn(self, s, a, r, s_, done):
        if self._learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_ops)

        # 将行为转换为one hot形式
        one_hot_action = np.zeros(self.a_dim)
        one_hot_action[a] = 1

        # r = np.asarray([r])

        self.memory.store_transition(s, one_hot_action, r, s_, [done])
        summary = self._learn()
        self._learn_step_counter += 1
        return summary

    def _learn(self):
        s, a, r, s_, done = self.memory.get_mini_batches()
        r = np.reshape(r, [-1, 1])
        done = np.reshape(done, [-1, 1])
        # print("s:",s.shape)
        # print("a:",a.shape)
        # print("r:",r.shape)
        # print("s_:",s_.shape)
        # print("done:",done.shape)

        loss, summary, _ = self.sess.run([self.loss, merged_summary_op, self.optimizer], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.done: done
        })
        print("steps: {} loss: {}  rewards: {}".format(steps, loss, r))
        return summary


ENV_SPACE = 24
ACT_SPACE = 30
BATCH_SIZE = 12
MAX_EPISODE = 200
REPLACE_TARGET_ITERSTEP = 20

if __name__ == '__main__':
    # sess = tf.Session()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        rl = DQN(
            sess=sess,
            s_dim=ENV_SPACE,  # 状态维度
            a_dim=ACT_SPACE,  # 行为one hot形式维度
            batch_size=BATCH_SIZE,
            gamma=0.99,
            lr=0.01,  # learning rate
            epsilon=0.1,  # epsilon-greedy
            replace_target_iter=REPLACE_TARGET_ITERSTEP  # 经历C步后更新target参数
        )
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('DQN_test_log', sess.graph)

        tf.global_variables_initializer().run()
        myenv = MyEnv()
        last_reward = None
        delta_day = 1
        steps = 0
        for i_episode in range(MAX_EPISODE):
            # s = env.reset()
            ek, ev = myenv.fetch(np.array([17, 19, 19, 20, 20, 16, 16, 17,
                                           2, 1, 1, 2,
                                           400, 800, 800, 400,
                                           60, 80, 80, 60,
                                           45, 35, 35, 45]), Tplus=delta_day)
            s = ev[0][0][0]
            s = np.delete(s, 0, axis=0)
            # print("s:", s)
            # print("ek:", ek)
            # print("ev:",ev[-1][-1])

            last_reward = ev[-1][-1]
            # print("last_reward:", last_reward)
            # 一次episode的奖励总和
            # r_sum = 0
            while True:
                delta_day += 1
                # 选行为
                a = rl.choose_action(s)
                # print("a:", a)
                action = np.array([a, a, a, a, a, a, a, a,
                                   2, 1, 1, 2,
                                   400, 800, 800, 400,
                                   60, 80, 80, 60,
                                   45, 35, 35, 45])
                # 根据行为获得下个状态的信息
                ek, ev = myenv.fetch(action, Tplus=delta_day)
                reward = ev[-1][-1]
                r = reward - last_reward
                # tf.summary.scalar('reward',r)
                # print("r:", r)
                if reward >= 50:
                    done = True
                else:
                    done = False
                s_ = ev[0][0][0]
                s_ = np.delete(s_, 0, axis=0)

                print("a:", a)
                summary = rl.store_transition_and_learn(s, a, r, s_, done)

                # r_sum += r
                if done:
                    print(i_episode, reward)
                    break

                s = s_
                last_reward = reward
                steps += 1
                summary_writer.add_summary(summary, steps)
                # summary_str = sess.run(merged_summary_op)
                # summary_writer.add_summary(summary_str, i_episode)
                if steps % 100 == 0:
                    saver.save(sess, "DQN_test")
