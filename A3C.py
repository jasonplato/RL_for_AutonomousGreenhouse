import multiprocessing  # 多线程模块
import threading  # 线程模块
import queue
import tensorflow as tf
import numpy as np
import gym
import os
import shutil  # 拷贝文件用
import matplotlib.pyplot as plt
from FeudalBatchProcessor import FeudalBatchProcessor
import policy_utils
from LSTMmodel import SingleStepLSTM

Game = 'CartPole-v0'
N_workers = multiprocessing.cpu_count()  # 独立玩家个体数为cpu数
# MAX_GLOBAL_EP = 2000  # 中央大脑最大回合数
MAX_GLOBALE_STEP = 100000  # 中央大脑最大步数
GLOBAL_NET_SCOPE = 'Global_Net'  # 中央大脑的名字
UPDATE_GLOBALE_ITER = 10  # 中央大脑每N次提升一次
GAMMA = 0.9  # 衰减度
LR_A = 0.0001  # Actor网络学习率
LR_C = 0.001  # Critic 网络学习率
beta_start = 0.01
beta_end = 0.001
decay_steps = 50000

GLOBALE_RUNNING_R = []  # 存储总的reward
# GLOBALE_EP = 0  # 中央大脑步数
GLOBALE_STEP = 0  # 中央大脑步数

env = gym.make(Game)  # 定义游戏环境

N_S = env.observation_space.shape[0]  # 观测值个数
N_A = env.action_space.n  # 行为值个数


class ACnet(object):  # 这个class即可用于生产global net，也可生成 worker net，因为结构相同
    def __init__(self, scope, globalAC=None, global_step=None):  # scope 用于确定生成什么网络
        # global GLOBALE_STEP
        # self.global_step = GLOBALE_STEP
        if scope == GLOBAL_NET_SCOPE:  # 创建中央大脑
            with tf.variable_scope(scope):
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
                self.obs_space = N_S
                self.act_space = N_A
                self.k = 16
                self.g_dim = 256
                self.c = 10
                self.vf_hidden_size = 128  # for value function network
                self.alpha = 0.5  # for build loss
                self.batch_processor = FeudalBatchProcessor(self.c)
                self.build_model()  # build feudal policy model

        else:  # 创建worker两个网络的具体步骤
            with tf.variable_scope(scope):  # 这里的scope传入的是worker的名字
                self.global_step = globalAC.global_step
                self.obs_space = N_S
                self.act_space = N_A
                self.k = 16
                self.g_dim = 256
                self.c = 10
                self.vf_hidden_size = 128  # for value function network
                self.alpha = 0.5  # for build loss
                self.batch_processor = FeudalBatchProcessor(self.c)
                self.build_model()  # build feudal policy model

            with tf.name_scope('local_grad'):
                grads = tf.gradients(self.loss, self.var_list)
                grads, _ = tf.clip_by_global_norm(grads, 40)

            with tf.name_scope('sync'):  # worker和global的同步过程
                with tf.name_scope('pull'):  # 获取global参数,复制到local—net
                    self.pull_params_op = tf.group(*[v1.assign(v2)
                                                     for v1, v2 in zip(self.var_list, globalAC.var_list)])
                with tf.name_scope('push'):  # 将参数传送到gloabl中去
                    self.update_params_op = OPT.apply_gradients(zip(grads, globalAC.var_list))
                    # 其中传送的是local—net的actor和critic的参数梯度grads,具体计算在上面定义
                    # apply_gradients是tf.train.Optimizer中自带的功能函数，将求得的梯度参数更新到global中
            self.inc_step = self.global_step.assign_add(tf.shape(self.obs)[0])
            self.train_op = tf.group(self.update_params_op, self.inc_step)
            # GLOBALE_STEP += tf.shape(self.obs)[0]

    def build_model(self):
        """
        Builds the manager and worker models.
        """
        with tf.variable_scope('FeUdal'):
            self.build_placeholders()
            self.build_perception()
            self.build_manager()
            self.build_worker()
            self.build_loss()
            self.var_list = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        # for v in self.var_list:
        #     print v.name

        self.state_in = [self.worker_lstm.state_in[0],
                         self.worker_lstm.state_in[1],
                         self.manager_lstm.state_in[0],
                         self.manager_lstm.state_in[1]
                         ]
        self.state_out = [self.worker_lstm.state_out[0],
                          self.worker_lstm.state_out[1],
                          self.manager_lstm.state_out[0],
                          self.manager_lstm.state_out[1]
                          ]
        # for v in self.var_list:
        #     print v

    def build_placeholders(self):
        # standard for all policies
        self.obs = tf.placeholder(tf.float32, [None,
                                               self.obs_space])  # ! self.obs = tf.placeholder(tf.float32, [None] + list(self.obs_space))
        # ! self.obs_space = env.observation_space.shape
        self.r = tf.placeholder(tf.float32, (None,1))
        self.ac = tf.placeholder(tf.float32, (None, self.act_space))
        self.adv = tf.placeholder(tf.float32, [None])  # unused

        # specific to FeUdal
        self.prev_g = tf.placeholder(tf.float32, (None, None, self.g_dim))
        self.ri = tf.placeholder(tf.float32, (None,))
        self.s_diff = tf.placeholder(tf.float32, (None, self.g_dim))

    def build_perception(self):
        self._obs = tf.expand_dims(self.obs, -1)  # !
        self._obs = tf.expand_dims(self._obs, -1)  # !
        conv1 = tf.layers.conv2d(inputs=self._obs,
                                 filters=16,
                                 kernel_size=[2, 1],  # ! kernel_size = [8,8]
                                 activation=tf.nn.elu,
                                 strides=1)  # ! strides = 4
        conv2 = tf.layers.conv2d(inputs=conv1,
                                 filters=32,
                                 kernel_size=[2, 1],  # ! kernel_size = [4,4]
                                 activation=tf.nn.elu,
                                 strides=1)  # ! strides = 2

        flattened_filters = policy_utils.flatten(conv2)
        self.z = tf.layers.dense(inputs=flattened_filters,
                                 units=256,
                                 activation=tf.nn.elu)

    def build_manager(self):
        with tf.variable_scope('manager'):
            # Calculate manager internal state
            self.s = tf.layers.dense(inputs=self.z,
                                     units=self.g_dim,
                                     activation=tf.nn.elu)

            # Calculate manager output g
            x = tf.expand_dims(self.s, [0])
            self.manager_lstm = SingleStepLSTM(x,
                                               self.g_dim,
                                               step_size=tf.shape(self.obs)[:1])
            g_hat = self.manager_lstm.output
            self.g = tf.nn.l2_normalize(g_hat, dim=1)

            self.manager_vf = self.build_value(g_hat)

    def build_worker(self):
        with tf.variable_scope('worker'):
            num_acts = self.act_space

            # Calculate U
            self.worker_lstm = SingleStepLSTM(tf.expand_dims(self.z, [0]),
                                              size=num_acts * self.k,
                                              step_size=tf.shape(self.obs)[:1])
            flat_logits = self.worker_lstm.output

            self.worker_vf = self.build_value(flat_logits)

            U = tf.reshape(flat_logits, [-1, num_acts, self.k])

            # Calculate w
            cut_g = tf.stop_gradient(self.g)
            cut_g = tf.expand_dims(cut_g, [1])
            gstack = tf.concat([self.prev_g, cut_g], axis=1)

            self.last_c_g = gstack[:, 1:]
            # print self.last_c_g
            gsum = tf.reduce_sum(gstack, axis=1)
            phi = tf.get_variable("phi", (self.g_dim, self.k))
            w = tf.matmul(gsum, phi)
            w = tf.expand_dims(w, [2])
            # Calculate policy and sample
            logits = tf.reshape(tf.matmul(U, w), [-1, num_acts])
            self.pi = tf.nn.softmax(logits)
            self.log_pi = tf.nn.log_softmax(logits)
            self.sample = policy_utils.categorical_sample(
                tf.reshape(logits, [-1, num_acts]), num_acts)[0, :]

    def build_value(self, _input):
        with tf.variable_scope('VF'):
            hidden = tf.layers.dense(inputs=_input,
                                     units=self.vf_hidden_size,
                                     activation=tf.nn.elu)

            w = tf.get_variable("weights", (self.vf_hidden_size, 1))
            return tf.matmul(hidden, w)

    def build_loss(self):
        cutoff_vf_manager = tf.reshape(tf.stop_gradient(self.manager_vf), [-1])
        dot = tf.reduce_sum(tf.multiply(self.s_diff, self.g), axis=1)
        gcut = tf.stop_gradient(self.g)
        mag = tf.norm(self.s_diff, axis=1) * tf.norm(gcut, axis=1) + .0001
        dcos = dot / mag
        manager_loss = -tf.reduce_sum((self.r - cutoff_vf_manager) * dcos)

        cutoff_vf_worker = tf.reshape(tf.stop_gradient(self.worker_vf), [-1])
        log_p = tf.reduce_sum(self.log_pi * self.ac, [1])
        worker_loss = (self.r + self.alpha * self.ri - cutoff_vf_worker) * log_p
        worker_loss = -tf.reduce_sum(worker_loss, axis=0)

        Am = self.r - self.manager_vf
        manager_vf_loss = .5 * tf.reduce_sum(tf.square(Am))

        Aw = (self.r + self.alpha * self.ri) - self.worker_vf
        worker_vf_loss = .5 * tf.reduce_sum(tf.square(Aw))

        entropy = -tf.reduce_sum(self.pi * self.log_pi)

        beta = tf.train.polynomial_decay(beta_start, self.global_step,
                                         end_learning_rate=beta_end,
                                         decay_steps=decay_steps,
                                         power=1)

        # worker_loss = tf.Print(worker_loss,[manager_loss,worker_loss,manager_vf_loss,worker_vf_loss,entropy])
        self.loss = worker_loss + manager_loss + \
                    worker_vf_loss + manager_vf_loss - \
                    entropy * beta

    def update_global(self, feed_dict):  # 定义更新global参数函数
        SESS.run([self.update_params_op], feed_dict)  # 分别更新actor和critic网络

    def pull_global(self):  # 定义更新local参数函数
        SESS.run([self.pull_params_op])

    def action(self, ob, g, cw, hw, cm, hm):  # 定义选择动作函数
        # ob = ob[np.newaxis, :]
        ob = ob.reshape([-1, self.obs_space])
        return SESS.run([self.sample, self.manager_vf, self.g, self.s, self.last_c_g] + self.state_out,
                        feed_dict={self.obs: ob, self.state_in[0]: cw, self.state_in[1]: hw, self.state_in[2]: cm,
                                   self.state_in[3]: hm, self.prev_g: g})
        # return np.random.choice(range(probs.shape[1]), p=probs.ravel())  # 从probs中按概率选取出某一个动作

    def value(self, ob, g, cw, hw, cm, hm):
        sess = tf.get_default_session()
        return sess.run(self.manager_vf,
                        {self.obs: [ob], self.state_in[0]: cw, self.state_in[1]: hw,
                         self.state_in[2]: cm, self.state_in[3]: hm,
                         self.prev_g: g})[0]

    def get_initial_features(self):
        return np.zeros((1, 1, self.g_dim), np.float32), self.worker_lstm.state_init + self.manager_lstm.state_init

    def update_batch(self, batch):
        return self.batch_processor.process_batch(batch)


class Worker(object):
    def __init__(self, name, globalAC):  # 传入的name是worker的名字，globalAC是已经建立好的中央大脑GLOBALE—AC
        self.env = gym.make(Game).unwrapped
        self.name = name  # worker的名字
        self.global_AC = globalAC
        self.local_AC = ACnet(scope=name, globalAC=globalAC)  # 第二个参数当传入的是已经建立好的GLOBALE—AC时创建的是local net
        # 建立worker的AC网络
        self.runner = policy_utils.RunnerThread(self.env, self.local_AC, 20, visualise=0)

    def pull_batch_from_queue(self):
        """
        self explanatory:  take a rollout from the queue of the thread runner.
        """
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)

    def work(self):  # 定义worker运行的的具体过程
        global GLOBALE_STEP, MAX_GLOBALE_STEP
        # global GLOBALE_RUNNING_R, GLOBALE_EP  # 两个全局变量，R是所有worker的总reward，ep是所有worker的总episode
        # total_step = 1  # 本worker的总步数
        # buffer_s, buffer_a, buffer_r = [], [], []  # state,action,reward的缓存
        SESS.run(self.local_AC.pull_params_op)
        self.start(SESS, summary_writer=0)
        global_step = SESS.run(self.global_AC.global_step)
        # print(type(GLOBALE_STEP < MAX_GLOBALE_STEP))
        while not COORD.should_stop() and global_step < MAX_GLOBALE_STEP:  # 停止本worker运行的条件
            # 本循环一次是一个回合

            # s = self.env.reset()  # 初始化环境
            if self.name == 'W_0':  # 只有worker0才将动画图像显示
                self.env.render()
            ep_r = 0  # 本回合总的reward
            SESS.run(self.local_AC.pull_params_op)
            rollout = self.pull_batch_from_queue()
            batch = policy_utils.process_rollout(rollout, gamma=.99)
            batch = self.local_AC.update_batch(batch)
            # batch.ri = [item for sublist in batch.ri for item in sublist]
            # returns = [item for sublist in batch.returns for item in sublist]
            # batch._replace(returns=returns)
            # print("batch.returns.shape:",batch.returns.shape)
            # print("batch.ri.shape:",batch.ri.le)
            fetches = [self.local_AC.train_op]
            feed_dict = {
                self.local_AC.obs: batch.obs,
                self.global_AC.obs: batch.obs,

                self.local_AC.ac: batch.a,
                self.global_AC.ac: batch.a,

                self.local_AC.r: batch.returns,
                self.global_AC.r: batch.returns,

                self.local_AC.s_diff: batch.s_diff,
                self.global_AC.s_diff: batch.s_diff,

                self.local_AC.prev_g: batch.gsum,
                self.global_AC.prev_g: batch.gsum,

                self.local_AC.ri: batch.ri,
                self.global_AC.ri: batch.ri
            }

            for i in range(len(self.local_AC.state_in)):
                feed_dict[self.local_AC.state_in[i]] = batch.features[i]
                feed_dict[self.global_AC.state_in[i]] = batch.features[i]

            fetched = SESS.run(fetches, feed_dict=feed_dict)
            # while True:  # 本循环一次是一步
            #     if self.name == 'W_0':  # 只有worker0才将动画图像显示
            #         self.env.render()
            #
            #     fetched = self.AC.action(last_state, *last_features)  # 将当前状态state传入AC网络选择动作action
            #     action, value_, g, s, last_c_g, features = fetched[0], fetched[1], \
            #                                                fetched[2], fetched[3], \
            #                                                fetched[4], fetched[5:]
            #     a = action.argmax()
            #     state, reward, done, info = self.env.step(a)  # 行动并获得新的状态和回报等信息
            #     rollout.add(last_state,action,reward,value_,g,s,done,last_features)
            #
            #     if done: reward = -5  # 如果结束了，reward给一个惩罚数
            #
            #     ep_r += reward  # 记录本回合总体reward
            #     # buffer_s.append(s)  # 将当前状态，行动和回报加入缓存
            #     # buffer_a.append(a)
            #     # buffer_r.append(r)
            #     last_state = state
            #     last_features = features
            #     if total_step % UPDATE_GLOBALE_ITER == 0 or done:  # 每iter步完了或者或者到达终点了，进行同步sync操作
            #         if done:
            #             v_s_ = 0  # 如果结束了，设定对未来的评价值为0
            #         else:
            #             v_s_ = SESS.run(self.AC.v, feed_dict={self.AC.s: s_[np.newaxis, :]})[
            #                 0, 0]  # 如果是中间步骤，则用AC网络分析下一个state的v评价
            #
            #         buffer_v_target = []
            #         for r in buffer_r[::-1]:  # 将下一个state的v评价进行一个反向衰减传递得到每一步的v现实
            #             v_s_ = r + GAMMA * v_s_
            #             buffer_v_target.append(v_s_)  # 将每一步的v现实都加入缓存中
            #         buffer_v_target.reverse()  # 反向后，得到本系列操作每一步的v现实(v-target)
            #
            #         buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
            #             buffer_v_target)
            #
            #         feed_dict = {
            #             self.AC.obs: buffer_s,  # 本次走过的所有状态，用于计算v估计
            #             self.AC.ac: buffer_a,  # 本次进行过的所有操作，用于计算a—loss
            #             self.AC.v: buffer_v_target  # 走过的每一个state的v现实值，用于计算td
            #         }
            #
            #         self.AC.update_global(feed_dict)  # update—global的具体过程在AC类中定义，feed-dict如上
            #
            #         buffer_s, buffer_a, buffer_r = [], [], []  # 清空缓存
            #
            #         self.AC.pull_global()  # 从global—net提取出参数赋值给local—net
            #
            #     s = s_  # 跳转到下一个状态
            #     total_step += 1  # 本回合总步数加1
            #
            #     if done:  # 如果本回合结束了
            #         if len(GLOBALE_RUNNING_R) == 0:  # 如果尚未记录总体running
            #             GLOBALE_RUNNING_R.append(ep_r)
            #         else:
            #             GLOBALE_RUNNING_R.append(0.9 * GLOBALE_RUNNING_R[-1] + 0.1 * ep_r)
            #
            #         print(self.name, 'EP:', GLOBALE_EP)
            #         GLOBALE_EP += 1  # 加一回合
            #         break  # 结束本回合

            # global_step = SESS.run(self.global_AC.global_step)


if __name__ == '__main__':
    SESS = tf.Session()

    with tf.device('/cpu:0'):
        OPT = tf.train.AdamOptimizer(1e-4)  # 后续主要是使用该optimizer中的apply—gradients操作
        # OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')  # 定义critic训练过程
        GLOBAL_AC = ACnet(scope=GLOBAL_NET_SCOPE)  # 创建中央大脑GLOBALE_AC，只创建结构(A和C的参数)
        workers = []
        for i in range(N_workers):  # N—workers等于cpu数量
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(name=i_name, globalAC=GLOBAL_AC))  # 创建独立的worker

        COORD = tf.train.Coordinator()  # 多线程
        SESS.run(tf.global_variables_initializer())  # 初始化所有参数

        worker_threads = []
        for worker in workers:  # 并行过程
            job = lambda: worker.work()  # worker的工作目标,此处调用Worker类中的work
            t = threading.Thread(target=job)  # 每一个线程完成一个worker的工作目标
            t.start()  # 启动每一个worker
            worker_threads.append(t)  # 每一个worker的工作都加入thread中
        COORD.join(worker_threads)  # 合并几个worker,当每一个worker都运行完再继续后面步骤

        plt.plot(np.arange(len(GLOBALE_RUNNING_R)), GLOBALE_RUNNING_R)  # 绘制reward图像
        plt.xlabel('step')
        plt.ylabel('Total moving reward')
        plt.show()
