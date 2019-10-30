import tensorflow as tf
import gym

GLOBALE_NET_SCOPE = 'Globale_Net'  # 中央大脑的名字
Game = 'CartPole-v0'
env = gym.make(Game)  # 定义游戏环境

N_S = env.observation_space.shape[0]  # 观测值个数
N_A = env.action_space.n  # 行为值个数


class ACnet(object):  # 这个class即可用于生产global net，也可生成 worker net，因为结构相同
    def __init__(self, scope, globalAC=None):  # scope 用于确定生成什么网络
        if scope == GLOBALE_NET_SCOPE:  # 创建中央大脑
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # 初始化state，None代表batch，N—S是每个state的观测值个数
                self.build_net(scope)  # 建立中央大脑神经网络
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                # 定义中央大脑actor和critic的参数

        else:  # 创建worker两个网络的具体步骤
            with tf.variable_scope(scope):  # 这里的scope传入的是worker的名字
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # 初始化state
                self.a_his = tf.placeholder(tf.int32, [None, 1], 'A_his')  # 初始化action,是一个[batch，1]的矩阵，第二个维度为1，
                # 格式类似于[[1],[2],[3]]
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # 初始化v现实(V_target)，数据格式和上面相同

                self.acts_prob, self.v = self.build_net(scope)  # 建立神经网络，acts_prob为返回的概率值,v为返回的评价值
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

                td = tf.subtract(self.v_target, self.v, name='TD_error')  # 计算td—error即v现实和v估计之差
                # v—target和v都是一串值，v-target（现实）已经计算好并传入了，v估计由传入的
                # 一系列state送入critic网络确定

                with tf.name_scope('c_loss'):  # 计算Critic网络的loss
                    self.c_loss = tf.reduce_mean(tf.square(td))  # Critic的loss就是td—error加平方避免负数

                with tf.name_scope('a_loss'):  # 计算actor网络的损失
                    log_prob = tf.reduce_sum(
                        tf.log(self.acts_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1,
                        keep_dims=True)
                    # 这里是矩阵乘法，目的是筛选出本batch曾进行的一系列选择的概率值，acts—prob类似于一个向量[0.3,0.8,0.5]，
                    # one—hot是在本次进行的的操作置位1，其他位置置为0，比如走了三次a—his为[1,0,3],N—A是4，则one—hot就是[[0,1,0,0],[1,0,0,0],[0,0,0,1]]
                    # 相乘以后就是[[0,0.3,0,0],[0.8,0,0,0],[0,0,0,0.5]],log_prob就是计算这一系列选择的log值。

                    self.exp_v = log_prob * td  # td决定梯度下降的方向
                    self.a_loss = tf.reduce_mean(-self.exp_v)  # 计算actor网络的损失a-loss

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)  # 实现a_loss对a_params每一个参数的求导，返回一个list
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)  # 实现c_loss对c_params每一个参数的求导，返回一个list

            with tf.name_scope('sync'):  # worker和global的同步过程
                with tf.name_scope('pull'):  # 获取global参数,复制到local—net
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):  # 将参数传送到gloabl中去
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
                    # 其中传送的是local—net的actor和critic的参数梯度grads,具体计算在上面定义
                    # apply_gradients是tf.train.Optimizer中自带的功能函数，将求得的梯度参数更新到global中

    def build_net(self, scope):  # 建立神经网络过程
        w_init = tf.random_normal_initializer(0., .1)  # 初始化神经网络weights
        with tf.variable_scope('actor'):  # actor神经网络结构
            l_a = tf.layers.dense(inputs=self.s, units=200, activation=tf.nn.relu6,
                                  kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1),
                                  name='la')  # 建立第一层神经网络
            acts_prob = tf.layers.dense(inputs=l_a, units=N_A, activation=tf.nn.softmax,
                                        kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1),
                                        name='act_prob')  # 第二层神经网络其中之一输出为动作的均值

        with tf.variable_scope('critic'):  # critic神经网络结构,输入为位置的观测值，输出为评价值v
            l_c = tf.layers.dense(self.s, 20, tf.nn.relu6, kernel_initializer=w_init,
                                  bias_initializer=tf.constant_initializer(0.1), name='lc')  # 建立第一层神经网络
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1),
                                name='v')  # 第二层神经网络

        return acts_prob, v  # 建立神经网络后返回的是输入当前state得到的actor网络的动作概率和critic网络的v估计

    def update_global(self, feed_dict):  # 定义更新global参数函数
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # 分别更新actor和critic网络

    def pull_global(self):  # 定义更新local参数函数
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # 定义选择动作函数
        s = s[np.newaxis, :]
        probs = SESS.run(self.acts_prob, feed_dict={self.s: s})
        return np.random.choice(range(probs.shape[1]), p=probs.ravel())  # 从probs中按概率选取出某一个动作
