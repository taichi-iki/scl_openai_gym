# coding: utf-8

import numpy as np
import queue
import time
from multiprocessing import Process, Queue
from threading import Timer

import chainer
import chainer.cuda as cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, serializers
from chainer.variable import Variable
from chainer.optimizer import GradientClipping, WeightDecay
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy

class TrajectorySegment(object):
    def __init__(self, s=None, a=None, r=None, f=None, v=0):
        self.s = s
        self.a = a
        self.r = r
        self.f = f
        self.v = v

    def is_prepared(self):
        return not (self.s is None or self.a is None or self.r is None)

class NetworkProcessBase(object):
    def __init__(self, receiver, sender, prefix, msg_interval=0.01):
        # queue to comunicate with parent procrss
        self.receiver = receiver
        self.sender   = sender
        self.prefix       = prefix
        self.msg_interval = msg_interval

    def write_info(self, s):
        print('[%s] %s'%(self.prefix, s))

    def routine(self):
        time.sleep(0.1)

    def msg_handler(self):
        try:
            msg, args = self.receiver.get_nowait()
            if hasattr(self, msg): getattr(self, msg)(*args)
        except queue.Empty:
            pass
        Timer(self.msg_interval, self.msg_handler).start()
        
    def run(self):
        print('[%s] process started'%(self.prefix))
        Timer(self.msg_interval, self.msg_handler).start()
        while True:
            self.routine()
        print('[%s] process ended'%(self.prefix))
    
    def copy_param_data(self, src, dest, p_dest=0.5):
        # Uitl for chainer network
        assert isinstance(src, chainer.Chain)
        assert isinstance(dest, chainer.Chain)
        for child in src.children():
            if child.name not in dest.__dict__: continue
            dst_child = dest[child.name]
            if type(child) != type(dst_child): continue
            if isinstance(child, chainer.Chain):
                self.copy_param_data(child, dst_child)
            if isinstance(child, chainer.Link):
                match = True
                for a, b in zip(child.namedparams(), dst_child.namedparams()):
                    if a[0] != b[0]:
                        match = False
                        break
                    if a[1].data.shape != b[1].data.shape:
                        match = False
                        break
                if not match:
                    print('Ignore %s because of parameter mismatch' % child.name)
                    continue
                for a, b in zip(child.namedparams(), dst_child.namedparams()):
                    b[1].data = p_dest*b[1].data + (1-p_dest)*a[1].data
                for pn in child._persistent:
                    a = getattr(child, pn)
                    b = getattr(dst_child, pn)
                    setattr(dst_child, pn, p_dest*b + (1-p_dest)*a)

class Agent(object):
    def __init__(self, output_path=''):
        self.output_path = output_path
        
        #self.action_count = 18 #for Montezuma 
        self.action_count = 6 # for Pong
        self.tragectory_packet_size = 10
        self.trajectory = []
        self.latest = TrajectorySegment()
        
        self.rl_proces   = None
    
    def rl_main(self, args):
        RLProcess(**args).run()
       
    def start_subprocesses(self):
        self.to_rl   = Queue()
        self.from_rl = Queue()
        args = {
                'receiver':     self.to_rl,
                'sender':       self.from_rl,
                'packet_size':  self.tragectory_packet_size,
                'output_path':  self.output_path,
                'action_count': self.action_count,
                'gpu_id':       0,
            }
        self.rl_process = Process(target=self.rl_main, args=(args,))
        self.rl_process.start()
    
    def stop_subprocesses(self):
        if not self.rl_process is None:
            self.rl_process.terminate()
        self.rl_process = None

    def select_action(self, s, a, greedy=True):
        self.to_rl.put_nowait(('get_action', (s, a, True)))
        p, v, f= self.from_rl.get()
        a = np.random.choice(range(0, self.action_count), p=p)
        info = ' '.join(['%.3f'%(float(p[i])) for i in range(0, self.action_count)]) + ' %.3f'%(v)
        return a, f, info
     
    def send_trajectory(self, traj):
        self.to_rl.put_nowait(('put_trajectory', (traj,)))
    
    def reward(self, r):
        self.latest.r = r
    
    def next_step(self, s):
        s = F.average_pooling_2d(s.transpose(2, 1, 0).astype('float32')[None, ...]/255-0.5, 2).data
        if self.latest.is_prepared():
            self.trajectory.append(self.latest)
            if len(self.trajectory) >= self.tragectory_packet_size:
                self.send_trajectory(self.trajectory)
                self.trajectory = []
        
        a, f, info = self.select_action(s, self.latest.a)
        #print(info)
        
        self.latest = TrajectorySegment(s=s[0], a=a, f=f)
        return a

class QNet(chainer.Chain):
    def __init__(self, action_count, input_dim, feature_dim=96, hidden_dim=128):
        '''(3, 210, 160)'''
        super(QNet, self).__init__(
                #
                feat1_conv = L.Convolution2D(3, 24, 3, stride=1, pad=1),
                feat2_conv = L.Convolution2D(24,24, 3, stride=1, pad=1),
                feat3_bn   = L.BatchNormalization(24, use_beta=True, use_gamma=True),
                feat4_conv = L.Convolution2D(24,24, 3, stride=1, pad=1),
                feat5_conv = L.Convolution2D(24,24, 3, stride=1, pad=1),
                #
                q1_lstm = L.StatelessLSTM(96 + action_count, hidden_dim),
                q2_fc   = L.Linear(hidden_dim, hidden_dim),
                q3_fc   = L.Linear(hidden_dim, action_count),
            )
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.action_count = action_count
        
    def logsumexp(self, x):
        x_max = F.max(x, axis=1)[:, None]
        exp_x_bar = F.exp((x - F.broadcast_to(x_max, x.data.shape)))
        sum_exp_x_bar = F.sum(exp_x_bar, axis=1)[:, None]
        return x_max + F.log(sum_exp_x_bar)
    
    def __call__(self, c, h, s, prev_a, tau, train=True):
        f = self.to_feature(s, train)
        c, h, q = self.calc_q(c, h, f, prev_a, train)
        v = tau*self.logsumexp(q/tau)
        p = (q - F.broadcast_to(v, q.data.shape))/tau
        return c, h, f, v, p
    
    def to_feature(self, s, train=True):
        # 80, 105
        y = self.feat1_conv(s)
        y = F.leaky_relu(y)
        y = F.max_pooling_2d(y, 2)
        # 40, 53
        y = self.feat2_conv(y)
        y = F.leaky_relu(y)
        y = F.max_pooling_2d(y, 2)
        # 20, 27
        y = self.feat3_bn(y, test=not train)
        y = self.feat4_conv(y)
        y = F.leaky_relu(y)
        y = F.max_pooling_2d(y, 4)
        # 5, 7
        y = self.feat5_conv(y)
        y = F.leaky_relu(y)
        y = F.max_pooling_2d(y, 4)
        # 2, 2
        y = F.reshape(y, (-1, self.feature_dim))
        return y
    
    def calc_q(self, c, h, f, prev_a, train=True):
        y = F.concat([f, prev_a], axis=1)
        c, h = self.q1_lstm(c, h, y)
        y = h
        y = self.q2_fc(y)
        y = F.leaky_relu(y)
        q = self.q3_fc(y)
        return c, h, q

class RLProcess(NetworkProcessBase):
    def __init__(self, receiver, sender, packet_size, output_path, action_count, gpu_id):
        super(RLProcess, self).__init__(receiver, sender, 'RL')

        self.packet_size  = packet_size
        self.output_dir   = output_path
        self.action_count = action_count
        self.gpu_id       = gpu_id

        self.not_updated = True
        
        self.minibatch_size = 32
        self.gamma = 0.995
        self.tau   = 0.10
        self.alpha = 0.99
        self.kappa = self.alpha*(1 - self.alpha)
        
        self.episode_length = 100
        self.d              = 10
        self.replay_buffer_max = 50000
        self.replay_buffer  = []
        
        self.QNet            = QNet(self.action_count, 80*105)
        self.QNet_learner    = QNet(self.action_count, 80*105)
        if self.gpu_id >= 0:
            self.QNet.to_gpu(self.gpu_id)
            self.QNet_learner.to_gpu(self.gpu_id)
        self.latest_QNet_h   = self.QNet.xp.zeros(shape=(1, self.QNet.hidden_dim)).astype('float32')
        self.latest_QNet_c   = self.QNet.xp.zeros(shape=(1, self.QNet.hidden_dim)).astype('float32')
        self.QNet_optimizer  = optimizers.MomentumSGD(lr=0.01)
        self.QNet_optimizer.setup(self.QNet_learner)
        self.QNet_optimizer.add_hook(GradientClipping(10))
        self.QNet_optimizer.add_hook(WeightDecay(1e-5))
        
        self.copy_param_data(self.QNet_learner, self.QNet, p_dest=0.0)
        self.write_info('parameters copied')

    def routine(self):
        trajectories = self.sample_trajectories(self.episode_length, self.minibatch_size)
        if len(trajectories) == 0:
            time.sleep(0.05)
        else:
            self.update(trajectories)
            self.copy_param_data(self.QNet_learner, self.QNet, p_dest=0.0)
            self.write_info('parameters copied')
    
    def get_action(self, s, a, greedy):
        if (not greedy) or self.not_updated:
            a = np.random.randint(0, self.action_count)
            p = np.zeros(shape=(self.action_count,)).astype('float32')
            p[a] = 1.0
            v = 0.0
            f = None
        else:
            a_vec = self.QNet.xp.zeros(shape=(1, self.action_count)).astype('float32')
            if not a is None:
                a_vec[0, a] = 1
            a_vec = Variable(a_vec, volatile=True)
            s = Variable(self.QNet.xp.asarray(s), volatile=True)
            c, h, f, v, p = self.QNet(self.latest_QNet_c, self.latest_QNet_h, s, a_vec, 
                    self.tau, train=False)
            self.latest_QNet_h = h.data
            self.latest_QNet_c = c.data
            v = float(v.data[0])
            p = np.exp(cuda.to_cpu(p.data[0]))
            f = cuda.to_cpu(f.data[0])
            #print(' '.join(['%.3f'%pp for pp in p]))
        self.sender.put((p, v, f))
        
    def put_trajectory(self, traj):
        for segm in traj:
            if segm.f is None:
                segm.f = np.zeros((self.QNet.feature_dim,)).astype('float32')
            else:
                segm.v = self.kappa*float((segm.f**2).mean())
        
        self.replay_buffer.extend(traj)
        while len(self.replay_buffer) > self.replay_buffer_max:
            self.replay_buffer.pop(0)
        self.write_info('a trajectory put. #segments=%d'%(len(self.replay_buffer)))
    
    def sample_trajectories(self, length, max_count):
        # currently uniform sampling without replace
        l = len(self.replay_buffer)
        if l < length:
            return []
        if l < length + max_count - 1:
            count = l - length + 1
            start_pos = list(range(0, count))
        else:
            count = max_count
            start_pos = np.random.choice(range(0, l - length + 1), size=count, replace=False)
            start_pos[-1] = l - length
        return [self.replay_buffer[i:i+length] for i in start_pos]
    
    def update(self, trajectories):
        mb_size = len(trajectories)
        if mb_size == 0:
            return 
        self.write_info('updating with %d mbs to begin'%(mb_size))

        g     = self.gamma
        tau   = self.tau
        alpha = self.alpha
        kappa = self.kappa
        xp = self.QNet_learner.xp
        
        # mbidx, time, ... 
        mat_r = xp.asarray([[[segm.r] for segm in traj] for traj in trajectories]).astype('float32')
        mat_s = xp.stack([xp.asarray([segm.s for segm in traj]) for traj in trajectories]).astype('float32')
        mat_a = xp.asarray([[[segm.a] for segm in traj] for traj in trajectories])
        mat_a = (mat_a == xp.arange(0, self.action_count)[None, None, :]).astype('float32')
        
        mat_f = xp.stack([xp.asarray([segm.f for segm in traj]) for traj in trajectories]).astype('float32')
        mat_v = xp.asarray([[[segm.v] for segm in traj] for traj in trajectories]).astype('float32')
        
        c_all_sum = 0.0
        self.QNet_learner.zerograds()

        QNet_h_seed = xp.zeros(shape=(mb_size, self.QNet_learner.hidden_dim)).astype('float32')
        QNet_c_seed = xp.zeros(shape=(mb_size, self.QNet_learner.hidden_dim)).astype('float32')
        QNet_h = Variable(QNet_h_seed, volatile=False)
        QNet_c = Variable(QNet_c_seed, volatile=False)

        last_action = Variable(0*mat_a[:,0,...], volatile=False)
        s = Variable(mat_s[:,0,...], volatile=False)
        QNet_c, QNet_h, pf, vs, p_all = self.QNet_learner(QNet_c, QNet_h, s, last_action, tau)
        
        mat_v[:,0,...] = alpha*mat_v[:,0,...] + (1-alpha)*kappa*((mat_f[:,0,...] - pf.data)**2).mean()
        mat_f[:,0,...] = alpha*mat_f[:,0,...] + (1-alpha)*pf.data
        
        weight = 1
        for t in range(0, self.episode_length-self.d+1, self.d-1):
            c_all = 0
            for i in range(t, t+self.d-1):
                p = F.sum(p_all*mat_a[:,i,...], axis=1)[:, None]
                c = -vs + mat_r[:,i,...] - tau*p
                
                last_action = Variable(mat_a[:,i,...], volatile=False)
                s = Variable(mat_s[:,i+1,...], volatile=False)
                QNet_c, QNet_h, f, vs, p_all = self.QNet_learner(QNet_c, QNet_h, s, last_action, tau)
                
                mat_v[:,i+1,...] = alpha*mat_v[:,i+1,...] + (1-alpha)*kappa*((mat_f[:,i+1,...] - f.data)**2).mean()
                mat_f[:,i+1,...] = alpha*mat_f[:,i+1,...] + (1-alpha)*f.data
                
                u = mat_v[:,i+1,...] #+ F.sum((f - pf)**2, axis=1)[:, None]/f.data.shape[1]
                #u = F.minimum(u, xp.ones(shape=u.shape).astype('float32'))
                u = xp.minimum(u, 1)
                c = 0.5*(c + g*vs + 0.05*u)**2
                if i != 0:
                    c_all += c*weight
                pf = f
                #weight *= g
            c_all = F.sum(c_all) / (mb_size*(self.d-1 -(t==0)))
            c_all.backward()
            c_all_sum += float(c_all.data)
            del c_all
            QNet_h = Variable(QNet_h.data, volatile=False)
            QNet_c = Variable(QNet_c.data, volatile=False)
            
        self.QNet_optimizer.update()
        
        #mat_v[:,-1,...] = alpha*mat_v[:,-1,...] + (1-alpha)*kappa*((mat_f[:,-1,...] - pf.data)**2).mean()
        #mat_f[:,-1,...] = alpha*mat_f[:,-1,...] + (1-alpha)*pf.data
        mat_f = cuda.to_cpu(mat_f)
        mat_v = cuda.to_cpu(mat_v)
        for i in range(0, len(trajectories)):
            for j in range(0, len(trajectories[i])):
                trajectories[i][j].f = mat_f[i, j,...]
                trajectories[i][j].v = float(mat_v[i, j, 0])
        
        self.not_updated = False
        self.write_info('updated c_all=%.4f'%(c_all_sum))

