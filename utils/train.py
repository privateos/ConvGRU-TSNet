import heapq
import torch
#Trainer的设计想法参考以下链接
# https://www.jianshu.com/p/c88df856dbc8
class Trainer(object):
    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None, use_cuda=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.use_cuda = use_cuda
        if use_cuda:
            self.model = model.cuda()
            self.criterion = criterion.cuda()
        self.model.train()
        #总的迭代次数
        self.iterations = 0
        '''
        Trainer的状态，注意这里的状态包含了所有插件提供的状态。初始化为空
        '''
        self.stats = {}

        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }
        '''
        作者将插件的调用进行了分类:
        (1)iteration:一般是在完成一个batch 训练之后进行的事件调用序列（一般不改动网络或者优化器，如：计算准确率）调用序列；
        (2)batch 在进行batch 训练之前需要进行的事件调用序列
        (3)epoch 完成一个epoch 训练之后进行的事件调用序列
        (4)update 完成一个batch训练之后进行的事件(涉及到对网络或者优化器的改动,如:学习率的调整)
        
        iteration 跟update 两种插件调用的时候传入的参数不一样,iteration 会传入batch output,loss 等训练过程中的数据,
        而update传入的的model ,方便对网络的修改
        '''

    def register_plugin(self, plugin):
        #注册插件
        plugin.register(self)

        #插件的触发间隔,一般是这样的形式[(1, 'iteration'), (1, 'epoch')]
        intervals = plugin.trigger_interval

        if not isinstance(intervals, list):
            intervals = [intervals]
        for duration, unit in intervals:
            #unit 是事件的触发类别
            queue = self.plugin_queues[unit]
            '''添加事件， 这里的duration就是触发间隔,，以后在调用插件的时候，
            会进行更新  duration 决定了比如在第几个iteration or epoch 触发事件。len(queue)这里应当理解为优先级（越小越高）
            【在相同duration的情况下决定调用的顺序】，根据加入队列的早晚决定。'''
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        #调用插件
        args = (time,) + args
        #这里的time 最基本的意思是次数,如(iteration or epoch)
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            '''如果队列第一个事件的duration（也就是触发时间点）小于当前times'''
            plugin = queue[0][2]
            '''调用相关队列相应的方法，所以如果是继承Plugin类的插件，
                       必须实现 iteration、batch、epoch和update中的至少一个且名字必须一致。'''
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            '''根据插件的事件触发间隔，来更新事件队列里的事件 duration'''
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)
            '''加入新的事件并弹出最小堆的堆头。最小堆重新排序。'''

    def run(self, epochs=1, batch_size=64, shuffle=True):
        for q in self.plugin_queues.values():
            '''对四个事件调用序列进行最小堆排序。'''
            heapq.heapify(q)

        for i in range(1, epochs + 1):
            self.train(batch_size, shuffle)
            #进行每次epoch 的更新
            self.call_plugins('epoch', i)

    def train(self, *args, **kwargs):
        dataloader = self.dataset#torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        for i, data in enumerate(dataloader, self.iterations + 1):
            batch_input, batch_target = data
            if self.use_cuda:
                batch_input = batch_input.cuda()
                batch_target = batch_target.cuda()
            #在每次获取batch data 后进行更新
            self.call_plugins('batch', i, batch_input, batch_target)
            input_var = batch_input
            target_var = batch_target
            #这里是给后续插件做缓存部分数据,这里是网络输出与loss
            plugin_data = [None, None]

            def closure():
                batch_output = self.model(input_var)
                loss = self.criterion(batch_output, target_var)
                self.optimizer.zero_grad()
                loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = batch_output
                    plugin_data[1] = loss
                return loss

            self.optimizer.step(closure)
            self.call_plugins('iteration', i, batch_input, batch_target,
                              *plugin_data)
            self.call_plugins('update', i, self.model)

        self.iterations += i

class Plugin(object):
    def __init__(self, interval=None):
        if interval is None:
            interval = []
        self.trigger_interval = interval

    def register(self, trainer):
        raise NotImplementedError()