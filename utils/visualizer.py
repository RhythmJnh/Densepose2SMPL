import visdom
import numpy as np
import cv2

class Visualizer():
    def __init__(self, args):
        self.data = dict(
            train_all_history = [],
            test_list_hist = [],
            test_screen = [],
            train_list_hist = [],
            info_len = 20,
            plot_arr = np.empty((2,0), dtype = float),
            plot_tmp = np.empty((2,0), dtype = float),
            plot_hist = np.empty((2,0), dtype = float),
            plot_hist_test = np.empty((2,0), dtype = float),
            sil = 40,
            wind = 200,
            linex = 0,
        )

        self.vis = None
        if args.use_visdom:
            try:
                if args.visdom_id == '':
                    self.vis = visdom.Visdom(server=args.visdom_ip, port=args.visdom_port, env=args.visdom_name)
                else:
                    self.vis = visdom.Visdom(server=args.visdom_ip, port=args.visdom_port, env=args.visdom_name,
                                             username=args.visdom_id, password=args.visdom_pw)
            except:
                print('Visdom connect error!')
                self.vis = None

    def load_data(self, checkpoints):
        if 'visdom_data' in checkpoints:
            self.data = checkpoints['visdom_data']

    def display_test_hist(self, txt):
        self.data['test_list_hist'] += txt.split('\n')

        if self.vis is None: pass
        try:
            self.vis.text('<br>'.join(self.data['test_list_hist']), win='test log',
                          opts=dict(title='testing log'))
        except: self.vis = None

    def display_test_clean(self):
        self.data['test_screen'] = []

    def display_test(self, txt):
        self.data['test_screen'] += txt.split('\n')

        if self.vis is None: pass
        try:
            self.vis.text('<br>'.join(self.data['test_screen']), win='testing screen',
                          opts=dict(title='testing screen'))
        except: self.vis = None

    def display_train_hist(self, info):
        self.data['train_list_hist'].append(info)
        self.data['train_all_history'].append(info)
        if len(self.data['train_list_hist']) > self.data['info_len']:
            self.data['train_list_hist'] = self.data['train_list_hist'][1:]

        if self.vis is None: pass
        try:
            self.vis.text('<br>'.join(self.data['train_list_hist']), win='disp train_list',
                          opts=dict(title='training log'))
        except: self.vis = None

    def show_image(self, image, win_id, text):
        self.data[win_id] = image
        if self.vis is None: pass
        try:
            self.vis.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1), win=win_id,
                                 opts=dict(title=text, caption=''))
        except: self.vis = None

    def plot_test_losses(self, epoch, loss):
        self.data['plot_hist_test'] = np.hstack((self.data['plot_hist_test'], np.array([[epoch], [loss]])))

        if self.vis is None: pass
        try:
            if self.data['plot_hist_test'].shape[1] > 0:
                self.vis.line(
                    X=self.data['plot_hist_test'][0, :],
                    Y=self.data['plot_hist_test'][1, :],
                    opts={
                        'title': 'test loss over epoch {}'.format(epoch),
                        'xlabel': 'epoch',
                        'legend': ['loss'],
                        'ylabel': 'loss'},
                    win='test_hist_all')
        except: self.vis = None

    def plot_train_losses(self, epoch, iteration, reduced_loss):
        self.data['plot_arr'] = np.hstack((self.data['plot_arr'], np.array([[self.data['linex']], [reduced_loss]])))
        self.data['plot_tmp'] = np.hstack((self.data['plot_tmp'], np.array([[self.data['linex']], [reduced_loss]])))
        self.data['linex'] += 1

        if self.data['plot_tmp'].shape[1] >= self.data['sil']:
            it = self.data['plot_tmp'][0, -1]
            tmp_vec = np.reshape(np.sum(self.data['plot_tmp'], axis=1), (2, 1)) / self.data['plot_tmp'].shape[1]
            tmp_vec[0, 0] = it
            self.data['plot_hist'] = np.hstack((self.data['plot_hist'], tmp_vec))
            self.data['plot_tmp'] = np.empty((2, 0), dtype=float)
        if self.data['plot_arr'].shape[1] > self.data['wind']:
            self.data['plot_arr'] = self.data['plot_arr'][:, -self.data['wind']:]

        if self.vis is None: pass
        try:
            if self.data['plot_arr'].shape[1] > 0:
                self.vis.line(
                    X=self.data['plot_arr'][0, :],
                    Y=self.data['plot_arr'][1, :],
                    opts={
                        'title': 'train loss screen ep{} it{}'.format(epoch, iteration),
                        'xlabel': 'step',
                        'legend': ['loss'],
                        'ylabel': 'loss'},
                    win='hist_realtime')
            if self.data['plot_hist'].shape[1] > 0:
                self.vis.line(
                    X=self.data['plot_hist'][0, :],
                    Y=self.data['plot_hist'][1, :],
                    opts={
                        'title': 'train loss history ep{} it{}'.format(epoch, iteration),
                        'xlabel': 'step',
                        'legend': ['loss'],
                        'ylabel': 'loss'},
                    win='hist_all')
        except: self.vis = None

