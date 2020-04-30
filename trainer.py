import torch
import torch.utils.data as Data
import torchvision.utils as vutils
import torch.nn as nn
from datahandler import DataHandler
import torch.optim as optim
from tensorboardX import SummaryWriter
import logging
import os
import datetime
from tqdm import tqdm
from model import MstNet
from VGG16 import VGG16
import numpy as np
from utils import gram_matrix, add_mean_std




class Trainer:
    def __init__(
        self,
        content_dir,
        style_dir,
        log_dir='./log',
        weight_dir='./weight',
        learn_rate=1e-3,
        batch_size=128,
        num_workers=0,
        max_n_weights=10,
        style_weight=10000,
        content_weight=1,
        tv_weight=1e-6,
        vision_steps=1000,
        save_steps=1000,
        target_size=256,
        cuda=True
    ):

        self.vgg = VGG16()
        self.vgg.eval()
        self.mst = MstNet()
        self.log_dir = log_dir
        self.weight_dir = weight_dir
        self.max_n_weights = max_n_weights
        self.loss_accord = []

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if not os.path.exists(self.weight_dir):
            os.mkdir(self.weight_dir)

        logging.basicConfig(level=logging.DEBUG,
                            filename=os.path.join(self.log_dir, 'Trainer_{}.log'.format(
                                datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                            )),
                            filemode='w',
                            format=
                            '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                            )
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(logdir=self.log_dir)
        self.add_graph = True
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.content_weight = content_weight
        self.mse = nn.MSELoss()
        self.cuda = cuda and torch.cuda.is_available()
        self.vision_steps = vision_steps
        self.save_steps = save_steps


        # Optimizers
        self.optimizer = torch.optim.Adam(self.mst.parameters(), self.learn_rate)

        self.content_dh = DataHandler(content_dir, target_size=256, partly=80000)
        self.style_dh = DataHandler(style_dir, target_size=256, partly=100)

        self.content_ddh = Data.DataLoader(
            self.content_dh, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


        #rands = torch.rand(3, 3, target_size, target_size)
        if self.cuda:
            self.mst.cuda()
            self.vgg.cuda()
            #rands = rands.cuda()

        # self.writer.add_graph(self.mst, rands)
        # del rands

    def train_kernel(self, content_batch, style_image):

        self.mst.train()

        # Reset gradients
        self.optimizer.zero_grad()

        # self.mst.setTarget(style_image)
        trans_batch = self.mst(content_batch, style_image, True)

        style_features = self.vgg(style_image)
        content_features = self.vgg(content_batch)
        trans_features = self.vgg(trans_batch)

        # content loss
        content_loss = self.content_weight * self.mse(trans_features[1], content_features[1])

        #style loss
        style_loss = .0
        for ii, features in enumerate(trans_features):
            trans_gram = gram_matrix(features)
            style_gram = gram_matrix(style_features[ii].expand_as(features))
            style_loss += self.mse(trans_gram, style_gram)
        style_loss *= self.style_weight

        #total variation loss
        y = trans_batch
        tv_loss = self.tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                    torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

        # 求和
        loss = content_loss + style_loss + tv_loss

        loss.backward()
        # Update weights with gradients
        self.optimizer.step()

        return {"loss": loss, "content_loss": content_loss,
                "style_loss": style_loss, "tv_loss": tv_loss}
        # return {"loss": loss, "content_loss": content_loss,
        #         "style_loss": style_loss}

    def save_weights(self, epoch):

        self.mst.eval()
        self.mst.cpu()

        torch.save({'epoch': epoch,
                    'model_state_dict': self.mst.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   os.path.join(self.weight_dir, 'mst_epoch_{}.pkl'.format(epoch + 1)))
        if self.cuda:
            self.mst.cuda()

    def visual(self, content_batch, style_image, n_iter):
        self.mst.eval()
        #self.mst.setTarget(style_image, style_image, True)
        visualization_transformed_images = self.mst(content_batch, style_image, True)
        self.writer.add_images('Style', add_mean_std(style_image), n_iter)
        self.writer.add_images('Content', add_mean_std(content_batch), n_iter)
        self.writer.add_images('Transfer', add_mean_std(visualization_transformed_images), n_iter)
        del visualization_transformed_images

    def visual_params(self, n_iter):

        for name, param in self.mst.named_parameters():
            self.writer.add_histogram('MSTNet.' + name, param.clone().cpu().data.numpy(),
                                 n_iter, bins='auto')

    def train(self, epochs):
        for epoch in range(epochs):
            with tqdm(desc='epoch %d' % epoch, total=len(self.content_ddh)) as pbar:
                for i, content_batch in enumerate(self.content_ddh):
                    style_image = self.style_dh[np.random.randint(0, len(self.style_dh), 1)[0]].unsqueeze(0)

                    # 获取输入
                    if self.cuda:
                        content_batch = content_batch.cuda()
                        style_image = style_image.cuda()

                    loss_dict = self.train_kernel(content_batch, style_image)

                    self.writer.add_scalar('loss/loss', loss_dict['loss'].item(), epoch * len(self.content_ddh) + i)
                    self.writer.add_scalar('loss/loss_content', loss_dict['content_loss'].item(), epoch * len(self.content_ddh) + i)
                    self.writer.add_scalar('loss/loss_style', loss_dict['style_loss'].item(), epoch * len(self.content_ddh) + i)
                    self.writer.add_scalar('loss/loss_tv', loss_dict['tv_loss'].item(), epoch * len(self.content_ddh) + i)

                    if (epoch * len(self.content_ddh) + i + 1) % self.vision_steps == 0:
                            self.visual(content_batch, style_image, epoch * len(self.content_ddh) + i)

                    if (epoch * len(self.content_ddh) + i + 1) % self.vision_steps == 0:
                            self.visual_params(epoch * len(self.content_ddh) + i)

                    pbar.update()
            if (epoch + 1) % self.save_steps == 0:
                self.save_weights(epoch)

















