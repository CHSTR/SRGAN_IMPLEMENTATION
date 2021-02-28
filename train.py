import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm

from models import Generator, Discriminator, VGGEncoder, MobileNetV2
from torch.utils.tensorboard.writer import SummaryWriter
#from apex import amp

class Sgg(object):
    def __init__(self, model_pretrained):
        super(Sgg, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.discriminator = Discriminator({}).to(self.device)
        self.generator = Generator({}).to(self.device)
        self.logger = SummaryWriter('logs')
        if model_pretrained == "vgg":
            self.encoder = VGGEncoder().to(self.device)
        elif model_pretrained == "mobile":
            self.encoder = MobileNetV2().to(self.device)
        else:
            print('Ingresa un Encoder valido.')
            return 0
        for p in self.encoder.parameters():
            p.trainable = False

        self.loss = nn.MSELoss().to(self.device)
        self.adversarial_error = nn.BCEWithLogitsLoss().to(self.device)
        self.imagenet_mean = torch.as_tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        self.imagenet_std = torch.as_tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)


    def pretrain(self, dataset, optimizer):
        step = 0
        bar = tqdm(range(0, 100))
        for _ in bar:
            for x, y in dataset:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                z = self.generator(x)
                loss = self.loss(z, y)
                loss.backward()
                optimizer.step()
                bar.set_description(f'Loss: {loss.item()}')
                step += 1
        torch.save(self.generator.state_dict(), 'generator_test.pt')

    def normalize(self, batch_tensor):
        batch_tensor -= self.imagenet_mean
        return batch_tensor / self.imagenet_std

    def generator_step(self, x, y):
        self.encoder.eval()
        z = self.generator(x)
        #print(z.shape)
        fake_prediction = self.discriminator(z)

        fake_features = self.encoder(self.normalize((z + 1) / 2.0))
        #print(fake_features.shape)
        real_features = self.encoder(self.normalize((y + 1) / 2.0)).detach() #no se necesita para el proceso del gradiente.
        #print(real_features.shape)

        loss_1 = self.loss(fake_features, real_features)
        loss_2 = self.adversarial_error(fake_prediction, (torch.ones(fake_prediction.shape).to(self.device) - torch.rand(fake_prediction.shape).to(self.device)* 0.2).type_as(fake_prediction))

        loss_2 = 0.001 * loss_2
        content_loss = 1 * loss_1
        g_loss = loss_2 + content_loss
        
        return g_loss, z

    def discriminator_step(self, z, y):
        real_prediction = self.discriminator(y)
        real_loss = self.adversarial_error(real_prediction, (torch.ones(real_prediction.shape).to(self.device) - torch.rand(real_prediction.shape).to(self.device) * 0.2).type_as(real_prediction))
        
        fake_prediction = self.discriminator(z.detach())
        fake_loss = self.adversarial_error(fake_prediction, (torch.ones(fake_prediction.shape).to(self.device)  - torch.rand(fake_prediction.shape).to(self.device) * 0.3).type_as(fake_prediction))

        d_loss = real_loss + fake_loss
        return d_loss


    def train(self, train_data, value_data, fp16, epochs):
        #self.generator.load_state_dict(torch.load('/media/csr/NVDEM/sgg/generator_loss-2.440833431703073.pt'))
        opt_g, opt_d = self.configure_optimizers(fp16)
        bar = tqdm(range(epochs)) # epochs
        step = 0
        for epoch in bar:
            self.generator.train()
            self.discriminator.train()
            g_total_loss, d_total_loss = [], []
            for x, y in train_data:
                #print(x.shape)
                #print(y.shape)
                ## Generator step ##
                x, y = x.to(self.device), y.to(self.device)
                opt_g.zero_grad()
                loss_g, image_g = self.generator_step(x, y)
                if not fp16:
                    loss_g.backward()
                else:
                    loss_g.backward(retain_graph=True)
                    #with amp.scale_loss(loss_g, opt_g) as scaled_loss_g:
                    #    scaled_loss_g.backward()
                opt_g.step()

                ## Discriminator step ##
                opt_d.zero_grad()
                loss_d = self.discriminator_step(image_g, y)
                if not fp16:
                    loss_d.backward()
                else:
                    loss_d.backward(retain_graph=True)
                    #with amp.scale_loss(loss_d, opt_d) as scaled_loss_d:
                    #    scaled_loss_d.backward()
                opt_d.step()

                g_total_loss.append(loss_g.item())
                d_total_loss.append(loss_d.item())
                bar.set_description(f'Step: {step}, G Loss: {loss_g.item()} | D Loss: {loss_d.item()}')
                step += 1

            # Terminanda la pasada con los datos de entrenamiento...

            g_avg_loss = sum(g_total_loss) / len(g_total_loss)
            d_avg_loss = sum(d_total_loss) / len(d_total_loss)

            # Evaluar los avances.

            self.generator.eval()
            self.discriminator.eval()

            bar.set_description(f'Epoch: {epoch}, G Loss: {g_avg_loss}, D Loss: {d_avg_loss}')
            for x, y in value_data:
                x, y = x.to(self.device), y.to(self.device)
                loss_g, image_g = self.generator_step(x, y)
                bar.set_description(f'Validated Step: {step}, G Loss: {loss_g.item()}')
                # x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
            
            self.log_val_loss(loss_g, epoch)
            self.log_train_losses(g_avg_loss, d_avg_loss, epoch)
            self.log_images(x, y, image_g, epoch)

            if epoch % 10 == 0:
            	torch.save(self.generator.state_dict(), "generator_vgg_loss-{}.pt".format(g_avg_loss))
            step = 0

    def log_train_losses(self, g_loss, d_loss, epoch):
        self.logger.add_scalar('Train/GeneratorLoss', g_loss, epoch)
        self.logger.add_scalar('Train/DiscriminatorLoss', d_loss, epoch)

    def log_val_loss(self, g_loss, epoch):
        self.logger.add_scalar('Val/GeneratorLoss', g_loss, epoch)

    def log_images(self, low_res, high_res, super_res, epoch):
        #print(super_res.shape)
        #print(high_res.shape)
        self.logger.add_image('Generada',
                                make_grid(super_res,
                                        scale_each=True,
                                        normalize=True),
                                epoch)
        self.logger.add_image('lr',
                                make_grid(low_res,
                                        scale_each=True,
                                        normalize=True),
                                epoch)
        self.logger.add_image('hr',
                                make_grid(high_res,
                                        scale_each=True,
                                        normalize=True),
                                epoch)
        self.logger.flush()

    def configure_optimizers(self, fp16):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=0.0002,
                                 weight_decay=0.002)
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=0.0001,
                                 weight_decay=0.001)
        if fp16:
            model, opt_g = amp.initialize(self.generator, opt_g, opt_level="O1")
            model, opt_d = amp.initialize(self.discriminator, opt_d, opt_level="O1")

        return opt_g, opt_d


