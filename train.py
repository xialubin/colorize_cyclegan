from data import data_loader
from model import Model
from option import option

opt = option()
iter_total = 0
epoch = 0
models = Model(opt)
for epoch in range(opt.epoch, opt.n_epochs):
    data = data_loader(opt.dataset, 'resize', batchsize=8)
    iter_batch = 0
    epoch += 1

    for i, batch in enumerate(data):
        models.set_input(batch)
        models.optimize_params()

        loss_G = models.loss_G
        loss_D_A = models.loss_D_A
        loss_D_B = models.loss_D_B
        loss_GAN = models.loss_GAN
        loss_cycle = models.loss_cycle

        loss_g = loss_G.to("cpu").data.numpy()
        loss_da = loss_D_A.to("cpu").data.numpy()
        loss_db = loss_D_B.to("cpu").data.numpy()
        loss_gan = loss_GAN.to("cpu").data.numpy()
        loss_c = loss_cycle.to("cpu").data.numpy()

        # loss_g = loss_g.data.numpy()

        print("epoch: %s; batch_iter: %s; loss_G: %s; loss_GAN: %s; loss_cycle: %s" % (epoch, iter_batch, loss_g, loss_gan, loss_c))
        iter_batch += 1
        iter_total += 1

        if iter_total % 1000 == 0:
            models.save_network()
