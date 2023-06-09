import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import generate_loader
from option import get_option
from model import Generator, Discriminator
from tqdm import tqdm


class Solver():
    def __init__(self, opt):
        self.opt = opt
        self.img_size = opt.input_size
        self.c_dim = opt.c_dim
        self.selected_attrs = opt.selected_attrs

        self.cls_lambda = opt.cls_lambda
        self.rec_lambda = opt.rec_lambda
        self.gp_lambda = opt.gp_lambda
        self.n_critic = opt.n_critic

        self.dev = torch.device("cuda:{}".format(
            opt.gpu) if torch.cuda.is_available() else "cpu")
        print("device: ", self.dev)

        self.generator = Generator(c_dim=self.c_dim)
        self.discriminator = Discriminator(
            img_size=self.img_size, c_dim=self.c_dim)

        if opt.pretrained:
            load_path = os.path.join(opt.chpt_root, "Gen.pt")
            self.generator.load_state_dict(torch.load(load_path))

            load_path = os.path.join(opt.chpt_root, "Disc.pt")
            self.discriminator.load_state_dict(torch.load(load_path))

        if opt.multigpu:
            self.generator = nn.DataParallel(
                self.generator, device_ids=self.opt.device_ids).to(self.dev)
            self.discriminator = nn.DataParallel(
                self.discriminator, device_ids=self.opt.device_ids).to(self.dev)

        print("# Generator params:", sum(
            map(lambda x: x.numel(), self.generator.parameters())))
        print("# Discriminator params:", sum(
            map(lambda x: x.numel(), self.discriminator.parameters())))

        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        self.train_loader = generate_loader(opt)
        print("train set ready")

    def fit(self):
        opt = self.opt
        print("start training")

        for epoch in range(opt.n_epoch):
            loop = tqdm(self.train_loader)

            for i, (img, label) in enumerate(loop):
                x_real, label_org = img, label

                # Generate target domain labels ramdonly
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                c_org = label_org.clone()
                c_trg = label_trg.clone()

                x_real = x_real.to(self.dev)        # Input images
                c_org = c_org.to(self.dev)          # Original domain labels
                c_trg = c_trg.to(self.dev)          # Target Domain labels
                # Labels for computng classification loss
                label_org = label_org.to(self.dev)
                # Labels for computng classification loss
                label_trg = label_trg.to(self.dev)

                # Train Discriminator
                out_src, out_cls = self.discriminator(x_real)
                D_loss_real = -torch.mean(out_src)
                D_loss_cls = self.classification_loss(out_cls, label_org)

                x_fake = self.generator(x_real, c_trg)
                out_src, out_cls = self.discriminator(x_fake.detach())
                D_loss_fake = torch.mean(out_src)

                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.dev)
                x_hat = (alpha * x_real.data + (1 - alpha)
                         * x_fake.data).requires_grad_(True)
                out_src, _ = self.discriminator(x_hat)
                D_loss_gp = self.gradient_penalty(out_src, x_hat)

                D_loss = D_loss_real + D_loss_fake + \
                    (self.cls_lambda * D_loss_cls) + \
                    (self.gp_lambda * D_loss_gp)
                self.optimizer_D.zero_grad()
                D_loss.backward()
                self.optimizer_D.step()

                # Train Generator
                if (i+1) % self.n_critic == 0:
                    x_fake = self.generator(x_real, c_trg)
                    out_src, out_cls = self.discriminator(x_fake)
                    G_loss_fake = -torch.mean(out_src)
                    G_loss_cls = self.classification_loss(out_cls, label_trg)

                    x_recon = self.generator(x_fake, c_org)
                    G_loss_rec = torch.mean(torch.abs(x_real - x_recon))

                    G_loss = G_loss_fake + \
                        (self.rec_lambda * G_loss_rec) + \
                        (self.cls_lambda * G_loss_cls)
                    self.optimizer_G.zero_grad()
                    G_loss.backward()
                    self.optimizer_G.step()

            print(
                f"[Epoch {epoch+1}/{opt.n_epoch}] [D loss: {D_loss.item():.6f}] [G loss: {G_loss.item():.6f}]")

            if (epoch+1) % 25 == 0:
                self.save()

    def save(self):
        os.makedirs(os.path.join(self.opt.ckpt_root), exist_ok=True)
        G_save_path = os.path.join(self.opt.ckpt_root, "Gen.pt")
        D_save_path = os.path.join(self.opt.ckpt_root, "Disc.pt")
        torch.save(self.generator.state_dict(), G_save_path)
        torch.save(self.discriminator.state_dict(), D_save_path)

    def classification_loss(self, logit, target):
        return nn.functional.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def gradient_penalty(self, y, x):
        weight = torch.ones(y.size()).to(self.dev)
        dydx = torch.autograd.grad(
            outputs=y, inputs=x, grad_outputs=True, create_graph=True, only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))

        return torch.mean((dydx_l2norm - 1)**2)


def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    solver = Solver(opt)
    solver.fit()


if __name__ == "__main__":
    main()
