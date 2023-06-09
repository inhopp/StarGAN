import os
import torch
from data import generate_loader
from option import get_option
from model import Generator
import torchvision.transforms as transforms
from torchvision.utils import save_image


@torch.no_grad()
def main(opt):
    dev = torch.device("cuda:{}".format(opt.gpu)
                       if torch.cuda.is_available() else "cpu")
    ft_path = os.path.join(opt.ckpt_root, "Gen.pt")

    model = Generator(c_dim=5).to(dev)
    model.load_state_dict(torch.load(ft_path))

    data_loader = generate_loader(opt)

    data_iter = iter(data_loader)
    x_fixed, c_org = next(data_iter)
    x_fixed = x_fixed.to(dev)
    c_fixed_list = create_labels(
        c_org=c_org, c_dim=5, selected_attrs=opt.selected_attrs)

    c_fixed_list_dev = []
    for i in range(len(c_fixed_list)):
        c_fixed_list_dev.append(c_fixed_list[0].to(dev))

    x_fake_list = [x_fixed]

    for c_fixed in c_fixed_list_dev:
        x_fake_list.append(model(x_fixed, c_fixed))

    x_concat = torch.cat(x_fake_list, dim=3)
    sample_path = os.path.join("./sample_image.jpg")
    sample_img = (x_concat.data.cpu() + 1) / 2
    sample_img = sample_img.clamp_(0, 1)

    save_image(sample_img, sample_path, nrow=1, padding=0)

    print("########## inference Finished ###########")


def create_labels(c_org, c_dim=5, selected_attrs=None):
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        c_trg = c_org.clone()

        # Set one hair color to 1 and the rest to 0.
        if i in hair_color_indices:
            c_trg[:, i] = 1
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0

        else:
            c_trg[:, i] = (c_trg[:, i] == 0)

        c_trg_list.append(c_trg)

    return c_trg_list


if __name__ == '__main__':
    opt = get_option()
    main(opt)
