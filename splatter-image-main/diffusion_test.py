import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from utils.networkHelper import *
from Unet.UNet import Unet

def extract(v, t, x_shape):
    # v[T]
    # t[B] x_shape = [B,C,H,W]
    out = torch.gather(v, index=t, dim=0).float()
    # [B,1,1,1],分别代表batch_size,通道数,长,宽
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

def ddpm_add_noise(x0,alpha,t,noise=None):
    alphas_cumprod = torch.cumprod(alpha, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    alphas_cumprod.to("cuda")
    sqrt_alphas_cumprod.to("cuda")
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x0.shape
        )

    return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

def denioise_with_pretrained_model(xt,model,timestep,alpha,beta,t_index):
    t = timestep
    alphas_cumprod = torch.cumprod(alpha, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alpha)
    beta_t = extract(beta, timestep, xt.shape)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    model_mean = sqrt_recip_alphas_t * (
                x - beta_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
    posterior_variance = beta * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    if t_index == 0:
            return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
            # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

if __name__ == '__main__':
    # v = torch.randn(3, 2, 2)
    # t = torch.tensor([0, 1, 0])
    # x_shape = [3, 2, 2]
    # print(extract(v, t, x_shape))
    dataset = torchvision.datasets.FashionMNIST("./dataset", train=True, download=True, transform=transforms.ToTensor())

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    #随机生成一个batch_size=8的数据，shape为[8,128,128,1]
    beta = torch.linspace(0, 1, 10)
    alpha = 1. - beta
    # x = torch.randn(8, 3, 128, 128)
    Unet = Unet(dim=28, channels=1, dim_mults=(1, 2, 4,))
    Unet = Unet.to('cpu')
    optimizer = Adam(Unet.parameters(), lr=1e-3)
    epoches = 1
    timestep = 10
    # for i in range(epoches):
    #     x0, noise, output_list = ddpm_add_noise(x, timestep,beta=beta,alpha=alpha)
    #     noise_predict = Unet(x0, torch.randint(0, timestep, (8,), device="cpu").long())
    #     loss = torch.mean((noise_predict-noise)**2)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print("epoches:{},loss:{}".format(i,loss))
    for i in range(epoches):
        for j,(x, _) in enumerate(dataloader):
            x = x.to('cpu')
            t= torch.randint(0, timestep, (8,), device="cpu").long()
            noise = torch.randn_like(x)
            xt = ddpm_add_noise(x,alpha=alpha,t=t,noise= noise)
            noise_predict = Unet(xt, t)
            # print("x shape:",x.shape)
            # print("x0 shape:",x0.shape)
            # print("noise shape:",noise.shape)
            # print("noise_predicted shape:",noise_predict.shape)
            #计算noise_predict和noise的loss，使用MSE
            loss = torch.mean((noise_predict-noise)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoches:{},loss:{}".format(i,loss))

    print("训练完成")
test_xt = torch.randn(8, 1, 128, 128)
b=8
for i in tqdm(reversed(range(0, timestep)), desc='sampling loop time step', total=timestep):
    denoised_xt = denioise_with_pretrained_model(test_xt,Unet,torch.full((b,), i, device='cpu', dtype=torch.long),alpha,beta,i)
print("denoised_xt shape:",denoised_xt.shape)