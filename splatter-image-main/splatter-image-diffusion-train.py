import json
import os
import sys
import tqdm
from omegaconf import OmegaConf

import lpips as lpips_lib

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim import Adam
from utils.networkHelper import *
from Unet.UNet import Unet
from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from scene.dataset_factory import get_dataset
from utils.loss_utils import ssim as ssim_fn

class Metricator():
    def __init__(self, device):
        self.lpips_net = lpips_lib.LPIPS(net='vgg').to(device)
    def compute_metrics(self, image, target):
        lpips = self.lpips_net( image.unsqueeze(0) * 2 - 1, target.unsqueeze(0) * 2 - 1).item()
        psnr = -10 * torch.log10(torch.mean((image - target) ** 2, dim=[0, 1, 2])).item()
        ssim = ssim_fn(image, target).item()
        return psnr, ssim, lpips

def extract(v, t, x_shape):
    # v[T]
    # t[B] x_shape = [B,C,H,W]
    out = torch.gather(v, index=t, dim=0).float()
    # [B,1,1,1],分别代表batch_size,通道数,长,宽
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def ddpm_add_noise(x0,alpha,t,noise=None):
    # print("a iscuda?",alpha.is_cuda)
    alphas_cumprod = torch.cumprod(alpha, dim=0)
    # alphas_cumprod.to("cuda")
    # print("a_cup iscuda?",alphas_cumprod.is_cuda)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    # alphas_cumprod.to("cuda")
    # sqrt_alphas_cumprod.to("cuda")
    # print("sqrt_a_cup iscuda?",sqrt_alphas_cumprod.is_cuda)
    # print("t iscuda?",t.is_cuda)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    # sqrt_one_minus_alphas_cumprod.to("cuda")
    sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x0.shape
        )

    return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()
def train_Unet(model, dataloader, device, model_cfg, epoches=300, timestep=1000):
    """
    Runs evaluation on the dataset passed in the dataloader. 
    Computes, prints and saves PSNR, SSIM, LPIPS.
    Args:
        save_vis: how many examples will have visualisations saved
    """
    beta = torch.linspace(0.0001, 0.02, 1000,device = "cuda")
    alpha = 1. - beta
    alpha.to("cuda")
    Unet_union = Unet(dim=28, channels=23, dim_mults=(1, 2, 4,))
    Unet_union.to("cuda")
    optimizer = Adam(Unet_union.parameters(), lr=1e-4)
    bg_color = [1, 1, 1] if model_cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for i in range(epoches):
        for d_idx ,data in enumerate(dataloader):
            data = {k: v.to(device) for k, v in data.items()}
            rot_transform_quats = data["source_cv2wT_quat"][:, :model_cfg.data.input_images]

            if model_cfg.data.category == "hydrants" or model_cfg.data.category == "teddybears":
                focals_pixels_pred = data["focals_pixels"][:, :model_cfg.data.input_images, ...]
            else:
                focals_pixels_pred = None

            if model_cfg.data.origin_distances:
                input_images = torch.cat([data["gt_images"][:, :model_cfg.data.input_images, ...],
                                      data["origin_distances"][:, :model_cfg.data.input_images, ...]],
                                      dim=2)
            else:
                input_images = data["gt_images"][:, :model_cfg.data.input_images, ...]


        # batch has length 1, the first image is conditioning
            reconstruction = model(input_images,
                               data["view_to_world_transforms"][:, :model_cfg.data.input_images, ...],
                               rot_transform_quats,
                               focals_pixels_pred)

            t=torch.randint(0, timestep, (8,), device="cuda").long()
            data_xyz = reconstruction["xyz"]
            data_opacity = reconstruction["opacity"]
            data_scaling = reconstruction["scaling"]
            data_rotation = reconstruction["rotation"]
            data_features_dc = reconstruction["features_dc"].squeeze(2)
            data_features_rest = reconstruction["features_rest"].reshape(8,16384,9)

            # print("data_xyz",data_xyz.shape)
            # print("data_opacity",data_opacity.shape)
            # print("data_scaling",data_scaling.shape)
            # print("data_rotation",data_rotation.shape)
            # print("data_features_dc",data_features_dc.shape)
            data = torch.cat([data_xyz, data_opacity, data_scaling, data_rotation, data_features_dc,data_features_rest], dim=2)
            # print("data",data.shape)
            data = data.reshape(-1,23,128,128)
            data.to("cuda")
            noise = torch.randn_like(data)
            # print("new_xyz",data_xyz.shape)
            noisy_data = ddpm_add_noise(data,alpha=alpha,t=t,noise=noise)
            # print("noisy_xyz",noisy_data_xyz.shape)
            noise_predict = Unet_union(noisy_data, t)
            loss = torch.mean((noise_predict-noise)**2)
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            # # print("xyz",data_xyz.shape)
            # data_xyz = data_xyz.reshape(-1,3,128,128)
            # data_xyz.to("cuda")
            # noise_xyz = torch.randn_like(data_xyz)
            # # print("new_xyz",data_xyz.shape)
            # noisy_data_xyz = ddpm_add_noise(data_xyz,alpha=alpha,t=t,noise=noise_xyz)
            # # print("noisy_xyz",noisy_data_xyz.shape)
            # noise_xyz_predict = Unet_xyz(noisy_data_xyz, t)
      
            # loss_xyz = torch.mean((noise_xyz_predict-noise_xyz)**2)
            
            # optimizer.zero_grad()
            # loss_xyz.requires_grad_(True)
            # loss_xyz.backward()
            # optimizer.step()
        print("epoches:{},loss:{}".format(i,loss))
    torch.save(Unet_union.state_dict(), './Unet_union.pth')

def main(experiment_path, device_idx, split='val'):
    
    # set device and random seed
    device = torch.device("cuda:{}".format(device_idx))
    torch.cuda.set_device(device)

    # load cfg
    training_cfg = OmegaConf.load(os.path.join(experiment_path, "configs_val", "config_val.yaml"))

    # load model
    model = GaussianSplatPredictor(training_cfg)
    ckpt_loaded = torch.load(os.path.join(experiment_path, "model_latest.pth"), map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model = model.to(device)
    model.eval()
    print('Loaded model!')

    # instantiate dataset loader
    dataset = get_dataset(training_cfg, split)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                            persistent_workers=True, pin_memory=True, num_workers=1,drop_last=True)
    
    train_Unet(model, dataloader, device, training_cfg, epoches=200, timestep=1000)

if __name__ == "__main__":

    experiment_path = sys.argv[1]
    split = 'train' 
    out_folder = 'out'
    main(experiment_path, 0, split=split)







        

        # for r_idx in range( data["gt_images"].shape[1]):
        #     if "focals_pixels" in data.keys():
        #         focals_pixels_render = data["focals_pixels"][0, r_idx]
        #     else:
        #         focals_pixels_render = None
        #     image = render_predicted({k: v[0].contiguous() for k, v in reconstruction.items()},
        #                              data["world_view_transforms"][0, r_idx],
        #                              data["full_proj_transforms"][0, r_idx], 
        #                              data["camera_centers"][0, r_idx],
        #                              background,
        #                              model_cfg,
        #                              focals_pixels=focals_pixels_render)["render"]

