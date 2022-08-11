from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip, emotion
import audio

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

import wandb

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITHOUT the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', required=True, type=str)

parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2]//2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y

def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

recon_loss = nn.L1Loss()
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

def get_emo_loss(x, emo_idx):
    x = x.permute(0,2,1,3,4).clone().to(device)
    x = x.contiguous().view(-1, *x.shape[2:])
    x = F.interpolate(x, 224, mode='bilinear', align_corners=False)

    emo_probs = emotion.emo_probs(x, emo_idx) 
    emo_loss = 1 - emo_probs
    emo_loss = emo_loss.mean()

    print(f"emo_loss is: {emo_loss}")

    return emo_loss

# def get_gt_idx(gt):
#     gt = gt.permute(0,2,1,3,4)
#     gt = gt.contiguous().view(-1, *gt.shape[2:])
#     gt = F.interpolate(gt, 224, mode='bilinear', align_corners=False)

#     emo_idx = emotion.emo_idx(gt) 


#     print(f"emo_loss is: {emo_idx}")

#     return emo_idx.item()


def get_grad_norm(model):
    # from https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/2
    norm_type = 2
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), 2.0).item()

    return total_norm

def clamp_grad_norm_(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
    clip_coef = hparams.disc_max_grad_norm / (total_norm + 1e-6)
    stretch_coef = hparams.disc_min_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
            for p in parameters:
                        p.grad.detach().mul_(clip_coef.to(p.grad.device))
    elif stretch_coef > 1.0:
            for p in parameters:
                        p.grad.detach().mul_(stretch_coef.to(p.grad.device))

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step

    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss, = 0., 0.
        running_emo_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))


        #### MINE ####
        n_steps_per_epoch = len(train_data_loader)
        print(f"NUM SAMPLE IN TRAIN_SET: {n_steps_per_epoch}")
        #### END  ####
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            print(f"global epoch is: {global_epoch}")
            model.train()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            ### Train generator now. Remove ALL grads. 
            optimizer.zero_grad()

            g = model(indiv_mels, x)

            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.


            if hparams.emo_wt > 0.:
                # get_gt_idx(gt) # sanity check
                emo_loss = get_emo_loss(g, hparams.train_emo_idx)
            else:
                emo_loss = 0.

            l1loss = recon_loss(g, gt)

            loss = hparams.syncnet_wt * sync_loss + \
                                    (1. - hparams.syncnet_wt - hparams.emo_wt) * l1loss + \
                                        hparams.emo_wt * emo_loss


            loss.backward()
            optimizer.step()


            # if hparams.disc_max_grad_norm and hparams.disc_min_grad_norm: 
                 
            #     clamp_grad_norm_(disc.parameters())

            # disc_grad_norm = get_grad_norm(disc)

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            # Logs
            global_step += 1
            cur_session_steps = global_step - resumed_step

            running_l1_loss += l1loss.item()
            if hparams.syncnet_wt > 0.:
                print("sync_loss is on!")
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if hparams.emo_wt > 0.:
                running_emo_loss += emo_loss.item()
            else:
                running_emo_loss += 0.

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)


            # if global_step % hparams.eval_interval == 0:
            #     with torch.no_grad():
            #         average_sync_loss = eval_model(test_data_loader, global_step, device, model, disc)

            #         if average_sync_loss < .75:
            #             hparams.set_hparam('syncnet_wt', 0.03)


             #### MINE ####
            metrics = { 
                        "train/loss": loss,
                        "train/l1loss": l1loss,
                        "train/sync_loss": sync_loss,
                        "train/emo_loss": emo_loss,
                        # "train/disc_grad_norm": disc_grad_norm,
                        "train/running_l1_loss": running_l1_loss / (step + 1),
                        "train/running_sync_loss": running_sync_loss / (step + 1),
                        "train/running_emo_loss": running_emo_loss / (step + 1),
                       "train/epoch": (step + 1 + (n_steps_per_epoch * global_epoch)) / n_steps_per_epoch


                    }
            
            if step + 1 < n_steps_per_epoch:
                # 🐝 Log train metrics to wandb 
                wandb.log(metrics)
            #### END ####

         ## Validate Epoch
            if global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss, averaged_recon_loss = eval_model_2(test_data_loader, global_step, device, model, checkpoint_dir)

                    #### MINE ####
                    val_metrics = {"val/average_sync_loss": average_sync_loss,
                                    "val/averaged_recon_loss": averaged_recon_loss}
                    wandb.log({**metrics, **val_metrics})
                    #### END ####

                    if global_epoch > 50:
                        hparams.set_hparam('syncnet_wt', 0.01) #no gan  
                        
                    if global_epoch > 30:
                        hparams.set_hparam('emo_wt', 0.19)


            prog_bar.set_description('L1: {}, Sync: {}, emo_loss: {}'.format(running_l1_loss / (step + 1),
                                                                                        running_sync_loss / (step + 1),
                                                                                        running_emo_loss / (step +1)))
            

            
                

        global_epoch += 1

# def eval_model(test_data_loader, global_step, device, model, disc):
#     eval_steps = 300
#     print('Evaluating for {} steps'.format(eval_steps))
#     running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss = [], [], [], [], []
#     while 1:
#         for step, (x, indiv_mels, mel, gt) in enumerate((test_data_loader)):
#             model.eval()
#             disc.eval()

#             x = x.to(device)
#             mel = mel.to(device)
#             indiv_mels = indiv_mels.to(device)
#             gt = gt.to(device)

#             pred = disc(gt)
#             disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))

#             g = model(indiv_mels, x)
#             pred = disc(g)
#             disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))

#             running_disc_real_loss.append(disc_real_loss.item())
#             running_disc_fake_loss.append(disc_fake_loss.item())

#             sync_loss = get_sync_loss(mel, g)
            
#             if hparams.disc_wt > 0.:
#                 perceptual_loss = disc.perceptual_forward(g)
#             else:
#                 perceptual_loss = 0.
            
#             if hparams.emo_wt > 0.:
#                 emo_loss = get_emo_loss(g)
#             else:
#                 emo_loss = 0.


#             l1loss = recon_loss(g, gt)

#             loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
#                                     (1. - hparams.syncnet_wt - hparams.disc_wt) * l1loss

#             running_l1_loss.append(l1loss.item())
#             running_sync_loss.append(sync_loss.item())
            
#             if hparams.disc_wt > 0.:
#                 running_perceptual_loss.append(perceptual_loss.item())
#             else:
#                 running_perceptual_loss.append(0.)

#             if step > eval_steps: break

#         print('L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}'.format(sum(running_l1_loss) / len(running_l1_loss),
#                                                             sum(running_sync_loss) / len(running_sync_loss),
#                                                             sum(running_perceptual_loss) / len(running_perceptual_loss),
#                                                             sum(running_disc_fake_loss) / len(running_disc_fake_loss),
#                                                              sum(running_disc_real_loss) / len(running_disc_real_loss)))
#         return sum(running_sync_loss) / len(running_sync_loss)

def eval_model_2(test_data_loader, global_step, device, model, checkpoint_dir):    
    # print('Evaluating for {} steps'.format(len(test_data_loader)))
    sync_losses, recon_losses = [], []
    for x, indiv_mels, mel, gt in test_data_loader:
        # step += 1
        model.eval()

        # Move data to CUDA device
        x = x.to(device)
        gt = gt.to(device)
        indiv_mels = indiv_mels.to(device)
        mel = mel.to(device)

        g = model(indiv_mels, x)

        sync_loss = get_sync_loss(mel, g)
        l1loss = recon_loss(g, gt)

        sync_losses.append(sync_loss.item())
        recon_losses.append(l1loss.item())

    averaged_sync_loss = sum(sync_losses) / len(sync_losses)
    averaged_recon_loss = sum(recon_losses) / len(recon_losses)

    print(f'Val on N={len(sync_losses)} samples: [L1: {averaged_recon_loss}, Sync loss: {averaged_sync_loss}]')
    return averaged_sync_loss,averaged_recon_loss


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")

     # Model
    model = Wav2Lip().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5, 0.999))
  
    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

        
    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, 
                                overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    emotion.init(device)

    """
    My wandb code
    """
    wandb.init(
        project="wav2lip_VOX_HAP_NO_DISC",
        entity="talkingfacegen",
        config={
            "nepochs": 1000,
            "batch_size": 16,
            "initial_learning_rate": 1e-4,
            "lr": 1e-4,
            })
    # Train!
    train(device, model, train_data_loader, test_data_loader, optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)
