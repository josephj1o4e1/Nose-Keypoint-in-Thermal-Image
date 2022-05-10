if __name__ == '__main__':   
    import argparse
    import os
    import sys
    import numpy as np
    import math
    import cv2
    import datetime
    import time
    from skimage import io
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torch.autograd import Variable

    from datasets_nosetip import *
    from model_nosetip import *
    import torch.nn as nn
    import torch

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.fastest = True

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="Joseph", help="name of the dataset")
    parser.add_argument("--datasetroot", type=str, required=True, help="dataset root path")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--g_lr", type=float, default=0.00001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between saving generator samples")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
    parser.add_argument("--img_height", type=int, default=224, help="size of image height")
    parser.add_argument("--img_width", type=int, default=224, help="size of image width")
    parser.add_argument("--channels", type=int, default=4, help="number of image channels")
    parser.add_argument("--model", type=str, default="mobilenetv2", help="mobilenetv2, efficientnet_v2_s")
    parser.add_argument("--pretrained_model_path", type=str, default="", help="the pretrainedmodel's path. Ex: saved_models/...pth")
    opt = parser.parse_args()
    print(opt)

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    os.makedirs("saved_models/%s/%s/%s" % ('dataset_'+opt.dataset_name, opt.model, current_time), exist_ok=True)
    os.makedirs("saved_models_val/%s/%s/%s" % ('dataset_'+opt.dataset_name, opt.model, current_time), exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter       
    if opt.pretrained_model_path != "":
        writer = SummaryWriter(comment = current_time + 'pretrained_' + opt.model)
    else:
        writer = SummaryWriter(comment = current_time + opt.model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device cuda or cpu? : {device}')
    
    if opt.model=="mobilenetv2" :
        generator = nosetip_mobilenetv2()
        opt.channels = 3
    elif opt.model=="efficientnet_v2_s":
        generator = nosetip_effnetv2_s()
        opt.channels = 3
    else :
        sys.exit(f"--model should be: {opt.model}")

    if opt.pretrained_model_path != "":
        generator.load_state_dict(torch.load(opt.pretrained_model_path))
    else :
        generator.apply(weights_init)

    
    generator = generator.to(device) # move model to gpu or cpu
    
    # Loss functions
    # mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    # Optimizers
    optimizerAdam = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
    optimizerSGDmom = torch.optim.SGD(generator.parameters(), lr=0.001, momentum=0.9)

    Tensor = torch.cuda.FloatTensor if device=='cuda' else torch.Tensor


    input_shape = (opt.channels, opt.img_height, opt.img_width)
    orig_shape = (768, 1024) # for keypoint resize purposes
    datasetroot = opt.datasetroot
    dataset = FaceLandmarksDataset(datasetroot, input_shape)

    validation_split = .2
    indices = list(range(len(dataset)))
    split = int(np.floor(validation_split * len(dataset)))
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        # shuffle=True, # sampler option is mutually exclusive with shuffle
        num_workers=opt.n_cpu,
        sampler=train_sampler,
    )
    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        sampler=val_sampler,
    )

    
    def denormalize_imgtensor_to_numpy(tensor_img):
        tmp_A = torch.squeeze(tensor_img.detach().clone().cpu())
        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])
        tmp_A = tmp_A * STD[:, None, None] + MEAN[:, None, None]
        tmp_A = tmp_A.numpy().transpose(1, 2, 0)
        return tmp_A
    
    def denormalize_kptstensor_to_numpy(tensor_kpts):
        tensor_kpts = torch.reshape(tensor_kpts.detach().clone().cpu(), (-1, 2))
        inp_h, inp_w = input_shape[-2], input_shape[-1]
        kplist=[]
        for i, kp in enumerate(tensor_kpts):
            denorm_kp = ((kp[0] + 1)*(inp_h-0)/2 + 0, (kp[1] + 1)*(inp_w-0)/2 + 0)
            kplist.append(torch.as_tensor(denorm_kp, device=torch.device('cpu')))
        return np.array(torch.stack(kplist))


    def save_tensorimg_withkeypoints(real_A, fake_B, epoch, batches_done, batch_id):
        tmp_A = denormalize_imgtensor_to_numpy(real_A)
        tmp_B = denormalize_kptstensor_to_numpy(fake_B)

        numpy_image = vis_keypoints(tmp_A, tmp_B)*255
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        
        imgfolder_path = os.path.join('image_results', 'validation_'+current_time, 'epoch_'+str(epoch))
        img_path = os.path.join(imgfolder_path, f'val_batchdone{batches_done}_{batch_id}.png')
        os.makedirs(imgfolder_path, exist_ok = True)
        cv2.imwrite(img_path, numpy_image)
        
    
    def sample_images(epoch, batches_done):
        """Saves a generated sample from the validation set"""    
        total_loss = 0.0
        generator.eval()
        for batch_id, vdata in enumerate(val_dataloader):
            real_A, real_B = vdata["image"].type(Tensor), vdata["landmark"].type(Tensor)
            real_B = torch.flatten(real_B)
            with torch.no_grad():
                fake_B = generator(real_A)
            total_loss += mse_loss(fake_B, real_B).item() * real_A.size(0) # real_A.size(0) is the batch size. Avoid loss error when averaging batches that have different batchsizes(especially the last batch). 
            if batch_id%50==0:
                save_tensorimg_withkeypoints(real_A, fake_B, epoch, batches_done, batch_id)
                # save_tensorimg_withkeypoints(real_A, real_B, epoch, batches_done, batch_id)
        
        val_loss = total_loss/len(val_dataloader.dataset) # dataset __len__ = len(self.files_A), not batches, unlike train loss len(dataloader)
        generator.train()
        return val_loss



    # ----------
    #  Training
    # ----------
    min_val = 1000000
    min_epoch = opt.epoch

    prev_time = time.time()
    running_loss = 0.0

    lastbatch_count = 0
    
    generator.train()
    for epoch in range(opt.epoch, opt.n_epochs):
        print("\nEpoch   %d/%d" %(epoch, opt.n_epochs))
        print("-----------------------------------")
        for i, batch in enumerate(dataloader):
            
            # Set model input
            real_A, real_B = batch["image"].type(Tensor), batch["landmark"].type(Tensor)
            real_B = torch.flatten(real_B) # flatten keypoints from (9,2) to (18)

            # -------------------------------
            #  Train Generator
            # -------------------------------

            optimizerAdam.zero_grad()
            # optimizerSGDmom.zero_grad()
            fake_B = generator(real_A)
            
            # Pixelwise loss
            loss_pixel_LR = mse_loss(fake_B, real_B) # added pixelwise mse loss, we can add different loss together if we want. 

            # ----------------------------------
            # Total Loss (Generator)
            # ----------------------------------

            loss_G = loss_pixel_LR # we can expand our loss function here loss_G = loss1+loss2+loss3. 
            running_loss += loss_G.item()

            loss_G.backward()
            optimizerAdam.step()
            # optimizerSGDmom.step()




            # --------------
            #  Validation Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()            
            lastbatch_count += 1

            if batches_done % opt.sample_interval == 0 and batches_done != 0:
                print("\n******************************************************************")
                print("Validation stage:")
                print("calculating val/train MSE loss...")
                sampletime0 = time.time()
                val_score = sample_images(epoch, batches_done)
                print(f'sampletime = {time.time()-sampletime0}')
                train_score = running_loss/opt.sample_interval
                print("write on tensorboard...")
                writer.add_scalar('Train/train_score',train_score,batches_done)
                writer.add_scalar('Val/val_score',val_score,batches_done)
                writer.flush() # if too slow then flush every epoch. now, it is every opt.sample_interval batches
                if epoch >= 5 and val_score < min_val:
                    print("save validation generator...")
                    torch.save(generator.state_dict(), "saved_models_val/%s/%s/%s/generator_%d.pth" % ('dataset_'+opt.dataset_name, opt.model, current_time, epoch)) # .module
                    min_val = val_score
                    min_epoch = epoch                
                print("[Epoch %d/%d] [Batch %d/%d]\n[train MSE Loss: %f, val MSE Loss: %f, min valloss: %f] ETA: %s" %(epoch, opt.n_epochs, i, len(dataloader), train_score, val_score, min_val, time_left))
                print("******************************************************************\n")
                running_loss = 0.0
                lastbatch_count = 0
                        

            # Print log
            sys.stdout.write(
                    "\n[Epoch %d/%d] [Batch %d/%d] [train MSE Loss: %f, min valloss: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_pixel_LR.item(),
                    min_val,
                    time_left,
                )
            )
        
        # epoch done, save model
        # Determine approximate time left
        batches_done = (epoch+1) * len(dataloader)
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if (opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0) or epoch < 5 or epoch == opt.n_epochs-1:
            print("\n******************************************************************")
            print("Save model checkpoint stage: (Save Every %d epochs)"%(opt.checkpoint_interval))            
            print("calculating val/train MSE loss and saving validation images...")
            val_score = sample_images(epoch, batches_done)
            train_score = running_loss/lastbatch_count            
            print("write on tensorboard...")
            writer.add_scalar('Train/train_score',train_score,batches_done)
            writer.add_scalar('Val/val_score',val_score,batches_done)
            writer.flush()
            print("%d Epoch train L2 loss: %f" %(epoch, train_score))
            print("%d Epoch val L2 loss: %f" %(epoch, val_score))
            print("%d Epoch min valloss: %.5f at %d epoch" %(epoch, min_val, min_epoch))
            
            # Save model checkpoints
            print("save generator...")
            torch.save(generator.state_dict(), "saved_models/%s/%s/%s/generator_%d.pth" % ('dataset_'+opt.dataset_name, opt.model, current_time, epoch)) # .module
            running_loss = 0.0
            lastbatch_count = 0
            print("*************************************************\n\n")  
            

        # Print log
        sys.stdout.write(
                "\n[Epoch %d/%d] [Batches DONE!!] [current trainloss: %f]\nmin valloss %.5f at %d epoch\ntimeleft: %s"
            % (
                epoch,
                opt.n_epochs,                    
                loss_pixel_LR.item(),
                min_val,
                min_epoch,
                time_left,
            )
        )

