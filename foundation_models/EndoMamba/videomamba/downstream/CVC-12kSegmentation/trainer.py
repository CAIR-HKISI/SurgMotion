import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from dataset import *
import medpy


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = medpy.metric.binary.dc(pred, gt)
        hd95 = medpy.metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1., 0.
    else:
        return 0., 0.


def eval_dice(pred_y, gt_y, classes=5):
    pred_y = torch.argmax(torch.softmax(pred_y, dim=1), dim=1)
    all_dice = []
    all_hd95 = []

    pred_y = pred_y.cpu().detach().numpy()
    gt_y = gt_y.cpu().detach().numpy()

    for cls in range(1, classes):
        dice, hd95 = calculate_metric_percase(pred_y == cls, gt_y == cls)

        all_dice.append(dice)
        all_hd95.append(hd95)

    all_dice = torch.tensor(all_dice).cuda()
    all_hd95 = torch.tensor(all_hd95).cuda()

    return torch.mean(all_dice), torch.mean(all_hd95)


def eval(model, val_loader, device, classes):
    all_dice = []
    all_hd95 = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            img, label = batch['image'].to(device).squeeze(), batch['label'].to(device).squeeze()
            img = img.unsqueeze(0).permute(0, 2, 1, 3, 4)
            output = model(img)

            dice, hd95 = eval_dice(output, label, classes=classes)
            all_dice.append(dice.item())
            all_hd95.append(hd95.item())

    return np.mean(np.array(all_dice)), np.mean(np.array(all_hd95))


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator, dynamic_padding_collate_fn
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    train_transform = transforms.Compose([
        Resize((args.img_size + 32, args.img_size + 32)),
        RandomCrop((args.img_size, args.img_size)),
        RandomFlip(),
        RandomRotation(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        Resize((args.img_size, args.img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    db_train = SegCTDataset(dataroot=args.root_path, mode='train', transforms=train_transform)
    db_test = SegCTDataset(dataroot=args.root_path, mode='test', transforms=test_transform)

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of test set is: {}".format(len(db_test)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn,collate_fn=dynamic_padding_collate_fn)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    if args.test:
        model.eval()
        model.load_state_dict(torch.load(args.pretrained_model_weights, map_location='cpu'))
        test_dice, test_hd95 = eval(model, testloader, 'cuda', classes=2)
        print('Test Dice: %.1f, HD95: %.1f' % (test_dice * 100., test_hd95))
        exit(0)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-2)

    if args.wandb:
        wandb.init(project='synapse-segmentation', config=args, dir="/mnt/tqy/wandb/")
        wandb.watch(model, log='all')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_dice = 0.
    best_hd95 = 0.
    iterator = range(max_epoch)
    for epoch_num in iterator:
        model.train()

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            image_batch = image_batch.permute(0, 2, 1, 3, 4)
            outputs = model(image_batch)
            
            # print("Shape:", label_batch.shape)
            # print(outputs.shape)
            
            label_batch = label_batch.view(-1, outputs.shape[2], outputs.shape[3])

            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            if args.wandb:
                wandb.log({'lr': lr_, 'total_loss': loss, 'loss_ce': loss_ce})

            if iter_num % 20 == 0 and args.wandb:
                index = 0
                image = image_batch[0, :, index].permute(1, 2, 0).cpu().numpy()
                image = (image - image.min()) / (image.max() - image.min())
                wandb.log({'train/Image': [wandb.Image(image)]})
                outputs = torch.argmax(torch.softmax(outputs[index], dim=0), dim=0, keepdim=False).unsqueeze(0)
                outputs = outputs.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
                wandb.log({'train/Prediction': [wandb.Image(outputs)]})
                labs = label_batch[index]
                labs = labs.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
                wandb.log({'train/GroundTruth': [wandb.Image(labs)]})

        test_dice, test_hd95 = eval(model, testloader, 'cuda', classes=2)
        if test_dice > best_dice:
            best_dice = test_dice
            best_hd95 = test_hd95
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))
            logging.info("save best model to best_model.pth")

        print('Epoch [%3d/%3d], Loss: %.4f, Dice: %.1f, HD95: %.1f, Best Dice: %.1f, Best HD95: %.1f' %
              (epoch_num + 1, max_epoch, loss.item(), test_dice * 100., test_hd95, best_dice * 100., best_hd95))

        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'last_model.pth'))
            logging.info("save last model to last_model.pth")
            break
    
    if args.wandb:
        wandb.finish()
    return "Training Finished!"
