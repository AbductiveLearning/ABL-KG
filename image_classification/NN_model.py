import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.resnet import resnet50
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from PIL import Image
import time
import pickle as pkl
import os
from info import get_scene_file, get_fig_object_list, get_cand_obj_idxs
from metrics import compute_f1


import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet50(nn.Module):
    def __init__(self, num_class, loss_criterion, pretrained_path=None, freeze_feature=True, num_workers=16):
        super(ResNet50, self).__init__()
        # encoder
        self.f = []
        for name, module in resnet50(pretrained=True).named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)
        # classifier
        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        self.fc = nn.Linear(2048, num_class, bias=True)
        self.loss_criterion = loss_criterion
        self.optimizer = None
        self.set_freeze_feature(freeze_feature)
        self.num_workers = num_workers
    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out, F.normalize(feature, dim=-1)

    def set_freeze_feature(self, freeze, lr=1e-4):
        for param in self.f.parameters():
            param.requires_grad = not freeze
        parameters = self.fc.parameters() if freeze else self.parameters()
        self.optimizer = optim.Adam(parameters, lr=lr, weight_decay=1e-5)
        # self.scheduler = CosineAnnealingLR(self.optimizer,T_max=50)
    
    # train or test for several epochs
    def train_val(self, epochs, is_train, data_loader):
        if is_train:
            self.train() # for BN
            scaler = GradScaler() # AMP
        else:
            self.eval()
            epochs = 1

        for epoch in range(1, epochs+1):
            total_loss, total_num,data_bar = 0.0, 0, tqdm(data_loader)
            model_results, targets = [], []
            with (torch.enable_grad() if is_train else torch.no_grad()):
                for data, target in data_bar:
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    with autocast(): # AMP
                        out, feature = self(data)
                        loss = self.loss_criterion(out, target)

                    if is_train:
                        self.optimizer.zero_grad()
                        # loss.backward()
                        # self.optimizer.step()
                        scaler.scale(loss).backward() # AMP
                        scaler.step(self.optimizer) # AMP
                        scaler.update() # AMP
                    
                    total_num += data.size(0)
                    total_loss += loss.item() * data.size(0)
                    model_results.extend(out.detach().cpu().numpy())
                    targets.extend(target.detach().cpu().numpy().astype(int))
                    micro_f1_score, macro_f1_score, f1_score_list, precision_list, recall_list = compute_f1(targets, model_results)
                    data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} Micro-F1: {:.2f} Macro-F1: {:.2f}'
                                            .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num, micro_f1_score, macro_f1_score))
                # print("Learning rate:%f" % (self.optimizer.param_groups[0]['lr']))
                # self.scheduler.step() # adjust lr

        return total_loss / total_num, micro_f1_score, macro_f1_score, f1_score_list, precision_list, recall_list

    def predict(self, data_loader):
        self.eval()
        results_out = []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.cuda(non_blocking=True)
                out, feature = self(data)
                results_out.append(out)
        return torch.sigmoid(torch.cat(results_out, axis=0)).cpu().numpy()



class TorchDataset(Dataset):
    def __init__(self, root_dataset, image_name_list, label, training=True, scale_size=640, crop_size=576):
        self.root_dataset = root_dataset
        self.image_name_list = image_name_list
        self.label = torch.FloatTensor(label)
        self.training = training
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_data_transform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                                transforms.RandomChoice([transforms.RandomCrop(640),
                                                transforms.RandomCrop(576),
                                                transforms.RandomCrop(512),
                                                transforms.RandomCrop(384)
                                                # transforms.RandomCrop(320)
                                                ]),
                                                transforms.Resize((crop_size, crop_size)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize])
        self.test_data_transform = transforms.Compose([
                                                # transforms.Resize((scale_size, scale_size)),
                                                # transforms.CenterCrop(crop_size),
                                                transforms.Resize((crop_size, crop_size)),
                                                transforms.ToTensor(),
                                                normalize])

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dataset, self.image_name_list[idx]+".jpg")
        img = Image.open(image_path).convert('RGB')
        if self.training:
            img = self.train_data_transform(img)
        else:
            img = self.test_data_transform(img)
        label = self.label[idx]
        return img, label

if __name__ == '__main__':
    n_label = 2000 # 20210
    n_training = 20210
    crop_size = 320
    batch_size, epochs = 128, 50
    index_file = 'index_ade20k.pkl'

    with open(index_file, 'rb') as f:
        index_ade20k = pkl.load(f)

    # Scene info from ADE2016, and object info from 2021
    file_name_list, scene_list = get_scene_file("sceneCategoriesADE2016.txt", shuffle_train=True, n_training=n_training)
    fig_object_list = get_fig_object_list(index_ade20k, file_name_list)
    # Get objects that appear more than threshold in the dataset
    obj_cnt_thres = 2900
    cand_obj_idxs = get_cand_obj_idxs(fig_object_list, obj_cnt_thres=obj_cnt_thres)
    fig_object_list = fig_object_list[:,cand_obj_idxs]

    num_class = fig_object_list.shape[1]
    print("num_class:", num_class)

    train_data = TorchDataset(root_dataset="images/training", image_name_list=file_name_list[:n_label], training=True, label=fig_object_list[:n_label], scale_size=640, crop_size=crop_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_data = TorchDataset(root_dataset="images/validation", image_name_list=file_name_list[n_training:], training=False, label=fig_object_list[n_training:], scale_size=crop_size, crop_size=crop_size)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    model = ResNet50(num_class=num_class, loss_criterion=nn.MultiLabelSoftMarginLoss(), freeze_feature=False).cuda() #BCEWithLogitsLoss()
    # print(resnet50())

    
    # train_data = TorchDataset(root_dataset="images/training", image_name_list=file_name_list, label=fig_object_list)
    # print(train_data[0][0][None, ...].cuda())
    # y = model(train_data[0][0][None, ...].cuda())
    # print(y)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_micro_f1 = model.train_val(1, True, data_loader=train_loader)
        test_loss, test_micro_f1 = model.train_val(1, False, data_loader=test_loader)
        print()
        # if test_acc_1 > best_acc:
        #     best_acc = test_acc_1
        #     torch.save(model.state_dict(), 'results/linear_model.pth')
        #y_pred, feature = model.predict(X=X_test)