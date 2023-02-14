# -*- coding: utf-8 -*-
"""
**Drive Mounting**
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/gdrive')
# %cd /gdrive


from google.colab import drive
drive.mount('/content/gdrive') 


# %cd /content/gdrive/My Drive

"""**Libraries**"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import torch.nn as nn
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader

"""**Code**"""

def IOU(bpred, blabels, bform="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    # top x & top y coz origin in top left corner (in comp vision)
    # bpred shape is in form of (n, 4) n is number of boxes
    # blabels shape is (n, 4)
    if bform == "midpoint":
      # here we are converting midpoints into corner points first

        # for x1 val we divide the width by 2 cause this val (bpred[..., 0:1] - bpred[..., 2:3] --> gives us midpoints)
        b1x1 = bpred[..., 0:1] - bpred[..., 2:3] / 2
        # in case of y we calc it from height given by bpred[..., 1:2] - bpred[..., 3:4]
        b1y1 = bpred[..., 1:2] - bpred[..., 3:4] / 2
        b1x2 = bpred[..., 0:1] + bpred[..., 2:3] / 2
        b1y2 = bpred[..., 1:2] + bpred[..., 3:4] / 2
        b2x1 = blabels[..., 0:1] - blabels[..., 2:3] / 2
        b2y1 = blabels[..., 1:2] - blabels[..., 3:4] / 2
        b2x2 = blabels[..., 0:1] + blabels[..., 2:3] / 2
        b2y2 = blabels[..., 1:2] + blabels[..., 3:4] / 2

    elif bform == "corners":
      # we are slicing the tensor to maintain shape in form (N, 1) with out it it would just be (n)
      # box1 = [x1, y1, x2, y2]
      # box2 = [x1, y1, x2, y2]
        b1x1 = bpred[..., 0:1] # get index [0:1] through slicing it will return only 0 index element which is x1
        b1y1 = bpred[..., 1:2] # get index 1 element which is y1
        b1x2 = bpred[..., 2:3] # get index 2 element which is x2
        b1y2 = bpred[..., 3:4] # get index 3 element which is y2
        b2x1 = blabels[..., 0:1] # so on...
        b2y1 = blabels[..., 1:2]
        b2x2 = blabels[..., 2:3]
        b2y2 = blabels[..., 3:4]

     # For calculating corner points we use formula: 
     # x1 = max(box1[0], box2[0]
     # y1 = max(box1[1], box2[1]
     # x2 = min(box1[2], box2[2])
     # y2 = min(box1[3], box2[3])
    x1 = torch.max(b1x1, b2x1)
    y1 = torch.max(b1y1, b2y1)
    x2 = torch.min(b1x2, b2x2)
    y2 = torch.min(b1y2, b2y2)

    # .clamp(0) for no intersection (just turn (x2 - x1) or (y2 - y1) to zero)
    intersect = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # for union
    b1 = abs((b1x2 - b1x1) * (b1y2 - b1y1)) 
    b2 = abs((b2x2 - b2x1) * (b2y2 - b2y1))

    res = intersect / (b1 + b2 - intersect + 1e-6) # value between 0-1

    return res


def NMS(bbx, inter_threshold, threshold, bform="corners"):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    #assert type(bboxes) == list

    bbx = [b for b in bbx if b[1] > threshold]
    # highest prob bounding boxes in front or begining
    bbx = sorted(bbx, key=lambda x: x[1], reverse=True)
    bbx2 = []

    while bbx:
        bselect = bbx.pop(0) # select 1st one which also highest

        # bounding boxes < probability threshold
        bbx = [ b for b in bbx
               # start with index 2 cause first two index has class and probility
             if IOU(torch.tensor(bselect[2:]), torch.tensor(b[2:]), bform=bform,)
            < inter_threshold
        ]

        bbx2.append(bselect) # store selected boxes

    return bbx2


def M_Avg_Prec(pred_b, true_b, inter_threshold=0.5, bform="midpoint", cls=1):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    avg_precision = []

    # only for baancing equation
    e = 1e-6

    for c in range(cls):
        detect = []
        true_val = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the current class c

        for det in pred_b:
            if det[1] == c:
                detect.append(det)

        for trb in true_b:
            if trb[1] == c:
                true_val.append(trb)

        # find amount of bboxes for each training
        # lieke for each training, if img0 = 3, img 1 = 5 then we have dictionary
        # total_b = {0:3, 1:5}
        total_b = Counter([x[0] for x in true_val])

        # We go through each key, val in dictionary and convert to
        # total_b = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in total_b.items():
            total_b[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detect.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detect)))
        FP = torch.zeros((len(detect)))
        total_tru_b = len(total_b)
        
        # If none exists for this class then we can safely skip
        if total_tru_b == 0:
            continue

        for idx, d in enumerate(detect):
            # Only take out the org_img that have the same training idx as detection
            org_img = [ b for b in org_img if b[0] == d[0]]

            no_img = len(org_img)
            best = 0

            for ind, x in enumerate(org_img):
              # to skip training index, class prob and send only bounding boxes
                temp_intr = IOU( torch.tensor(d[3:]), torch.tensor(x[3:]),bform=bform,)

                if temp_intr > best:
                    best = temp_intr
                    best_idx = ind

            if best > inter_threshold:
                # only detect ground truth detection once
                if total_b[d[0]][best_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[idx] = 1
                    total_b[d[0]][best_idx] = 1
                else:
                    FP[idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[idx] = 1
        # the way cumsum work ==> [1, 1, 0, 1, 0] --> [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_tru_b + e)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + e))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration; trapz work like trapz(y.val, x.val)
        avg_precision.append(torch.trapz(precisions, recalls))

    res = sum(avg_precision) / len(avg_precision)

    return res


def plot_image(img, boxes):

    im = np.array(img)
    h, w, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle
    for b in boxes:
        b = b[2:]
    
        topx = b[0] - b[2] / 2
        topy = b[1] - b[3] / 2
        rect = patches.Rectangle(
            (topx * w, topy * h),
            b[2] * w,
            b[3] * h,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.show()

def bbx_calc(
    data_load,
    model,
    inter_threshold,
    threshold,
    pform="cells",
    bform="midpoint",
    device="cuda",
):
    bpred_all = []
    btrue_all = []

    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(data_load):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predict = model(x)

        batch_size = x.shape[0]
        btrue = convert_bbx(labels)
        bbx = convert_bbx(predict)

        for idx in range(batch_size):
            nms_boxes = NMS(bbx[idx],inter_threshold=inter_threshold,threshold=threshold,bform=bform,)

            for b in nms_boxes:
                bpred_all.append([train_idx] + b)

            for b in btrue[idx]:
                # many will get converted to 0 pred
                if b[1] > threshold:
                    btrue_all.append([train_idx] + b)

            train_idx += 1

    model.train()
    return bpred_all, btrue_all



def convertion_bbx(predict, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. 
    """

    predict = predict.to("cpu")
    batch_size = predict.shape[0]
    predict = predict.reshape(batch_size, 7, 7, 11)
    bbx1 = predict[..., 2:6]
    bbx2 = predict[..., 7:11]
    scores = torch.cat((predict[..., 1].unsqueeze(0), predict[..., 6].unsqueeze(0)), dim=0)
    bestbbx = scores.argmax(0).unsqueeze(-1)
    bestbbx = bbx1 * (1 - bestbbx) + bestbbx * bbx2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (bestbbx[..., :1] + cell_indices)
    y = 1 / S * (bestbbx[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * bestbbx[..., 2:4]
    converted = torch.cat((x, y, w_y), dim=-1)
    cls_pred = predict[..., :1].argmax(-1).unsqueeze(-1)
    confidence = torch.max(predict[..., 1], predict[..., 6]).unsqueeze(-1)
    converted_preds = torch.cat((cls_pred, confidence, converted), dim=-1)

    return converted_preds


def convert_bbx(out, S=7):
    converted_pred = convertion_bbx(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    bbx_all = []

    for ind in range(out.shape[0]):
        bbx2 = []

        for bbidx in range(S * S):
            bbx2.append([x.item() for x in converted_pred[ind, bbidx, :]])
        bbx_all.append(bbx2)

    return bbx_all

def save_checkpoint(state, filename="dlcheckpoint.pth.tar"):
    print("Saved checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("Load checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])






###----------------------------------yolo architecture------------------------------- ###

""" 
Information about architecture:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

yolo_architecture = [
        ### 1st conv layer ###
  # (kernel_size, filters, stride, padding)
    (7, 64, 2, 3),"maxpool", # indicates maxpool
    ### 2nd conv layer ###
    (3, 192, 1, 1),"maxpool",
    ### 3rd conv layer ###
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),"maxpool",
    ### 4th conv layer ###
    # List:
    # 1st tuple: (1, 256, 1, 0)
    # 2nd tuple: (3, 512, 1, 1)
    # [1st tuple, 2nd tuple1, 4] ---> 4 means this will be repeated 4 times
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),"maxpool",
    ### 5th conv layer ###
    # List:
    # 1st tuple: (1, 512, 1, 0)
    # 2nd tuple: (3, 1024, 1, 1)
    # [1st tuple, 2nd tuple1, 2] ---> 2 means this will be repeated 2 times
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),

      # this will be followed by fully connected layers
]


class cnn(nn.Module):
    def __init__(self, input, out, **kwargs):
        super(cnn, self).__init__()
        self.conv = nn.Conv2d(input, out, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out)
        self.leakyrelu = nn.LeakyReLU(0.1)# 0.1 here is  alpha

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, input=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = yolo_architecture
        self.input = input
        self.darknet = self.create_layers(self.architecture)
        self.fcs = self.create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def create_layers(self, architecture):
        layers = []
        input = self.input

        '''
        over here is where all the layers in the architecture are going to be built in a sequence

        for x in architecture ==> this means take each layer in the architecture

          if type(x) == tuple ==> check if it's type is tuple i,e in this form (7, 64, 2, 3) then pass this x or layer into CNNBlock
                                which means in_channels (the one we already specified), pass x[0] as kernel which means '7' (check above),
                                followed by stride = x[2] i.e 2 ( from (7, 64, 2, 3) ), followed by padding = x[3] which is 3

          if type(x {or layer} ) == str ==> this will match all layers where we have written maxpool in string
                                            and add this --> nn.MaxPool2d(kernel_size=2, stride=2) to the layers

          if type(x) == list ==> [(1, 256, 1, 0), (3, 512, 1, 1), 4]
                                then add 1st tuple to conv1 i.e (1, 256, 1, 0)
                                and add 2nd tuple to conv2 i.e (3, 512, 1, 1)
                                and last list element that is num of repeatation i.e 4 to num_repeats
        '''

        for x in architecture:
            if type(x) == tuple:    # x[1] means out_channels in the cnn block statements
                layers += [ cnn( input, x[1], kernel_size=x[0], stride=x[2], padding=x[3],)]
                input = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]  # tuple
                conv2 = x[1]  # tuple
                repeat = x[2] # integer


                for _ in range(repeat):
                    layers += [cnn( input, conv1[1], kernel_size=conv1[0], stride=conv1[2],padding=conv1[3],)]
                    layers += [ cnn( conv1[1], conv2[1], kernel_size=conv2[0],stride=conv2[2],padding=conv2[3],)]
                    input = conv2[1]

        return nn.Sequential(*layers)

    def create_fcs(self, split_size, num_boxes, cls):
        S, B, C = split_size, num_boxes, cls

        return nn.Sequential( nn.Flatten(),nn.Linear(1024 * S * S, 4096),nn.Dropout(0.5),nn.LeakyReLU(0.1),nn.Linear(4096, S * S * (C + B * 5)),)




###-----------------------------loss function------------------------------- ###


class Loss(nn.Module):

    def __init__(self, S=7, B=2, C=1):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image
        B is number of boxes
        C is number of classes
        """
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predict, target):
        predict = predict.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox

        '''
        ... here indicates (N, S, S, 4)
        where s by s are size of cells
        N is example on which we're currently working on
        25 is just no of bounding boxes we calc like 2:6 or 7:11 it would change accordingly
        '''

        b1inter = IOU(predict[..., 2:6], target[..., 2:6])
        b2inter = IOU(predict[..., 7:11], target[..., 2:6])
        intersections = torch.cat([b1inter.unsqueeze(0), b2inter.unsqueeze(0)], dim=0)

        # select box with highest IoU
        inter_max, bestbox = torch.max(intersections, dim=0)
        if_box = target[..., 1].unsqueeze(3) 

        ### box cord

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        bpredict = if_box * ((bestbox * predict[..., 7:11]+ (1 - bestbox) * predict[..., 2:6])) # this for target dnt confuse with the existing ones

        bx_target = if_box * target[..., 2:6]

        # for absolute we do torch.sign of all
        bpredict[..., 2:4] = torch.sign(bpredict[..., 2:4]) * torch.sqrt(torch.abs(bpredict[..., 2:4] + 1e-6))
        bx_target[..., 2:4] = torch.sqrt(bx_target[..., 2:4])

        # we do flatten so that we could turn (N, S, S, 4) into --> (N*S*S, 4)
        # coz that's how mse expects the input shape to be
        box_loss = self.mse(torch.flatten(bpredict, end_dim=-2),torch.flatten(bx_target, end_dim=-2),)

        ### object loss ###

        pred_box = (bestbox * predict[..., 6:7] + (1 - bestbox) * predict[..., 1:2])

        object_loss = self.mse(torch.flatten(if_box * pred_box),torch.flatten(if_box * target[..., 1:2]),)

        ### no object loss ###

        no_object_loss = self.mse(torch.flatten((1 - if_box) * predict[..., 1:2], start_dim=1),torch.flatten((1 - if_box) * target[..., 1:2], start_dim=1),)

        no_object_loss += self.mse(torch.flatten((1 - if_box) * predict[..., 6:7], start_dim=1),torch.flatten((1 - if_box) * target[..., 1:2], start_dim=1))

        ### class loss ###

        cls_loss = self.mse(torch.flatten(if_box * predict[..., :1], end_dim=-2,),torch.flatten(if_box * target[..., :1], end_dim=-2,),)

        # self.lambda_coord * box_loss ---> first two rows in paper
        # + object_loss  --> third row in paper
        # + self.lambda_noobj * no_object_loss ---> forth row
        # + cls_loss ----> fifth row

        loss = (self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + cls_loss)

        return loss




###-----------------------------loading our dataset--------------------------###

class custom_data(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=1, transform=None,
    ):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.df.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
              # read c, x, y, w, h from text files & stoer in list
                cls_label, x, y, w, h = [ float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()]

                boxes.append([cls_label, x, y, w, h])

        img_path = os.path.join(self.img_dir, self.df.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        lmatrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            cls_label, x, y, w, h = box.tolist()
            cls_label = int(cls_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            
            # Calculat the w h of cell of bbx by cell like w = w*img(w) then acc to cell like w/cell
            cell_w, cell_h = ( w * self.S, h * self.S,)

            # If no object found for cell then consider only one obj
            if lmatrix[i, j, 1] == 0:
                # Set that there exists an object
                lmatrix[i, j, 1] = 1

                # Box coordinates
                bx_cord = torch.tensor([x_cell, y_cell, cell_w, cell_h])

                lmatrix[i, j, 2:6] = bx_cord

                # one hot encoding
                lmatrix[i, j, cls_label] = 1

        return image, lmatrix



###---------------------------training yolo---------------------- ###



seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 64 
WEIGHT_DECAY = 0
EPOCHS = 800
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "dlprjct.pth.tar"
IMG_DIR = "./data/images"
LABEL_DIR = "./data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bbx):
        for t in self.transforms:
            img, bbx = t(img), bbx

        return img, bbx


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    model = Yolov1(split_size=7, num_boxes=2, cls=1).to(DEVICE)
    optimizer = optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = Loss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = custom_data("./data/train.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,) # call function

    test_dataset = custom_data( "./data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True,)

    test_loader = DataLoader( dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False,)

    for epoch in range(EPOCHS):
       for x, y in test_loader:
        x = x.to(DEVICE)
        for idx in range(200):
          bbx = convert_bbx(model(x))
          bbx = NMS(bbx[idx], inter_threshold=0.5, threshold=0.4, bform="midpoint")
          plot_image(x[idx].permute(1,2,0).to("cpu"), bbx)

        import sys
        sys.exit()

        pred_b, target_boxes = bbx_calc(train_loader, model, inter_threshold=0.5, threshold=0.4)

        precision = M_Avg_Prec(pred_b, target_boxes, inter_threshold=0.5, bform="midpoint")


        if precision > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time
            time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()



