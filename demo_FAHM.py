import os
import time
import math
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from scipy.io import loadmat
import torch.utils.data as Data
from matplotlib import colors
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from fahm import proposed
logging.basicConfig(filename='logs/FAHM.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')    
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston', 'Trento', 'LongKou'], default='Indian', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=202407, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=20, help='number of evaluation')
parser.add_argument('--patches', type=int, default=9, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=200, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=1.1e-3, help='weight_decay')
parser.add_argument('--load', type=str, default='exp/FAHM/Indian/bs64_lr0.0005_epoch200_testF20.pth', help='weight_decay')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

def visualize_fourier_components(input_tensor):
    c = input_tensor.shape[2] // 2
    input_tensor = input_tensor[ :, :,c]
    input_tensor = input_tensor.astype(np.float128)

    f_transform = np.fft.fft2(input_tensor)
    f_transform_shifted = np.fft.fftshift(f_transform)

    rows, cols = input_tensor.shape
    crow, ccol = rows // 2 , cols // 2
    
    mask_low = np.zeros((rows, cols), np.uint8)
    mask_high = np.ones((rows, cols), np.uint8)
    r = min(rows, cols) // 14
    mask_low[crow-r:crow+r, ccol-r:ccol+r] = 1
    mask_high[crow-r:crow+r, ccol-r:ccol+r] = 0

    low_freq = f_transform_shifted * mask_low
    high_freq = f_transform_shifted * mask_high

    img_low_freq = np.fft.ifft2(np.fft.ifftshift(low_freq)).real
    img_high_freq = np.fft.ifft2(np.fft.ifftshift(high_freq)).real
    
    img_high_freq = np.clip((img_high_freq - img_high_freq.min()) / (img_high_freq.max() - img_high_freq.min()), 0, 1)
    img_high_freq = np.power(img_high_freq, 0.9) 

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.title('All frequency (Frequency Domain)')
    plt.axis('off')
    plt.imshow((np.abs(f_transform_shifted) + 1), cmap='viridis', vmin=0, vmax=70000)
    plt.savefig(f"image_vision/All frequency (Frequency Domain).png")
    plt.close()

    # 高频频域图像
    plt.subplot(2, 3, 2)
    plt.title('High frequency (Frequency Domain)')
    plt.axis('off')
    plt.imshow((np.abs(high_freq) + 1), cmap='viridis', vmin=0, vmax=70000)
    plt.savefig(f"image_vision/High frequency (Frequency Domain).png")
    plt.close()
    
    # 低频频域图像
    plt.subplot(2, 3, 3)
    plt.title('Low frequency (Frequency Domain)')
    plt.axis('off')
    plt.imshow((np.abs(low_freq) + 1), cmap='viridis', vmin=0, vmax=70000)
    plt.savefig(f"image_vision/Low frequency (Frequency Domain).png")
    plt.close()
    
    # 原始空间域图像
    plt.subplot(2, 3, 4)
    plt.title('All frequency (Spatial Domain)')
    plt.axis('off')
    plt.imshow(input_tensor, cmap='viridis')
    plt.savefig(f"image_vision/All frequency (Spatial Domain).png")
    plt.close()

    # 高频空间域图像
    plt.subplot(2, 3, 5)
    plt.title('High frequency (Spatial Domain)')
    plt.axis('off')
    plt.imshow(img_high_freq, cmap='viridis')
    plt.savefig(f"image_vision/High frequency (Spatial Domain).png")
    plt.close()
    
    # 低频空间域图像
    plt.subplot(2, 3, 6)
    plt.title('Low frequency (Spatial Domain)')
    plt.axis('off')
    plt.imshow(img_low_freq, cmap='viridis')
    plt.savefig(f"image_vision/Low frequency (Spatial Domain).png")
    plt.close()

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
    plt.savefig(f"image_vision/frequency.png")

def warm_up_learning_rate_adjust1(init_lr, epoch, warm_epoch, max_epoch, optimizer):
    for param_group in optimizer.param_groups:
        if epoch < warm_epoch:
            param_group['lr'] = init_lr*(epoch+1)/(warm_epoch+1)
        else:
            param_group['lr'] = init_lr*(math.cos(math.pi*(epoch-warm_epoch)/max_epoch)+1)/2

def warm_up_learning_rate_adjust2(init_lr, epoch, warm_epoch, max_epoch, optimizer):
    for param_group in optimizer.param_groups:
        if epoch < warm_epoch:
            param_group['lr'] = init_lr*(1-math.cos(math.pi/2*(epoch+1)/(warm_epoch)))
        else:
            param_group['lr'] = init_lr*(math.cos(math.pi*(epoch-warm_epoch)/max_epoch)+1)/2

def warm_up_learning_rate_adjust3(init_lr, epoch, warm_epoch, max_epoch, optimizer, power=0.9):
    for param_group in optimizer.param_groups:
        if epoch < warm_epoch:
            param_group['lr'] = init_lr*(epoch+1)/(warm_epoch+1)
        else:
            param_group['lr'] = init_lr*((1 - (epoch-warm_epoch)/(max_epoch-warm_epoch))**(power))

def warm_up_learning_rate_adjust4(init_lr, epoch, warm_epoch, max_epoch, optimizer, power=0.9):
    for param_group in optimizer.param_groups:
        if epoch < warm_epoch:
            param_group['lr'] = init_lr * epoch / (warm_epoch+1e-8)
        else:
            param_group['lr'] = init_lr*((1 - (epoch-warm_epoch)/(max_epoch-warm_epoch))**(power))

#-------------------------------------------------------------------------------
# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band, x_true_band
#-------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()       

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    features_list = []
    labels_list = [] 
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)

        # batch_features,_ = model.forward_features(batch_data)
        # batch_features = batch_features.mean(dim=1) 
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    #     # Store features and labels for t-SNE
    #     features_list.append(batch_features.detach().cpu().numpy())
    #     labels_list.append(batch_target.detach().cpu().numpy())

    # # Stack the features and labels into arrays
    # features = np.vstack(features_list)
    # labels = np.hstack(labels_list)

    return tar, pre#, features, labels

def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre
#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def plot_tsne(labels, pre, n_components=2):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = StandardScaler()
    pre_normalized = scaler.fit_transform(pre)
    color_custom_matrix = np.array([
        [255, 0, 0],       # 红色
        [0, 255, 0],       # 绿色
        [0, 0, 255],       # 蓝色
        [255, 255, 0],     # 黄色
        [255, 0, 255],     # 洋红
        [0, 255, 255],     # 青色
        [128, 0, 128],     # 紫色
        [255, 165, 0],     # 橙色
        [75, 0, 130],      # 靛蓝
        [0, 128, 0],       # 深绿色
        [128, 128, 0],     # 橄榄绿
        [255, 192, 203],   # 粉红色
        [139, 69, 19],     # 棕色
        [105, 105, 105],   # 暗灰色
        [255, 223, 186],   # 浅桃色
        [173, 216, 230],   # 浅蓝色
        ])
    color_custom_matrix = color_custom_matrix/255
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
    pre_tsne = tsne.fit_transform(pre_normalized)
    plt.figure(figsize=(8, 6))

    plt.scatter(pre_tsne[:, 0], pre_tsne[:, 1], c=[color_custom_matrix[label] for label in labels], marker='o', alpha=0.7)
    plt.xticks([])
    plt.yticks([])
    
    os.makedirs('image_vision',exist_ok=True)
    plt.savefig(f"image_vision/{args.dataset}/FAHM_tsne.png")

#-------------------------------------------------------------------------------
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data
if args.dataset == 'Indian':
    data = loadmat('./data/IndianPine.mat')
elif args.dataset == 'Pavia':
    data = loadmat('./data/Pavia.mat')
elif args.dataset == 'Trento':
    data = loadmat('./data/Trento.mat')
elif args.dataset == 'LongKou':
    data = loadmat('./data/LongKou.mat')
else:
    raise ValueError("Unkknow dataset")
color_mat = loadmat('./data/AVIRIS_colormap.mat')
TR = data['TR']
TE = data['TE']
input = data['input'] #(145,145,200)
label = TR + TE
num_classes = int(np.max(TR))
color_mat_list = list(color_mat)
color_matrix = color_mat[color_mat_list[3]] #(17,3)
# normalize data by band norm
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
# data size
height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))
# visualize_fourier_components(input)

#-------------------------------------------------------------------------------
# obtain train and test data
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches)
y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
#-------------------------------------------------------------------------------
# load data
x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
Label_train=Data.TensorDataset(x_train,y_train)
x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
Label_test=Data.TensorDataset(x_test,y_test)
x_true=torch.from_numpy(x_true_band.transpose(0,2,1)).type(torch.FloatTensor)
y_true=torch.from_numpy(y_true).type(torch.LongTensor)
Label_true=Data.TensorDataset(x_true,y_true)

label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)

#-------------------------------------------------------------------------------
# create model
model = proposed(args.dataset, args.patches)

model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//20, gamma=args.gamma)
os.makedirs(f'exp/FAHM/{args.dataset}',exist_ok=True)
weight_root = f'exp/FAHM/{args.dataset}'

#------------------------------------------------------------------------------
if args.flag_test == 'test':
    model.load_state_dict(torch.load(args.load),strict=True)
    model.eval()
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
    print(" val_OA: {:.4f} val_AA: {:.4f} val_Kappa: {:.4f} ".format(OA2, AA_mean2, Kappa2))
    print(AA2)
    color_costom_matrix = np.ones([16, 3])
    if args.dataset == 'Indian':
        dpi_large = 600
        color_costom_matrix = np.array([[79, 170, 72],
                                        [136, 186, 67],
                                        [62, 131, 91],
                                        [54, 132, 68],
                                        [144, 81, 54],
                                        [102, 188, 199],
                                        [255, 255, 255],
                                        [198, 175, 201],
                                        [218, 48, 44],
                                        [120, 34, 35],
                                        [86, 87, 89],
                                        [223, 220, 83],
                                        [217, 142, 52],
                                        [83, 47, 125],
                                        [227, 119, 90],
                                        [157, 86, 151], ])
    elif args.dataset == 'Pavia':
        dpi_large = 600
        color_costom_matrix = np.array([[199, 200, 202],
                                        [109, 177, 70],
                                        [102, 188, 199],
                                        [55, 123, 66],
                                        [73, 73, 179],  # metal sheets
                                        [149, 82, 49],
                                        [116, 45, 121],
                                        [200, 88, 76],  # brick
                                        [223, 220, 83], ])
    elif args.dataset == 'Trento':
        dpi_large = 600
        color_costom_matrix = np.array([[223, 220, 83],
                                        [218, 48, 44],
                                        [199, 200, 202],
                                        [149, 82, 49],
                                        [109, 177, 70], 
                                        [102, 188, 199], ])
    elif args.dataset == 'Houston':
        dpi_large = 1000
        color_costom_matrix = np.array([[79, 170, 72],
                                        [136, 186, 67],
                                        [62, 131, 91],
                                        [54, 132, 68],
                                        [144, 81, 54],
                                        [102, 188, 199],
                                        [255, 255, 255],
                                        [218, 48, 44],  # commercial
                                        [120, 133, 131],  # road
                                        [120, 34, 35],
                                        [50, 101, 67],
                                        [223, 220, 83],
                                        [198, 175, 201],
                                        [83, 47, 125],
                                        [227, 119, 90], ])
    elif args.dataset == 'LongKou':
        dpi_large = 1000
        color_costom_matrix = np.array([[79, 170, 72],
                                        [136, 186, 67],
                                        [198, 175, 201],
                                        [227, 119, 90],
                                        [144, 81, 54],
                                        [102, 188, 199],
                                        [223, 220, 83],
                                        [218, 48, 44],  # commercial
                                        [120, 133, 131], ])
    else:
        raise ValueError("Unkknow dataset")
    color_costom_matrix = color_costom_matrix/255
    time_start = time.time()
    pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    time_end = time.time()
    print(f"推理时间为{time_end-time_start}")

    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i] + 1
    plt.figure()
    plt.imshow(prediction_matrix, colors.ListedColormap(color_costom_matrix))
    plt.xticks([])
    plt.yticks([])
    os.makedirs('image_vision',exist_ok=True)
    plt.savefig(f"image_vision/{args.dataset}/FAHM.png", dpi=dpi_large)

    # plot_tsne(all_features_label, features, n_components=2)

    # # GT
    # ground_truth_matrix = np.zeros((height, width), dtype=float)
    # for i in range(total_pos_true.shape[0]):
    #     ground_truth_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = y_true[i]

    # plt.figure()
    # plt.imshow(ground_truth_matrix, colors.ListedColormap(color_costom_matrix))
    # plt.xticks([])
    # plt.yticks([])
    # os.makedirs('image_vision',exist_ok=True)
    # plt.savefig(f"image_vision/{args.dataset}/GT.png", dpi=dpi_large)
    plt.show()
    exit()

elif args.flag_test == 'train':
    print("start training")
    tic = time.time()
    best_score = {'epoch': 0, 
                  'score': [0,0,0]}
    for epoch in tqdm(range(args.epoches)): 
        warm_up_learning_rate_adjust2(args.learning_rate, epoch, 10, args.epoches, optimizer)
        # train model
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t) 
        tqdm.write("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}".format(epoch+1, train_obj, train_acc))

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):         
            model.eval()
            save = False
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)

            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            if (OA2+AA_mean2+Kappa2) >= sum(best_score['score']):
                save = True
                best_score['epoch'] =  epoch+1
                best_score['score'] =  [OA2, AA_mean2, Kappa2]
                best_score['AA'] = AA2
                print("Epoch: {:03d} val_OA: {:.4f} val_AA: {:.4f} val_Kappa: {:.4f} ".format(epoch+1, OA2, AA_mean2, Kappa2))
            tqdm.write("Epoch: {:03d} val_OA: {:.4f} val_AA: {:.4f} val_Kappa: {:.4f} ".format(epoch+1, OA2, AA_mean2, Kappa2))
            if save:
                torch.save(model.state_dict(),os.path.join(weight_root,f'bs{args.batch_size}_lr{args.learning_rate}_epoch{args.epoches}_testF{args.test_freq}.pth'))
                
    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")

logger.info("##############start##########")
logger.info("Final result:")
logger.info("BestEpoch: {:03d} OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(best_score['epoch'],best_score['score'][0], best_score['score'][1], best_score['score'][2]))
logger.info(f"All_Accuracy:\n {best_score['AA']}")
logger.info("**************************************************")
logger.info("Parameter:")
logger.info("train-time: {:.2f}".format(toc-tic))


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))
        logger.info("{0}: {1}".format(k,v))
print_args(vars(args))
print("BestEpoch: {:03d} OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(best_score['epoch'],best_score['score'][0], best_score['score'][1], best_score['score'][2]))
print(f"All_Accuracy:\n {best_score['AA']}")
logger.info("##############finish##########")
logger.info("\n\n\n")

