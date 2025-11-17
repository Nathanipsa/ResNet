import torch
import numpy as np
import cv2

MEAN = np.array([0.4000697731636383, 0.44971866014284007, 0.47799389505924583])
STD = np.array([0.2785395075911979, 0.26381946395705097, 0.2719872044484063])

def read_gt(gt_txt, num_img):
    cls = np.zeros(num_img)
    f = open(gt_txt, 'r')
    lines = f.readlines()

    for it in range(len(lines)):
        cls[it] = int(lines[it][:-1]) - 1
    f.close()

    return cls


def random_horizontal_flip(img, p=0.5):
    if np.random.rand() < p:
        img = np.fliplr(img)
    return img


def random_crop(img, padding=10):
    img_padded = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    h, w = img_padded.shape[:2]
    top = np.random.randint(0, h - 128 + 1)
    left = np.random.randint(0, w - 128 + 1)
    img_cropped = img_padded[top:top + 128, left:left + 128, :]
    return img_cropped


def random_rotation(img, max_angle=15, p=0.5):
    if np.random.rand() < p:
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img


def color_jitter(img, brightness=0.2, contrast=0.1, saturation=0.3):
    img = img.copy()

    if np.random.rand() < 0.8:
        alpha = 1.0 + np.random.uniform(-brightness, brightness)
        img = np.clip(img * alpha, 0, 255)

    if np.random.rand() < 0.8:
        alpha = 1.0 + np.random.uniform(-contrast, contrast)
        mean = img.mean()
        img = np.clip((img - mean) * alpha + mean, 0, 255)

    if np.random.rand() < 0.8:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        alpha = 1.0 + np.random.uniform(-saturation, saturation)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * alpha, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return img

def cutout(img, n_holes=1, length=32, normalized=False):
    h, w = img.shape[:2]
    img_cutout = np.copy(img)

    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        if normalized:
            img_cutout[y1:y2, x1:x2, :] = 0.0
        else:
            img_cutout[y1:y2, x1:x2, :] = 0

    return img_cutout


def Mini_batch_training_zip(z_file, z_file_list, train_cls, batch_size, augmentation=True):
    batch_img = np.zeros((batch_size, 128, 128, 3))
    batch_cls = np.zeros(batch_size)

    rand_num = np.random.randint(0, len(z_file_list), size=batch_size)

    for it in range(batch_size):
        temp = rand_num[it]
        img_temp = z_file.read(z_file_list[temp])
        img = cv2.imdecode(np.frombuffer(img_temp, np.uint8), 1)
        img = img.astype(np.float32)

        if augmentation:
            img = color_jitter(img, brightness=0.2, contrast=0.1, saturation=0.3)
            img = random_horizontal_flip(img, p=0.5)
            img = random_rotation(img, max_angle=15, p=0.5)
            img = random_crop(img, padding=10)

        img = img / 255.0

        if MEAN is not None and STD is not None:
            img = (img - MEAN) / STD

        if augmentation:
            img = cutout(img, n_holes=1, length=32, normalized=True)

        batch_img[it, :, :, :] = img
        batch_cls[it] = train_cls[temp]

    return batch_img, batch_cls


class CBAM(torch.nn.Module):
    def __init__(self, c_out, reduction=16):
        super(CBAM, self).__init__()
        self.c_out = c_out

        self.ch_attn = torch.nn.Sequential(
            torch.nn.Linear(c_out, c_out // reduction),
            torch.nn.ReLU(),
            torch.nn.Linear(c_out // reduction, c_out),
        )

        self.sp_attn = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=7, padding=3),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x_ch_avg = torch.mean(x, dim=(2, 3))
        x_ch_max, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)

        x_ch_avg = self.ch_attn(x_ch_avg)
        x_ch_max = self.ch_attn(x_ch_max)
        x_ch_at = torch.sigmoid(x_ch_avg + x_ch_max)
        x_ch_at = torch.reshape(x_ch_at, [-1, self.c_out, 1, 1])

        x_cbam = x * x_ch_at

        x_sp_avg = torch.mean(x_cbam, dim=1, keepdim=True)
        x_sp_max, _ = torch.max(x_cbam, dim=1, keepdim=True)
        x_sp = torch.cat([x_sp_avg, x_sp_max], dim=1)
        x_sp_at = self.sp_attn(x_sp)

        x_out = x_cbam * x_sp_at + x

        return x_out


class ResBlock(torch.nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super(ResBlock, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(c_in)
        self.conv1 = torch.nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(c_out)
        self.conv2 = torch.nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.ReLU = torch.nn.ReLU()

        if stride != 1 or c_in != c_out:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(c_out)
            )
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.bn1(x)
        out = self.ReLU(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.ReLU(out)
        out = self.conv2(out)
        x = out + identity
        return x


class ResNet(torch.nn.Module):
    def __init__(self, outputsize=200):
        super(ResNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)

        self.block1_1 = ResBlock(64, 64)
        self.block1_2 = ResBlock(64, 64)
        self.block1_3 = ResBlock(64, 64)

        self.block2_1 = ResBlock(64, 128, stride=2)
        self.block2_2 = ResBlock(128, 128)
        self.block2_3 = ResBlock(128, 128)

        self.block3_1 = ResBlock(128, 256, stride=2)
        self.block3_2 = ResBlock(256, 256)
        self.block3_3 = ResBlock(256, 256)
        self.cbam3 = CBAM(256, reduction=16)

        self.block4_1 = ResBlock(256, 512, stride=2)
        self.block4_2 = ResBlock(512, 512)
        self.block4_3 = ResBlock(512, 512)
        self.cbam4 = CBAM(512, reduction=16)

        self.bn_final = torch.nn.BatchNorm2d(512)
        self.ReLU = torch.nn.ReLU()
        self.AvgPool = torch.nn.AvgPool2d(kernel_size=16)

        self.Dropout = torch.nn.Dropout(p=0.5)
        
        self.fc1 = torch.nn.Linear(512, outputsize)


    def forward(self, x):
        x = self.conv1(x)

        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.cbam3(x)

        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block4_3(x)
        x = self.cbam4(x)

        x = self.bn_final(x)
        x = self.ReLU(x)
        x = self.AvgPool(x)
        x = torch.reshape(x, [-1, 512])
        x = self.fc1(x)

        return x