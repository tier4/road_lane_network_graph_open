
import numpy as np
import torch
import torch.nn as nn

from models.aspp import ASPP
from models.models_aux import conv3x3


def get_dsla_output_layers(output_tensor, batch=True):
    '''Returns a list of correctly sliced DSLA tensors.
    Args:
        output_tensor: DSLA model output tensor (batch_n, 11, dim, dim)
        batch: Retains the batch dimension if 'True'
    Returns:
        list[0]: SLA (1 layer)
        list[1]: DA_mean (3 layers)
        list[2]: DA_var (3 layers)
        list[3]: DA_w (3 layers)
        list[4]: entry_pnt (1 layer)
        list[5]: exit_pnt (1 layer)
    '''
    if batch:
        outputs_sla = output_tensor[:, 0:1]
        outputs_dir_mean = output_tensor[:, 1:4]
        outputs_dir_var = output_tensor[:, 4:7]
        outputs_dir_weight = output_tensor[:, 7:10]
        outputs_entry = output_tensor[:, 10:11]
        outputs_exit = output_tensor[:, 11:12]
    else:
        outputs_sla = output_tensor[0:1]
        outputs_dir_mean = output_tensor[1:4]
        outputs_dir_var = output_tensor[4:7]
        outputs_dir_weight = output_tensor[7:10]
        outputs_entry = output_tensor[10:11]
        outputs_exit = output_tensor[11:12]

    return (
        outputs_sla, outputs_dir_mean, outputs_dir_var, outputs_dir_weight,
        outputs_entry, outputs_exit
    )


class UnetDSLA(nn.Module):
    '''
    Output tensor shape: (batch_n, layers, dim, dim)
    DSLA output tensor layers:
    --------------------------------------------------
    [0] Soft lane affordance (1 layer)
    --------------------------------------------------
    [1]  Directional mean 1 (3 layers)
    [2]  Directional mean 2
    [3]  Directional mean 3
    --------------------------------------------------
    [4]  Directional var 1 (3 layers)
    [5]  Directional var 2
    [6]  Directional var 2
    --------------------------------------------------
    [7]  Directional weight 1 (3 layers)
    [8]  Directional weight 2
    [9]  Directional weight 3
    --------------------------------------------------
    [10] Entry point affordance (1 layer)
    --------------------------------------------------
    '''
    def __init__(self, base_channels=64, input_channels=2, dropout_prob=0.):
        super(UnetDSLA, self).__init__()

        # Input channels
        self.in_ch = input_channels
        # Network channels
        self.ch = base_channels
        # Output channels
        self.out_SLA_ch = 1
        self.out_DA_mean_ch = 3
        self.out_DA_var_ch = 3
        self.out_DA_weight_ch = 3
        self.out_entry_ch = 1
        self.out_exit_ch = 1

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False)
        
        self.dropout_prob = dropout_prob
        self.dropout = torch.nn.Dropout(p=dropout_prob)

        self.relu = nn.LeakyReLU()

        # Layer 0
        self.layer0 = ASPP(self.in_ch, self.ch)

        # Layer 1 (128, 64)
        self.layer1_1 = conv3x3(self.ch, self.ch)
        self.bn1_1 = nn.BatchNorm2d(self.ch)
        self.layer1_2 = conv3x3(self.ch, self.ch)
        self.bn1_2 = nn.BatchNorm2d(self.ch)
        self.layer1_3 = conv3x3(self.ch, self.ch)
        self.bn1_3 = nn.BatchNorm2d(self.ch)

        # Layer 2 (64, 128)
        self.layer2_1 = conv3x3(self.ch, self.ch*2)
        self.bn2_1 = nn.BatchNorm2d(self.ch*2)
        self.layer2_2 = conv3x3(self.ch*2, self.ch*2)
        self.bn2_2 = nn.BatchNorm2d(self.ch*2)
        self.layer2_3 = conv3x3(self.ch*2, self.ch*2)
        self.bn2_3 = nn.BatchNorm2d(self.ch*2)

        # Layer 3  (32, 256)
        self.layer3_1 = conv3x3(self.ch*2, self.ch*4)
        self.bn3_1 = nn.BatchNorm2d(self.ch*4)
        self.layer3_2 = conv3x3(self.ch*4, self.ch*4)
        self.bn3_2 = nn.BatchNorm2d(self.ch*4)
        self.layer3_3 = conv3x3(self.ch*4, self.ch*4)
        self.bn3_3 = nn.BatchNorm2d(self.ch*4)

        # Layer 4 (16, 512)
        self.layer4_1 = conv3x3(self.ch*4, self.ch*8)
        self.bn4_1 = nn.BatchNorm2d(self.ch*8)
        self.layer4_2 = conv3x3(self.ch*8, self.ch*8)
        self.bn4_2 = nn.BatchNorm2d(self.ch*8)
        self.layer4_3 = conv3x3(self.ch*8, self.ch*8)
        self.bn4_3 = nn.BatchNorm2d(self.ch*8)

        # Layer 5 (8, 1024) 
        self.layer5_1 = conv3x3(self.ch*8, self.ch*16)
        self.bn5_1 = nn.BatchNorm2d(self.ch*16)
        self.layer5_2 = conv3x3(self.ch*16, self.ch*16)
        self.bn5_2 = nn.BatchNorm2d(self.ch*16)
        self.layer5_3 = conv3x3(self.ch*16, self.ch*16)
        self.bn5_3 = nn.BatchNorm2d(self.ch*16)

        # Layer 6 (16, 512)
        self.layer6_1 = conv3x3(self.ch*16 + self.ch*8, self.ch*8)
        self.bn6_1 = nn.BatchNorm2d(self.ch*8)
        self.layer6_2 = conv3x3(self.ch*8, self.ch*8)
        self.bn6_2 = nn.BatchNorm2d(self.ch*8)
        self.layer6_3 = conv3x3(self.ch*8, self.ch*8)
        self.bn6_3 = nn.BatchNorm2d(self.ch*8)

        # Layer 7 (32, 256)
        self.layer7_1 = conv3x3(self.ch*8 + self.ch*4, self.ch*4)
        self.bn7_1 = nn.BatchNorm2d(self.ch*4)
        self.layer7_2 = conv3x3(self.ch*4, self.ch*4)
        self.bn7_2 = nn.BatchNorm2d(self.ch*4)
        self.layer7_3 = conv3x3(self.ch*4, self.ch*4)
        self.bn7_3 = nn.BatchNorm2d(self.ch*4)

        # Layer 8 (64, 125)
        self.layer8_1 = conv3x3(self.ch*4 + self.ch*2, self.ch*2)
        self.bn8_1 = nn.BatchNorm2d(self.ch*2)
        self.layer8_2 = conv3x3(self.ch*2, self.ch*2)
        self.bn8_2 = nn.BatchNorm2d(self.ch*2)
        self.layer8_3 = conv3x3(self.ch*2, self.ch*2)
        self.bn8_3 = nn.BatchNorm2d(self.ch*2)

        # Layer 9 (128, 64)
        self.layer9_1 = conv3x3(self.ch*2 + self.ch, self.ch)
        self.bn9_1 = nn.BatchNorm2d(self.ch)
        self.layer9_2 = conv3x3(self.ch, self.ch)
        self.bn9_2 = nn.BatchNorm2d(self.ch)
        self.layer9_3 = conv3x3(self.ch, self.ch)
        self.bn9_3 = nn.BatchNorm2d(self.ch)

        # Output tail 1 : Soft lane affordance
        self.tail_SLA_out = nn.Sequential(
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            # Output layer
            nn.Conv2d(self.ch, self.out_SLA_ch, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # Output tail 2 : Directional mean
        self.tail_DA_mean_out = nn.Sequential(
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),           
            # Output layer
            nn.Conv2d(self.ch, self.out_DA_mean_ch, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # Output tail 3 : Directional variance
        self.tail_DA_var_out = nn.Sequential(
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            # Output layer
            nn.Conv2d(self.ch, self.out_DA_var_ch, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # Output tail 4 : Directional weight
        self.tail_DA_weight_out = nn.Sequential(
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            # Output layer
            nn.Conv2d(self.ch, self.out_DA_weight_ch, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # Output tail 5 : Maneuver point
        self.tail_entry_out = nn.Sequential(
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            # Output layer
            nn.Conv2d(self.ch, self.out_entry_ch, 1, stride=1, padding=0),
            nn.Sigmoid()
        )
        # Output tail 6 : Maneuver point
        self.tail_exit_out = nn.Sequential(
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            nn.Conv2d(self.ch, self.ch, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.BatchNorm2d(self.ch),
            # Output layer
            nn.Conv2d(self.ch, self.out_exit_ch, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):

        #x = out = self.dropout(x)

        # Layer 0 (256)
        out0 = self.layer0(x)

        # Layer 1 (128, 64)
        out1 = self.layer1_1(out0)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        out1 = self.bn1_1(out1)
        out1 = self.layer1_2(out1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        out1 = self.bn1_2(out1)
        out1 = self.layer1_3(out1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        out1 = self.bn1_3(out1)
        # Pooling 1
        out1_pool = self.pool(out1)

        # Layer 2 (64, 128)
        out2 = self.layer2_1(out1_pool)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)
        out2 = self.bn2_1(out2)
        out2 = self.layer2_2(out2)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)
        out2 = self.bn2_2(out2)
        out2 = self.layer2_3(out2)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)
        out2 = self.bn2_3(out2)
        # Pooling 2
        out2_pool = self.pool(out2)

        # Layer 3 (32, 256)
        out3 = self.layer3_1(out2_pool)
        out3 = self.relu(out3)
        out3 = self.dropout(out3)
        out3 = self.bn3_1(out3)
        out3 = self.layer3_2(out3)
        out3 = self.relu(out3)
        out3 = self.dropout(out3)
        out3 = self.bn3_2(out3)
        out3 = self.layer3_3(out3)
        out3 = self.relu(out3)
        out3 = self.dropout(out3)
        out3 = self.bn3_3(out3)
        # Pooling 3
        out3_pool = self.pool(out3)

        # Layer 4 (16, 512)
        out4 = self.layer4_1(out3_pool)
        out4 = self.relu(out4)
        out4 = self.dropout(out4)
        out4 = self.bn4_1(out4)
        out4 = self.layer4_2(out4)
        out4 = self.relu(out4)
        out4 = self.dropout(out4)
        out4 = self.bn4_2(out4)
        out4 = self.layer4_3(out4)
        out4 = self.relu(out4)
        out4 = self.dropout(out4)
        out4 = self.bn4_3(out4)
        # Pooling 4
        out4_pool = self.pool(out4)

        # Layer 5 (8, 1024)
        out5 = self.layer5_1(out4_pool)
        out5 = self.relu(out5)
        out5 = self.dropout(out5)
        out5 = self.bn5_1(out5)
        out5 = self.layer5_2(out5)
        out5 = self.relu(out5)
        out5 = self.dropout(out5)
        out5 = self.bn5_2(out5)
        out5 = self.layer5_3(out5)
        out5 = self.relu(out5)
        out5 = self.dropout(out5)
        out5 = self.bn5_3(out5)
        # Pooling 5
        out5_upsample = self.upsample(out5)

        # Layer 6 (16, 512)
        out5_cat = torch.cat((out5_upsample, out4), dim=1)
        out6 = self.layer6_1(out5_cat)
        out6 = self.relu(out6)
        out6 = self.dropout(out6)
        out6 = self.bn6_1(out6)
        out6 = self.layer6_2(out6)
        out6 = self.relu(out6)
        out6 = self.dropout(out6)
        out6 = self.bn6_2(out6)
        out6 = self.layer6_3(out6)
        out6 = self.relu(out6)
        out6 = self.dropout(out6)
        out6 = self.bn6_3(out6)
        # Upsampling 6
        out6_upsample = self.upsample(out6)

        # Layer 7 (32, 256)
        out6_cat = torch.cat((out6_upsample, out3), dim=1)
        out7 = self.layer7_1(out6_cat)
        out7 = self.relu(out7)
        out7 = self.dropout(out7)
        out7 = self.bn7_1(out7)
        out7 = self.layer7_2(out7)
        out7 = self.relu(out7)
        out7 = self.dropout(out7)
        out7 = self.bn7_2(out7)
        out7 = self.layer7_3(out7)
        out7 = self.relu(out7)
        out7 = self.dropout(out7)
        out7 = self.bn7_3(out7)
        # Upsampling 7
        out7_upsample = self.upsample(out7)

        # Layer 8 (64, 128)
        out7_cat = torch.cat((out7_upsample, out2), dim=1)
        out8 = self.layer8_1(out7_cat)
        out8 = self.relu(out8)
        out8 = self.dropout(out8)
        out8 = self.bn8_1(out8)
        out8 = self.layer8_2(out8)
        out8 = self.relu(out8)
        out8 = self.dropout(out8)
        out8 = self.bn8_2(out8)
        out8 = self.layer8_3(out8)
        out8 = self.relu(out8)
        out8 = self.dropout(out8)
        out8 = self.bn8_3(out8)
        # Upsampling 8
        out8_upsample = self.upsample(out8)

        # Layer 9 (128, 64)
        out8_cat = torch.cat((out8_upsample, out1), dim=1)
        out9 = self.layer9_1(out8_cat)
        out9 = self.relu(out9)
        out9 = self.dropout(out9)
        out9 = self.bn9_1(out9)
        out9 = self.layer9_2(out9)
        out9 = self.relu(out9)
        out9 = self.dropout(out9)
        out9 = self.bn9_2(out9)
        out9 = self.layer9_3(out9)
        out9 = self.relu(out9)
        out9 = self.dropout(out9)
        out9 = self.bn9_3(out9)

        # Output tail 1: Soft lane affordance
        out_SLA = self.tail_SLA_out(out9)

        # Output tail 2: Directional mean
        out_DA_mean = self.tail_DA_mean_out(out9)

        # Output tail 3: Directional variance
        out_DA_var = self.tail_DA_var_out(out9)

        # Output tail 4: Directional weight
        out_DA_weight = self.tail_DA_weight_out(out9)

        # Output tail 5: Maneuver point
        out_entry = self.tail_entry_out(out9)  # (batch_n,1,128,128)
         # Output tail 6: Maneuver point
        out_exit = self.tail_exit_out(out9)  # (batch_n,1,128,128)
        # NOTE: To remove irrelevant output - points can only exist at edges
        mask = (out_entry > 10)
        mask[:,:,5:-5,5:-5] = True
        out_entry = torch.where(mask, torch.tensor(0.0).to("cuda:0"), out_entry)
        out_exit = torch.where(mask, torch.tensor(0.0).to("cuda:0"), out_exit)

        # Rescale directional mean [0,1] -> [0, 2*pi]
        out_DA_mean = out_DA_mean * 2.0*np.pi      

        # Normalize direcitonal weight
        out_DA_weight_sum = torch.sum(out_DA_weight.detach(), 1).unsqueeze(1)
        out_DA_weight = torch.div(out_DA_weight, out_DA_weight_sum)

        # Stack output into single tensor (batch_n, 11, n, n)
        out = torch.cat((out_SLA,
                         out_DA_mean,
                         out_DA_var,
                         out_DA_weight,
                         out_entry,
                         out_exit), dim=1)

        return out
