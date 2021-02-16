import tensorflow as tf
import tensorflow.nn as nn
import tf.keras.layers as layers


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = layers.BatchNormalization(out_channels, epsilon=1e-5)

        # out_channels = tf.nn.conv2d(in_channels, )

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.relu(x)


class CRelu(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(CRelu, self).__init__()
    # self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = layers.BatchNormalization(out_channels, eps=1e-5)

  def __call__(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = tf.concat([x, -x], 1)
    x = nn.relu(x)
    return x

  def multibox(self, num_classes):
    loc_layers = []
    conf_layers = []
    # loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1)]
    # conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1)]
    # loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    # conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    # loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    # conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    # return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)


class Inception(nn.Module):

  def __init__(self):
    super(Inception, self).__init__()
    self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
    self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
    self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

  def __call__(self, x):
    branch1x1 = self.branch1x1(x)

    branch1x1_pool = nn.avg_pool2d(x, ksize=3, stride=1, padding=1)
    branch1x1_2 = self.branch1x1_2(branch1x1_pool)

    branch3x3_reduce = self.branch3x3_reduce(x)
    branch3x3 = self.branch3x3(branch3x3_reduce)

    branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
    branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
    branch3x3_3 = self.branch3x3_3(branch3x3_2)

    outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
    return tf.concat(outputs, 1)


class FaceBoxes(nn.Module):

  def __init__(self, phase, size, num_classes):
    super(FaceBoxes, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.size = size

    self.conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3)
    self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)

    self.inception1 = Inception()
    self.inception2 = Inception()
    self.inception3 = Inception()
    self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
    self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

    self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
    self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

    # self.loc, self.conf = self.multibox(self.num_classes)

    # if self.phase == 'test':
    #     self.softmax = nn.Softmax(dim=-1)

    # if self.phase == 'train':
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             if m.bias is not None:
    #                 nn.init.xavier_normal_(m.weight.data)
    #                 m.bias.data.fill_(0.02)
    #             else:
    #                 m.weight.data.normal_(0, 0.01)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

  def __call__(self, x):

    detection_sources = list()
    loc = list()
    conf = list()

    x = self.conv1(x)
    max_pool_2d_1 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding=1)
    x = max_pool_2d_1(x)
    x = self.conv2(x)
    max_pool_2d_2 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding=1)
    x = max_pool_2d_2(x)
    x = self.inception1(x)
    x = self.inception2(x)
    x = self.inception3(x)
    detection_sources.append(x)

    x = self.conv3_1(x)
    x = self.conv3_2(x)
    detection_sources.append(x)

    x = self.conv4_1(x)
    x = self.conv4_2(x)
    detection_sources.append(x)

    for (x, l, c) in zip(detection_sources, self.loc, self.conf):
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())

    loc = tf.concat([o.view(o.size(0), -1) for o in loc], 1)
    conf = tf.concat([o.view(o.size(0), -1) for o in conf], 1)

    if self.phase == "test":
      output = (loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)))
    else:
      output = (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes))

    return output


