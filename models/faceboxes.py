import tensorflow as tf
import tf.keras.layers
from tensorflow.keras import Sequential

class BasicConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super(BasicConv2D, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        output = self.conv(inputs)
        output = self.bn(output)
        output = self.relu(output)

        return output


class CRelu(tf.keras.layers.Layer):

  def __init__(self, filters, kernel_size, strides, padding):
        super(CRelu, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

  def call(self, inputs, **kwargs):
        output = self.conv(inputs)
        output = self.bn(output)
        output = tf.keras.layers.concatenate([x, -x], axis=1)
        output = self.relu(output)

        return output


class Inception(tf.Module):

  def __init__(self):
    super(Inception, self).__init__()
    self.branch1x1 = BasicConv2D(filters=128,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding="same")
    self.branch1x1_2 = BasicConv2D(filters=128,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")

    self.branch3x3_reduce = BasicConv2D(filters=128,
                                        kernel_size=(1, 1),
                                        strides=1,
                                        padding="same")
    self.branch3x3 = BasicConv2D(filters=24,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding="same")
    self.branch3x3_reduce_2 = BasicConv2D(filters=128,
                                          kernel_size=(1, 1),
                                          strides=1,
                                          padding="same")
    self.branch3x3_2 = BasicConv2D(filters=24,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="same")
    self.branch3x3_3 = BasicConv2D(filters=32,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="same")

  def __call__(self, x):
    branch1x1 = self.branch1x1(x)
    branch1x1_pool = tf.keras.layers.AvgPool2D(pool_size=(3, 3),
                                             strides=1,
                                             padding="same")
    branch1x1_2 = self.branch1x1_2(branch1x1_pool)

    branch3x3_reduce = self.branch3x3_reduce(x)
    branch3x3 = self.branch3x3(branch3x3_reduce)

    branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
    branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)

    branch3x3_3 = self.branch3x3_3(branch3x3_2)

    outputs = tf.keras.layers.concatenate([branch1x1, branch1x1_2, branch3x3, branch3x3_3], axis=-1)

    return outputs


class FaceBoxes(tf.keras.Model):

  def __init__(self, phase, size, num_classes):
    super(FaceBoxes, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.size = size

    # self.conv1 = CRelu(3, 24, kernel_size=7, strides=4, padding='same')
    #   self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)
    self.conv1 = tf.keras.Sequential([CRelu(filters=24,
                                            kernel_size=(7, 7),
                                            strides=4,
                                            padding="same")])
    self.conv2 = tf.keras.Sequential([CRelu(filters=48,
                                            kernel_size=(5, 5),
                                            strides=2,
                                            padding="same")])

    self.inception1 = Inception()
    self.inception2 = Inception()
    self.inception3 = Inception()

    self.conv3_1 = BasicConv2D(filters=128,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same")

    self.conv3_2 = BasicConv2D(filters=128,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same")

    self.conv4_1 = BasicConv2D(filters=256,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same")
    self.conv4_2 = BasicConv2D(filters=128,
                               kernel_size=(3, 3),
                               strides=2,
                               padding="same")

    self.loc, self.conf = self.multibox(self.num_classes)

    if self.phase == 'test':
        self.softmax = tf.keras.layers.Softmax()

    # if self.phase == 'train':
    #     for m in self.modules():
    #         if isinstance(m, tf.nn.conv2d):
    #             if m.bias is not None:
    #                 tf.nn.init.xavier_normal_(m.weight.data)
    #                 m.bias.data.fill_(0.02)
    #             else:
    #                 m.weight.data.normal_(0, 0.01)
    #         elif isinstance(m, tf.nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

  def multibox(self, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers += [tf.nn.conv2d(128, 21 * 4, kernel_size=3, padding=1)]
    conf_layers += [tf.nn.conv2d(128, 21 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [tf.nn.conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [tf.nn.conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [tf.nn.conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [tf.nn.conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]

    model = Sequential()
    model.add(*loc_layers)
    model.add(*conf_layers)

    return model
    # return tf.nn.Sequential(*loc_layers), tf.nn.Sequential(*conf_layers)

  def __call__(self, x):

    detection_sources = list()
    loc = list()
    conf = list()

    x = self.conv1(x)
    max_pool_2d_1 = tf.keras.layers.AvgPool2D(pool_size=(3, 3),
                                              strides=2,
                                              padding="valid")
    x = max_pool_2d_1(x)
    x = self.conv2(x)
    max_pool_2d_2 = tf.keras.layers.AvgPool2D(pool_size=(3, 3),
                                              strides=2,
                                              padding="valid")
    x = max_pool_2d_2(x)
    x = self.inception1(x)
    x = self.inception2(x)
    x = self.inception3(x)
    detection_sources.append(x)

    x = self.conv3_2(x)
    x = self.conv3_1(x)
    detection_sources.append(x)

    x = self.conv4_1(x)
    x = self.conv4_2(x)
    detection_sources.append(x)

    for (x, l, c) in zip(detection_sources, self.loc, self.conf):
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())

    loc = tf.keras.layers.concatenate([o.view(o.size(0), -1) for o in loc], axis=1)
    conf = tf.keras.layers.concatenate([o.view(o.size(0), -1) for o in conf], axis=1)

    if self.phase == "test":
      output = (loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)))
    else:
      output = (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes))

    return output


