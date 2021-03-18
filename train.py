from __future__ import print_function
import os
import argparse
from data import AnnotationTransform, VOCDetection, detection_collate, preproc, cfg
from models.faceboxes import FaceBoxes

parser = argparse.ArgumentParser(description='FaceBoxes Training')
parser.add_argument('--training_dataset', default='./data/WIDER_FACE', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
# parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=300, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

img_dim = 1024 # only 1024 is supported
rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
# num_gpu = args.ngpu
num_workers = args.num_workers
batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
max_epoch = args.max_epoch
training_dataset = args.training_dataset
save_folder = args.save_folder
gpu_train = cfg['gpu_train']

"""def get_model():
    model = faceboxes.FacesBoxes(num_classes)

    model.build(input_shape=(None, img_dim, image_width, channels))
    model.summary()

    return model
"""

net = FaceBoxes('train', img_dim, num_classes)
print("Printing net...")
print(net)


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

# if args.resume_net is not None:
#     print('Loading resume network...')
#     state_dict = torch.load(args.resume_net)
#     # create new OrderedDict that does not contain `module.`
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         head = k[:7]
#         if head == 'module.':
#             name = k[7:] # remove `module.`
#         else:
#             name = k
#         new_state_dict[name] = v
#     net.load_state_dict(new_state_dict)

cudnn.benchmark = True
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
 """" ---------------------------------------------------------------------------- """"
# get the original_dataset
train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

# create model
model = get_model()

# define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adadelta()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
