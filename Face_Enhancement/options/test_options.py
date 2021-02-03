# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def initialize(parser):
        parser.which_epoch = "latest"
        parser.serial_batches = True
        parser.no_flip = True
        parser.phase = "test"
        parser.how_many = float("inf")
        parser.preprocess_mode = "resize"
        parser.crop_size = 256
        parser.load_size = 256
        parser.display_winsize = 256
        parser.tensorboard_log = False
        parser.label_nc = 18
        parser.no_instance = True
        parser.batchSize = 4
        parser.no_parsing_map = True
        parser.dataroot = "./Face_Enhancement/datasets/cityscapes/"
        parser.serial_batches = False
        parser.nThreads = 0
        parser.netG = "spade"
        parser.ngf = 64
        parser.num_upsampling_layers = 'normal'
        parser.aspect_ratio = 1.0
        parser.use_vae = False
        parser.injection_layer = 'all'
        parser.norm_G = 'spectralspadesyncbatch3x3'
        parser.norm_D = 'spectralinstance'
        parser.norm_E = 'spectralinstance'
        parser.contain_dontcare_label = False
        parser.semantic_nc = 18
        parser.init_type = 'normal'
        parser.init_variance = 0.02
        parser.checkpoints_dir = './Face_Enhancement/checkpoints'
        parser.add_argument(
            "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU"
        )
        parser.add_argument(
            "--old_face_folder", type=str, default="", help="The folder name of input old face"
        )
        parser.add_argument(
            "--old_face_label_folder", type=str, default="", help="The folder name of input old face label"
        )
        parser.add_argument(
            "--name",
            type=str,
            default="label2coco",
            help="name of the experiment. It decides where to store samples and models",)
        parser.add_argument("--results_dir", type=str, default="./results/", help="saves results here.")
        parser.isTrain = False
        return parser
