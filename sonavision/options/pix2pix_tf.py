import argparse
import time
import pathlib


class Pix2PixOptions:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "--name",
            default="pix2pix",
            help="name of the experiment. It decides where to store samples and models",
            required=True,
        )
        self.parser.add_argument(
            "--dataset_dir",
            required=True,
            default="datasets/dfki-divers",
            help="path to dataset",
        )
        self.parser.add_argument(
            "--train_subdir",
            required=False,
            default="train",
            help="train subfolder in dataset",
        )
        self.parser.add_argument(
            "--test_subdir",
            required=False,
            default="test",
            help="test subfolder in dataset",
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=8, help="train batch size"
        )
        self.parser.add_argument(
            "--test_batch_size", type=int, default=5, help="test batch size"
        )
        # self.parser.add_argument('--ngf', type=int, default=64)
        # self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument(
            "--input_shape",
            type=int,
            nargs=2,
            default=[256, 512],
            help="input shape, format: height width",
        )
        self.parser.add_argument(
            "--output_shape",
            type=int,
            nargs=2,
            default=[256, 512],
            help="output shape, format: height width",
        )
        self.parser.add_argument(
            "--crop_shape",
            type=int,
            nargs=2,
            default=[0, 0],
            help="crop shape (0 0 is false) format: height width",
        )
        self.parser.add_argument(
            "--resize",
            type=int,
            nargs=2,
            default=[0, 0],
            help="resize scale (0 is false) format: height width",
        )
        self.parser.add_argument(
            "--fliplr",
            action="store_true",
            default=True,
            help="random fliplr True or False",
        )

        self.parser.add_argument(
            "--train_epoch", type=int, default=200, help="number of train epochs"
        )
        self.parser.add_argument(
            "--lrD", type=float, default=0.0002, help="learning rate, default=0.0002"
        )
        self.parser.add_argument(
            "--lrG", type=float, default=0.0002, help="learning rate, default=0.0002"
        )
        self.parser.add_argument(
            "--ngf", type=int, default=32, help="base filters for generator"
        )
        self.parser.add_argument(
            "--ndf", type=int, default=32, help="base filters for discriminator"
        )
        self.parser.add_argument(
            "--lambda_l1", type=float, default=100, help="lambda for L1 loss"
        )
        self.parser.add_argument(
            "--beta1", type=float, default=0.5, help="beta1 for Adam optimizer"
        )
        self.parser.add_argument(
            "--beta2", type=float, default=0.999, help="beta2 for Adam optimizer"
        )
        self.parser.add_argument(
            "--results_dir", default="results", help="results save path"
        )

        self.parser.add_argument("--logs_dir", default="log", help="logs dir")
        self.parser.add_argument(
            "--checkpoints_dir",
            default="checkpoints",
            help="checkpoints save path",
        )

        self.parser.add_argument(
            "--model_save_dir",
            default="models",
            help="models save path",
        )

        self.parser.add_argument(
            "--image_format",
            default="png",
            help="image format for input",
        )

        self.parser.add_argument(
            "--num_images_per_image",
            default=3,
            type=int,
            help="number of images per sample",
        )

        self.parser.add_argument(
            "--arch_type",
            required=True,
            default="with-camera-vanilla",
            choices=["with-camera-early-fusion", "with-camera-late-fusion", "with-out-camera"],
            help="architecture type",
        )

        self.initialized = True

    def parse_inline(self, args):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(args)
        args = vars(self.opt)
        print("------------ Options -------------")
        print("time: %s" % self.time)
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        # create required directories
        results_dir = pathlib.Path(self.time) / self.opt.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        checkpoints_dir = pathlib.Path(self.time) / self.opt.checkpoints_dir
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        model_save_dir = pathlib.Path(self.time) / self.opt.model_save_dir
        model_save_dir.mkdir(parents=True, exist_ok=True)

        logs_dir = pathlib.Path(self.time) / self.opt.logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)

        # replace values for resuls, checkpoints, model and logs in self.opt
        self.opt.results_dir = results_dir
        self.opt.checkpoints_dir = checkpoints_dir
        self.opt.model_save_dir = model_save_dir
        self.opt.logs_dir = logs_dir

        # save to the disk
        file_name = pathlib.Path(self.time) / "opt.txt"
        with open(file_name, "wt") as opt_file:
            opt_file.write("------------ Options -------------\n")
            opt_file.write("time: %s \n" % self.time)
            for k, v in sorted(args.items()):
                opt_file.write("%s: %s \n" % (str(k), str(v)))
            opt_file.write("-------------- End ----------------\n")

        return self.opt

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print("------------ Options -------------")
        print("time: %s" % self.time)
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        # create required directories
        results_dir = pathlib.Path(self.time) / self.opt.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        checkpoints_dir = pathlib.Path(self.time) / self.opt.checkpoints_dir
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        model_save_dir = pathlib.Path(self.time) / self.opt.model_save_dir
        model_save_dir.mkdir(parents=True, exist_ok=True)

        logs_dir = pathlib.Path(self.time) / self.opt.logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)

        # replace values for resuls, checkpoints, model and logs in self.opt
        self.opt.results_dir = results_dir
        self.opt.checkpoints_dir = checkpoints_dir
        self.opt.model_save_dir = model_save_dir
        self.opt.logs_dir = logs_dir

        # save to the disk
        file_name = pathlib.Path(self.time) / "opt.txt"
        with open(file_name, "wt") as opt_file:
            opt_file.write("------------ Options -------------\n")
            opt_file.write("time: %s \n" % self.time)
            for k, v in sorted(args.items()):
                opt_file.write("%s: %s \n" % (str(k), str(v)))
            opt_file.write("-------------- End ----------------\n")

        return self.opt
