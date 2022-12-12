import argparse
import time
import pathlib


class Options:
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
            required=False,
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
            "--batch_size", type=int, default=1, help="train batch size"
        )
        self.parser.add_argument(
            "--test_batch_size", type=int, default=5, help="test batch size"
        )
        # self.parser.add_argument('--ngf', type=int, default=64)
        # self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument(
            "--input_size",
            type=int,
            nargs=2,
            default=[0, 0],
            help="input size, format: height width",
        )
        self.parser.add_argument(
            "--crop_size",
            type=int,
            nargs=2,
            default=[0, 0],
            help="crop size (0 0 is false) format: height width",
        )
        self.parser.add_argument(
            "--resize",
            type=int,
            nargs=2,
            default=[0, 0],
            help="resize scale (0 is false) format: height width",
        )
        self.parser.add_argument(
            "--fliplr", type=bool, default=True, help="random fliplr True or False"
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
            "--L1_lambda", type=float, default=100, help="lambda for L1 loss"
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

        self.initialized = True

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

        # save to the disk
        options_save_dir = pathlib.Path(self.opt.checkpoints_dir) / self.time
        options_save_dir.mkdir(parents=True, exist_ok=True)
        return self.opt
