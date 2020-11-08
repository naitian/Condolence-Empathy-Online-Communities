import argparse

def main(args):
    num_gpus = args.num_gpus
    gpu_offset = args.gpu_offset
    for i in range(args.month_offset, args.month_offset + args.num_months):
        timeframe = "2018-{:02}".format((i % 12) + 1)
        gpu = "cuda:{}".format((i % num_gpus) + gpu_offset)
        # gpu = "cpu"
        print(timeframe, gpu, end=" ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetches parent comment and submission references"
    )
    parser.add_argument(
        "month_offset",
        action="store",
        type=int,
        help="which month to start at (0-11)",
    )
    parser.add_argument(
        "num_months",
        action="store",
        type=int,
        help="number of months to parse",
    )
    parser.add_argument(
        "gpu_offset",
        action="store",
        type=int,
        help="0-7 which GPU to start at",
    )
    parser.add_argument(
        "num_gpus",
        action="store",
        type=int,
        help="Number of GPUs (1-8)",
    )
    main(parser.parse_args())
