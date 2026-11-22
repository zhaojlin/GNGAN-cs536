
import os
import torch

from absl import app, flags
from pytorch_gan_metrics import get_inception_score_and_fid

import train

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_path", None, "Path to trained checkpoint"
)
flags.DEFINE_enum(
    "weights", "ema_G", ["net_G", "ema_G"],
    "Generator weights to evaluate"
)


def main(argv):
    del argv

    train.set_seed(FLAGS.seed)

    # Load the exact checkpoint requested.
    ckpt = torch.load(
        FLAGS.checkpoint_path,
        map_location=train.device
    )
    print("Checkpoint step:", ckpt["step"])
    print("Generator weights:", FLAGS.weights)

    # Reuse the original generator architecture.
    generator = train.net_G_models[FLAGS.arch](
        FLAGS.z_dim, FLAGS.n_classes
    ).to(train.device)

    generator.load_state_dict(ckpt[FLAGS.weights])
    generator.eval()

    # Generate the images once, using the original method.
    print("Generating", FLAGS.num_images, "images...")
    images = train.generate_images(generator)

    # Evaluate the same images against both reference sets.
    for split in ["train", "test"]:
        stats_path = f"./stats/cifar10.{split}.npz"

        if not os.path.isfile(stats_path):
            raise FileNotFoundError(stats_path)

        print(f"\nEvaluating CIFAR-10 {split}...")
        (is_score, is_std), fid = get_inception_score_and_fid(
            images, stats_path, verbose=True
        )

        print(
            f"RESULT step={ckpt['step']} "
            f"weights={FLAGS.weights} split={split} "
            f"IS={is_score:.4f} IS_std={is_std:.4f} "
            f"FID={fid:.4f}",
            flush=True
        )


if __name__ == "__main__":
    app.run(main)
