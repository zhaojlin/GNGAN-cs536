# SN-GAN baseline on CIFAR-10

Abdul Mannan adapted the team’s GN-GAN training loop to apply spectral normalization to the discriminator’s convolutional and linear layers. The experiment uses the same standard CNN architecture, CIFAR-10 data and evaluator as Member A. The shared code comes from [Jialin Zhao’s GN baseline](https://github.com/zhaojlin/GNGAN-cs536/tree/7c221cff4715badf3e64f0de5ddb1ded3d70271a), based on the [authors’ implementation](https://github.com/basiclab/GNGAN-PyTorch).

## Experiment

The SN-GAN run used one NVIDIA A40 on Spiedie, seed 0, and 200,000 generator updates. The configuration sets one discriminator update per generator update, discriminator and generator batch sizes of 64 and 128, and a learning rate of 0.0002 for both optimizers.

From the repository root in the configured Spiedie environment, submit the job with:

```bash
conda activate sn200k
sbatch --partition=gpucompute-a40 spiedie_sn_200k.sbatch
```

## Results

[SN-GAN scores](results.csv) contains the three final evaluation rows reported from the Spiedie run. [GN-GAN and SN-GAN scores](comparison.csv) also includes Member A’s saved GN results. The evaluator uses 50,000 generated images for Inception Score and training-reference FID, and 10,000 for test-reference FID. The IS standard deviation measures variation across sample splits within one run.

| Final generator at 200,000 updates | IS ↑ | Train FID ↓ | Test FID ↓ |
| --- | ---: | ---: | ---: |
| GN-GAN | 7.5158 | 19.4485 | 23.3293 |
| SN-GAN | 7.7857 | 16.7693 | 20.8078 |

The final SN-GAN generator had a test FID 2.52 points lower than the final GN-GAN generator in these runs. The SN-GAN generator with exponentially averaged weights reached IS 8.0320 and test FID 19.6835. Jialin used an RTX A6000 and Abdul Mannan used an A40. A comparison of training speed requires matched hardware and software.

The runner saves measured training-loop time in `logs/SN-GAN_CIFAR10_CNN_200k/timing.json` on Spiedie. This measure includes intermediate sampling and evaluation. It excludes setup and the standalone final evaluations.
