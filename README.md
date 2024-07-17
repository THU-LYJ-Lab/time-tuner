# TimeTuner &mdash; Official PyTorch implementation

> **Towards More Accurate Diffusion Model Acceleration with A Timestep Tuner (CVPR 2024)** <br>
> Mengfei Xia, Yujun Shen, Changsong Lei, Yu Zhou, Deli Zhao, Ran Yi, Wenping Wang, Yong-Jin Liu <br>

[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xia_Towards_More_Accurate_Diffusion_Model_Acceleration_with_A_Timestep_Tuner_CVPR_2024_paper.pdf)]

Abstract: *A diffusion model, which is formulated to produce an image using thousands of denoising steps, usually suffers from a slow inference speed. Existing acceleration algorithms simplify the sampling by skipping most steps yet exhibit considerable performance degradation. By viewing the generation of diffusion models as a discretized integral process, we argue that the quality drop is partly caused by applying an inaccurate integral direction to a timestep interval. To rectify this issue, we propose a timestep tuner that helps find a more accurate integral direction for a particular interval at the minimum cost. Specifically, at each denoising step, we replace the original parameterization by conditioning the network on a new timestep, enforcing the sampling distribution towards the real one. Extensive experiments show that our plug-in design can be trained efficiently and boost the inference performance of various state-of-the-art acceleration methods, especially when there are few denoising steps. For example, when using 10 denoising steps on LSUN Bedroom dataset, we improve the FID of DDIM from 9.65 to 6.07, simply by adopting our method for a more appropriate set of timesteps.*

## TODO List

- [ ] Release inference code.
- [ ] Release training code.

## References

If you find the code useful for your research, please consider citing

```bib
@inproceedings{xia2024timetuner,
  title={Towards More Accurate Diffusion Model Acceleration with A Timestep Tuner},
  author={Xia, Mengfei and Shen, Yujun and Lei, Changsong and Zhou, Yu and Zhao, Deli and Yi, Ran and Wang, Wenping and Liu, Yong-Jin},
  booktitle={CVPR},
  year={2024},
}
```

## LICENSE

The project is under [MIT License](./LICENSE), and is for research purpose ONLY.