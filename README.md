# TimeTuner &mdash; Official PyTorch implementation

> **Towards More Accurate Diffusion Model Acceleration with A Timestep Tuner (CVPR 2024)** <br>
> Mengfei Xia, Yujun Shen, Changsong Lei, Yu Zhou, Deli Zhao, Ran Yi, Wenping Wang, Yong-Jin Liu <br>

[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xia_Towards_More_Accurate_Diffusion_Model_Acceleration_with_A_Timestep_Tuner_CVPR_2024_paper.pdf)]

Abstract: *A diffusion model, which is formulated to produce an image using thousands of denoising steps, usually suffers from a slow inference speed. Existing acceleration algorithms simplify the sampling by skipping most steps yet exhibit considerable performance degradation. By viewing the generation of diffusion models as a discretized integral process, we argue that the quality drop is partly caused by applying an inaccurate integral direction to a timestep interval. To rectify this issue, we propose a timestep tuner that helps find a more accurate integral direction for a particular interval at the minimum cost. Specifically, at each denoising step, we replace the original parameterization by conditioning the network on a new timestep, enforcing the sampling distribution towards the real one. Extensive experiments show that our plug-in design can be trained efficiently and boost the inference performance of various state-of-the-art acceleration methods, especially when there are few denoising steps. For example, when using 10 denoising steps on LSUN Bedroom dataset, we improve the FID of DDIM from 9.65 to 6.07, simply by adopting our method for a more appropriate set of timesteps.*

## Supported Models and Algorithms

### Models

We support the following four types of diffusion models. You can set the model type by the argument `model_type` in the function `model_wrapper`.

| Model Type                                        | Training Objective                                           | Example Paper                                                      |
| ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| "noise": noise prediction model $\boldsymbol\epsilon_\theta$ | $\mathbb E_{\mathbf x_{0},\boldsymbol\epsilon,t}\left[\omega_1(t)\|\|\boldsymbol\epsilon_\theta(\mathbf x_t,t)-\boldsymbol\epsilon\|\|_2^2\right]$ | [DDPM](https://arxiv.org/abs/2006.11239), [Stable-Diffusion](https://github.com/CompVis/stable-diffusion) |
| "\mathbf x_start": data prediction model $\mathbf x_\theta$       | $\mathbb E_{\mathbf x_0,\boldsymbol\epsilon,t}\left[\omega_2(t)\|\|\mathbf x_\theta(\mathbf x_t,t)-\mathbf x_0\|\|_2^2\right]$ | [DALL·E 2](https://arxiv.org/abs/2204.06125)                 |
| "v": velocity prediction model $v_\theta$         | $\mathbb E_{\mathbf x_0,\boldsymbol\epsilon,t}\left[\omega_3(t)\|\|v_\theta(\mathbf x_t,t)-(\alpha_t\boldsymbol\epsilon - \sigma_t \mathbf x_0)\|\|_2^2\right]$ | [Imagen Video](https://arxiv.org/abs/2210.02303)             |
| "score": marginal score function $s_\theta$       | $\mathbb E_{\mathbf x_0,\boldsymbol\epsilon,t}\left[\omega_4(t)\|\|\sigma_t s_\theta(\mathbf x_t,t)+\boldsymbol\epsilon\|\|_2^2\right]$ | [ScoreSDE](https://arxiv.org/abs/2011.13456)                 |

### Sampling Types

We support the following three types of sampling by diffusion models. You can set the argument `guidance_type` in the function `model_wrapper`.

| Sampling Type                               | Equation for Noise Prediction Model                          | Example Paper                                                      |
| ------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| "uncond": unconditional sampling            | $\tilde{\boldsymbol\epsilon}_\theta(\mathbf x_t,t)=\boldsymbol\epsilon_\theta(\mathbf x_t,t)$        | [DDPM](https://arxiv.org/abs/2006.11239)                     |
| "classifier": classifier guidance           | $\tilde{\boldsymbol\epsilon}\boldsymbol\epsilon_\theta(\mathbf x_t,t,c)=\boldsymbol\epsilon_\theta(\mathbf x_t,t)-s\cdot\sigma_t\nabla_{\mathbf x_t}\log q_\phi(\mathbf x_t,t,c)$ | [ADM](https://arxiv.org/abs/2105.05233), [GLIDE](https://arxiv.org/abs/2112.10741) |
| "classifier-free": classifier-free guidance | $\tilde{\boldsymbol\epsilon}\boldsymbol\epsilon_\theta(\mathbf x_t,t,c)=s\cdot \boldsymbol\epsilon_\theta(\mathbf x_t,t,c)+(1-s)\cdot\boldsymbol\epsilon_\theta(\mathbf x_t,t)$ | [DALL·E 2](https://arxiv.org/abs/2204.06125), [Imagen](https://arxiv.org/abs/2205.11487), [Stable-Diffusion](https://github.com/CompVis/stable-diffusion) |

## Code Examples for Sampling

Pre-trained timesteps of [LDM](https://github.com/CompVis/latent-diffusion) can be found [here](https://drive.google.com/file/d/1whJ-_d5EsQlTqNmiC4RIKhNSlC3AiDaK/view?usp=sharing).

### Example: Unconditional Sampling by TimeTuner

```python
from time_tuner import NoiseScheduleVP, model_wrapper, TimeTuner

# 1. Define the noise schedule.
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

## 2. Convert your discrete-time `model` to the continuous-time
## noise prediction model. Here is an example for a diffusion model
## `model` with the noise prediction type.
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type='noise',
    model_kwargs=model_kwargs,
    guidance_type='uncond',
)

# 3. Define TimeTuner for sampling.
time_tuner = TimeTuner(model_fn_continuous, noise_schedule)

# 4. Use corresponding pre-tuned `t_ratios` for sampling with NFE = 10.
x = time_tuner.ddim_sample(x=x_T,
                           num_steps=10
                           t_ratios=t_ratios,
                           eta=eta)
```

### Example: Classifier Guidance Sampling by TimeTuner

```python
from time_tuner import NoiseScheduleVP, model_wrapper, TimeTuner

# 1. Define the noise schedule.
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

# 2. Convert your discrete-time `model` to the continuous-time
# noise prediction model. Here is an example for a diffusion model
# `model` with the noise prediction type.
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type='noise',
    guidance_type='classifier',
    guidance_scale=guidance_scale,
    classifier_fn=classifier,
    model_kwargs=model_kwargs,
    classifier_kwargs=classifier_kwargs,
)

# 3. Define TimeTuner for sampling.
time_tuner = TimeTuner(model_fn_continuous, noise_schedule)

# 4. Use corresponding pre-tuned `t_ratios` for sampling with NFE = 10.
x = time_tuner.ddim_sample(x=x_T,
                           num_steps=10,
                           t_ratios=t_ratios,
                           eta=eta,
                           condition=condition)
```

### Example: Classifier-Free Guidance Sampling by TimeTuner

```python
from time_tuner import NoiseScheduleVP, model_wrapper, TimeTuner

# 1. Define the noise schedule.
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

# 2. Convert your discrete-time `model` to the continuous-time
# noise prediction model. Here is an example for a diffusion model
# `model` with the noise prediction type.
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type='noise',
    guidance_type='classifier-free',
    guidance_scale=guidance_scale,
    model_kwargs=model_kwargs,
)

# 3. Define TimeTuner for sampling.
time_tuner = TimeTuner(model_fn_continuous, noise_schedule)

# 4. Use corresponding pre-tuned `t_ratios` for sampling with NFE = 10.
x = time_tuner.ddim_sample(x=x_T,
                           num_steps=10,
                           t_ratios=t_ratios,
                           eta=eta,
                           condition=condition,
                           uncond_condition=uncond_condition)
```

### Use TimeTuner in your own code

It is very easy to combine TimeTuner with your own diffusion models. You can just copy the file `time_tuner.py` to your own code files and import it.

In each step, TimeTuner needs to replace the original input timestep condition with a tuned one. We in present support the commonly-used variance preserving (VP) noise schedule for both discrete-time and continuous-time DPMs:

- For discrete-time DPMs, we support a piecewise linear interpolation of $\log\alpha_t$  in the `NoiseScheduleVP` class. It can support all types of VP noise schedules.

- For continuous-time DPMs, we support linear schedule (as used in [DDPM](https://arxiv.org/abs/2006.11239) and [ScoreSDE](https://arxiv.org/abs/2011.13456)) in the `NoiseScheduleVP` class.

Moreover, TimeTuner is designed for the continuous-time DPMs. For discrete-time diffusion models, we also implement a wrapper function to convert the discrete-time diffusion models to the continuous-time diffusion models in the `model_wrapper` function.

## Training New Tuned Timesteps

It is also easy to optimize timesteps via TimeTuner upon your own diffusion models. Similarly, you need to first copy the file `time_tuner.py` to your own code files and import it.

After pre-defining the trajectory and sampler, you are then expected to prepare the corresponding data loader. Specifically, the `__getitem__` function for the dataset is required to be implemented with key `image` and `label` (optional for conditioned generation) as below:

```python
class Dataset(torch.data.utils.Dataset):
    def __getitem__(self, index):
        data_dict = dict()
        # Prepare the image.
        data_dict.update(image=image)
        # Prepare the label (optional).
        data_dict.update(label=label)
        return data_dict
```

Given the trajectory and the data loader, TimeTuner will accordingly optimize the given timesteps following either **sequential strategy** or **parallel strategy**, in which the latter one is capable of achieving on-par performance with extreme training acceleration. Beyond DDIM sampler, you are able to apply TimeTuner upon any sampler by simply implementing the one-step denoiser, and calling `TimeTuner.optimize_timesteps` with the parameter `step_fn`.

Note that we also implement `encode_fn` in `TimeTuner` and `cond_process_fn` in `model_wrapper`, for compatibility with [LDM](https://arxiv.org/abs/2112.10752).

### Example: Unconditional Training TimeTuner

```python
from time_tuner import NoiseScheduleVP, model_wrapper, TimeTuner

# 1. Define the noise schedule.
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

## 2. Convert your discrete-time `model` to the continuous-time
## noise prediction model. Here is an example for a diffusion model
## `model` with the noise prediction type.
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type='noise',
    model_kwargs=model_kwargs,
    guidance_type='uncond',
)

# 3. Define TimeTuner for optimizing, together with the DDIM sampler.
time_tuner = TimeTuner(model_fn_continuous, noise_schedule)
step_fn = time_tuner.ddim_step_fn
step_fn_kwargs = dict(eta=eta)
tune_type = 'sequential'

# 4. Optimize the preset timesteps with NFE = 10.
t_ratios = time_tuner.optimize_timesteps(data_loader=data_loader,
                                         step_fn=step_fn,
                                         num_steps=10,
                                         tune_type=tune_type,
                                         lr=lr,
                                         total_iters=total_iters,
                                         verbose=True,
                                         **step_fn_kwargs)
```

### Example: Classifier Guidance Training TimeTuner

```python
from time_tuner import NoiseScheduleVP, model_wrapper, TimeTuner

# 1. Define the noise schedule.
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

## 2. Convert your discrete-time `model` to the continuous-time
## noise prediction model. Here is an example for a diffusion model
## `model` with the noise prediction type.
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type='noise',
    guidance_type='classifier',
    guidance_scale=guidance_scale,
    classifier_fn=classifier,
    model_kwargs=model_kwargs,
    classifier_kwargs=classifier_kwargs,
)

# 3. Define TimeTuner for optimizing, together with the DDIM sampler.
time_tuner = TimeTuner(model_fn_continuous, noise_schedule)
step_fn = time_tuner.ddim_step_fn
step_fn_kwargs = dict(eta=eta)
tune_type = 'sequential'

# 4. Optimize the preset timesteps with NFE = 10.
t_ratios = time_tuner.optimize_timesteps(data_loader=data_loader,
                                         step_fn=step_fn,
                                         num_steps=10,
                                         tune_type=tune_type,
                                         lr=lr,
                                         total_iters=total_iters,
                                         verbose=True,
                                         **step_fn_kwargs)
```

### Example: Classifier-Free Guidance Training TimeTuner

```python
from time_tuner import NoiseScheduleVP, model_wrapper, TimeTuner

# 1. Define the noise schedule.
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

## 2. Convert your discrete-time `model` to the continuous-time
## noise prediction model. Here is an example for a diffusion model
## `model` with the noise prediction type.
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type='noise',
    guidance_type='classifier-free',
    guidance_scale=guidance_scale,
    model_kwargs=model_kwargs,
)

# 3. Define TimeTuner for optimizing, together with the DDIM sampler.
time_tuner = TimeTuner(model_fn_continuous, noise_schedule)
step_fn = time_tuner.ddim_step_fn
step_fn_kwargs = dict(eta=eta, uncond_condition=uncond_condition)
tune_type = 'sequential'

# 4. Optimize the preset timesteps with NFE = 10.
t_ratios = time_tuner.optimize_timesteps(data_loader=data_loader,
                                         step_fn=step_fn,
                                         num_steps=10,
                                         tune_type=tune_type,
                                         lr=lr,
                                         total_iters=total_iters,
                                         verbose=True,
                                         **step_fn_kwargs)
```

## TODO List

- [ ] Release inference code for DPM-Solver.
- [ ] Provide example codes upon LDM.

## References

If you find the code useful for your research, please consider citing

```bib
@inproceedings{xia2024timetuner,
  title={Towards More Accurate Diffusion Model Acceleration with A Timestep Tuner},
  author={Xia, Mengfei and Shen, Yujun and Lei, Changsong and Zhou, Yu and Zhao, Deli and Yi, Ran and Wang, Wenping and Liu, Yong-Jin},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
}
```

## LICENSE

The project is under [MIT License](./LICENSE), and is for research purpose ONLY.

## Acknowledgments

The whole codebase is build upon [DPM-Solver](https://github.com/LuChengTHU/dpm-solver). We highly appreciate Cheng Lu for the great efforts on this codebase.
