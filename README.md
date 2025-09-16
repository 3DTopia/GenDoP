# GenDoP: Auto-regressive Camera Trajectory Generation as a Director of Photography [ICCV 2025]


<p align="center">
<a href="https://arxiv.org/abs/2504.07083"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://kszpxxzmc.github.io/GenDoP/"><img src="https://img.shields.io/badge/Project-Website-red"></a>
<a href="https://www.youtube.com/watch?v=UWvR_A7yFeI"><img src="https://img.shields.io/static/v1?label=Demo&message=Video&color=orange"></a>
<a href="https://huggingface.co/datasets/Dubhe-zmc/DataDoP"><img src="https://img.shields.io/static/v1?label=Dataset&message=Data&color=yellow"></a>
<a href="" target='_blank'>
<img src="https://visitor-badge.laobi.icu/badge?page_id=TODO" />
</a>
</p>

[**Paper**](https://arxiv.org/abs/2504.07083) | [**Project page**](https://kszpxxzmc.github.io/GenDoP) | [**Video**](https://www.youtube.com/watch?v=UWvR_A7yFeI) | [**Data**](https://huggingface.co/datasets/Dubhe-zmc/DataDoP) 

[Mengchen Zhang](https://kszpxxzmc.github.io), [Tong Wu✉️](https://wutong16.github.io), [Jing Tan](https://sparkstj.github.io/), [Ziwei Liu](https://liuziwei7.github.io/), [Gordon Wetzstein](https://stanford.edu/~gordonwz/), [Dahua Lin✉️](http://dahua.site/)

![Teaser Image](./assets/teaser.png)


## ✨ Updates
[2025-07-03] Released inference code and model checkpoints!

[2025-07-03] Released training code.

[2025-09-11] Released the curated trajectory dataset DataDoP along with its construction code.

<!-- [2025-07-15] Launched the Gradio demo. -->

## 📦 Install 
Make sure ```torch``` with CUDA is correctly installed. For training, we rely on ```flash-attn``` (requires at least Ampere GPUs like A100). For inference, older GPUs like V100 are also supported, although slower.
```
# clone
git clone https://github.com/3DTopia/GenDoP.git
cd GenDoP

# environment
conda create --name GenDoP python=3.10
conda activate GenDoP
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```
## 💡 Inference 

### Pretrained Models
We provide the following pretrained models:

| Model Type                  | Description                | Download Link |
|-----------------------------|----------------------------|---------------|
| text_motion | Text (motion)-to-Trajectory| [Download](https://huggingface.co/Dubhe-zmc/GenDoP/blob/main/checkpoints/text_motion.safetensors)  |
| text_directorial | Text (directorial)-to-Trajectory  | [Download](https://huggingface.co/Dubhe-zmc/GenDoP/blob/main/checkpoints/text_directorial.safetensors)  |
| text_rgbd   | Text & RGBD-to-Trajectory         | [Download](https://huggingface.co/Dubhe-zmc/GenDoP/blob/main/checkpoints/text_rgbd.safetensors)  |

### Minimal Example

**Note:** You may choose one of the following options: either input the text directly using `--text`, or provide both `--text_path` and `--text_key`. For more examples, please refer to [assets/examples](./assets/examples).

**Inference Commands** 

- Text (motion)-to-Trajectory

  ```bash
  python eval.py ArAE --workspace outputs --name text_motion/case1 --resume "checkpoints/text_motion.safetensors" \
      --cond_mode 'text' \
      --text "The camera remains static, then moves right, followed by moving forward while yawing right, and finally moving left and forward while continuing to yaw right."
  ```

- Text (directorial)-to-Trajectory

  ```bash
  python eval.py ArAE --workspace outputs --name text_directorial/case1 --resume "checkpoints/text_directorial.safetensors" \
      --cond_mode 'text' \
      --text "The camera starts static, moves down to reveal clouds, pitches up to show more formations, and returns to a static position."
  ```

- Text & RGBD-to-Trajectory

  ```bash
  python eval.py ArAE --workspace outputs --name text_rgbd/case1 --resume "checkpoints/text_rgbd.safetensors" \
      --cond_mode 'depth+image+text' \
      --text "The camera moves right and yaws left, highlighting the notebook and cup, then shifts forward to emphasize the subject's expression, before coming to a stop." \
      --text_path assets/examples/text_rgbd/case1_caption.json \
      --text_key 'Concise Interaction' \
      --image_path assets/examples/text_rgbd/case1_rgb.png \
      --depth_path assets/examples/text_rgbd/case1_depth.npy
  ```

### Visualization

**Note:** Our default visualization, as shown in the `*_traj_cleaning.png` files in our dataset, displays how the camera moves through the scene. It includes three views: front, top-down, and side perspectives. The colors transition from red to purple to show the sequence of movement. In the front view, you can observe vertical and horizontal movements (up, down, left, right), while the top-down view highlights forward and backward motion.

For a clearer visualization, consistent with the figures in our paper, you can use the following method:

**Install**

Follow the [official instructions](https://www.blender.org/download/) to install Blender. The Blender version we used is `blender-3.3.1-linux-x64`.

Then, install the required Python packages:
```bash
<path-to-blender>/<version>/python/bin/python3.10 -m pip install trimesh
<path-to-blender>/<version>/python/bin/python3.10 -m pip install matplotlib
```

**Visualize**

To visualize the trajectory, run:
```bash
<path-to-blender>/blender --background --python Blender_visualization/blender_visualize.py
```
Modify the `traj_p` variable in [Blender_visualization/blender_visualize.py](./Blender_visualization/blender_visualize.py) to specify the JSON file you want to visualize. This JSON file should follow the same format as the `*_transforms_cleaning.json` files in our dataset, which are the standardized trajectory files.

### Evaluation

**CLaTr checkpoints**

We provide the following Contrastive Language-Trajectory embedding (CLaTr) checkpoints:

| Model Type                  | Description                | Download Link |
|-----------------------------|----------------------------|---------------|
| epoch99_motion | Evaluation for Text (motion)-to-Trajectory| [Download](https://huggingface.co/Dubhe-zmc/GenDoP/blob/main/CLaTr_checkpoints/epoch99_motion.ckpt)  |
| epoch99_directorial | Evaluation for Text (directorial)-to-Trajectory  | [Download](https://huggingface.co/Dubhe-zmc/GenDoP/blob/main/CLaTr_checkpoints/epoch99_directorial.ckpt)  |

Place the downloaded files into `./evaluate/CLaTr/CLaTr_checkpoints`.


**Evaluation Commands**

- Evaluation for Our Text (motion)-to-Trajectory Results

  **Note:** Modify keys in the config file [./evaluate/CLaTr/configs/config_eval.yaml](./evaluate/CLaTr/configs/config_eval.yaml)
  - data_dir: Text (motion)-to-Trajectory Testset Results <path-to-motion-output>
  - key: 'Movement'

  ```bash
  # Extract CLaTr (motion) features
  cd ./evaluate/CLaTr
  export HYDRA_FULL_ERROR=1  
  python -m src.extraction checkpoint_path=CLaTr_checkpoints/epoch99_motion.ckpt
  # A .npy file <<path-to-motion-output>-preds.npy> containing the CLaTr (motion) features will be saved in ./evaluate/CLaTr/output

  # Evaluate CLaTr (motion)
  cd ./evaluate/eval
  python -m src.eval_only --pred_path <<path-to-motion-output>-preds.npy>
  ```

- Evaluation for Our Text (directorial)-to-Trajectory Results
  
  **Note:** Modify keys in the config file [./evaluate/CLaTr/configs/config_eval.yaml](./evaluate/CLaTr/configs/config_eval.yaml)
  - data_dir: Text (directorial)-to-Trajectory Testset Results <path-to-directorial-output>
  - key: 'Concise Interaction'

  ```bash
  # Extract CLaTr (directorial) features
  cd ./evaluate/CLaTr
  export HYDRA_FULL_ERROR=1  
  python -m src.extraction checkpoint_path=CLaTr_checkpoints/epoch99_directorial.ckpt
  # A .npy file <<path-to-directorial-output>-preds.npy> containing the CLaTr (directorial) features will be saved in ./evaluate/CLaTr/output

  # Evaluate CLaTr (directorial)
  cd ./evaluate/eval
  python -m src.eval_only --pred_path <<path-to-directorial-output>-preds.npy>
  ```

  Our paper presents four metrics: `captions/fscore`, `clatr/clatr_score`, `clatr/coverage`, and `clatr/fcd`.



## 📚 Dataset
**Note:**  We provide [DataDoP](https://huggingface.co/datasets/Dubhe-zmc/DataDoP), a large-scale multi-modal dataset containing 29K realworld shots with free-moving camera trajectories, depth maps, and detailed captions in specific movements, interaction with the scene, and directorial intent. For an overview of the DataDoP dataset and its construction, please refer to [dataset/dataset.md](./dataset/dataset.md) for more details.

*Due to copyright restrictions, some of the original data has been replaced. As a result, training on the new dataset may yield slightly different results compared to those reported in the paper.*

<!-- Currently, we are releasing a subset of the dataset for validation purposes. Additional data will be made available **coming soon**. -->

## 🏋️‍♂️ Training

**Note:**  We have released the [DataDoP](https://huggingface.co/datasets/Dubhe-zmc/DataDoP) dataset. If you wish to use your own dataset, refer to our data format and modify the [core/provider.py](./core/provider.py) file as needed. Please organize your data in the following structure. 

```
GenDoP
├── DataDoP
│   ├── train
│   ├── test
│   ├── train_valid.txt
│   ├── test_valid.txt
```

**Training Commands**  
- Text (motion)-to-trajectory
  ```bash
  accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace --exp_name 'text_motion' --cond_mode 'text' --text_key 'Movement' --num_cond_tokens 77
  ```
- Text (directorial)-to-trajectory
  ```bash
  accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace --exp_name 'text_directorial' --cond_mode 'text' --text_key 'Concise Interaction' --num_cond_tokens 77
  ```
- Text & RGBD-to-trajectory
  ```bash
  accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace --exp_name 'text_rgbd' --cond_mode 'depth+image+text' --text_key 'Concise Interaction' --num_cond_tokens 591
  ```

**Training Details**  

The model is trained on a single A100 (80GB) GPU for approximately 8 hours, with a batch size of 16, using a dataset of 30k examples for around 100 epochs.
Recommended hyperparameters:
  ```
  --discrete_bins 256 --pose_length 30 --hidden_dim 1024 --num_heads 8 --num_layers 12
  ```
You can adjust these parameters in [core/options.py](./core/options.py) according to your specific requirements.

<!-- ## 📆 Todo
- [ ] Gradio Demo -->

## 📚 Acknowledgements
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
- [EdgeRunner](https://github.com/NVlabs/EdgeRunner)
- [E.T.](https://github.com/robincourant/the-exceptional-trajectories)

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝

```bibtex
@misc{zhang2025gendopautoregressivecameratrajectory,
      title={GenDoP: Auto-regressive Camera Trajectory Generation as a Director of Photography}, 
      author={Mengchen Zhang and Tong Wu and Jing Tan and Ziwei Liu and Gordon Wetzstein and Dahua Lin},
      year={2025},
      eprint={2504.07083},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.07083}, 
}
```
