# EventGPT: Event Stream Understanding with Multimodal Large Language Models
<div align="center">
  <img src="assets/overview.png" width="90%">
</div>

# Contents

- [Installation](#Installation)

- [Datasets and Models](#Datasets and Models)
- [Run](#Run)
- [Acknowledgement](#Acknowledgement)

# Installation

1. Clone this repository.

```bash
git clone https://github.com/XduSyL/EventGPT.git && cd EventGPT/
```



2. Install the required dependencies.

```bash
conda create -n eventgpt python=3.10 -y
conda activate eventgpt
pip install -e .
```



3. Download Vicuna checkpoints

Our project is built on **Vicuna v1.5**, an instruction-tuned chatbot optimized for diverse conversational tasks. Vicuna v1.5 is known for its robust transformer-based architecture and fine-tuned capabilities, making it highly effective in understanding and generating context-aware responses.



**Download the Pre-trained Model**

You can download the base model and other related resources from the following links:

**Model Weights**: [Vicuna v1.5 (Hugging Face)](https://huggingface.co/...)



4. Download CLIP checkpoints

We also integrate the **CLIP ViT-L/14-336** model, a vision-language model designed for efficient multimodal learning. This model enables seamless feature extraction from visual inputs and aligns them with textual representations.



**Download the Pre-trained Model**

**Model Weights**: [CLIP ViT-L/14-336 (Hugging Face)](https://huggingface.co/openai/clip-vit-large-patch14-336)



# Datasets and Models

The **EventGPT** model, along with the **N-ImageNet-Chat** and **Event-Chat** datasets, will be released after the acceptance of our paper.



# Run

1. Stage One: Image-Language Alignment

Train the image-language alignment module for EventGPT using the following command:

```bash
sh script/deepspeed_stage1.sh
```

​	•	--model_name_or_path: Path to the **Vicuna-v1.5** model. Replace this with your local path to the Vicuna checkpoint. [Vicuna v1.5 (Hugging Face)](https://huggingface.co/...)

​	•	--data_path: Path to the **LLaVA-Pretrain** dataset JSON file. This should point to the JSON file containing the pretraining data. [LLaVA-Pretrain datasets (Hugging Face)](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)

​	•	--image_folder: Path to the image folder containing all images referenced in the dataset. 

​	•	--vision_tower: Path to the CLIP model checkpoint**. Replace this with the local path to your pretrained CLIP model. [CLIP ViT-L/14-336 (Hugging Face)](https://huggingface.co/openai/clip-vit-large-patch14-336)

​	•	--output_dir: Directory to save the model checkpoints** after training. Ensure sufficient storage is available. 



2. Stage Two: Event-Language Alignment

Train the event-language alignment module for EventGPT using the following command:

```bash
sh script/deepspeed_stage2.sh
```

--data_path: Path to the **N-ImageNet-Chat** dataset JSON file. This should point to the JSON file containing the pretraining data.

 --pretrain_mm_mlp_adapter: Path to the adapter output from Stage One. Replace this with the local path to the adapter file trained in Stage One, which will be loaded for further training.

--event_folder: Path to the event stream folder containing all event stream(.npy) referenced in the dataset. 

--output_mm_mlp_adapter: Path to save the adapter output for this stage. The adapter trained in this stage will be saved to the specified location.



3. Stage Three: Instruction Tuning

In this stage, we perform instruction tuning on the entire parameter set of EventGPT using the **Event-Chat dataset**. The objective is to enable EventGPT to effectively follow complex, context-aware instructions grounded in multimodal inputs.

```bash
sh script/deepspeed_stage3.sh
```

--data_path: Path to the **Event-Chat** dataset JSON file. This should point to the JSON file containing the pretraining data.

--pretrain_mm_mlp_adapter: Path to the adapter output from Stage One. Replace this with the local path to the adapter file trained in Stage One, which will be loaded for further training.

--pretrain_feature_adaptor: Path to the adapter output from Stage Two. Replace this with the local path to the adapter file trained in Stage Two



# Acknowledgement

- [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-7B that has the amazing language capabilities!
- [CLIP](https://github.com/openai/CLIP): A powerful vision-language model from OpenAI, which we utilized for aligning visual and textual features in our EventGPT framework. Its robust capabilities greatly enhance multimodal understanding.



