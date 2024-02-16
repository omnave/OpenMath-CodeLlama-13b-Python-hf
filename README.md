---
license: llama2
base_model:
- codellama/CodeLlama-13b-Python-hf
datasets:
- nvidia/OpenMathInstruct-1
language:
- en
tags:
- nvidia
- code
- math
---


# OpenMath-CodeLlama-13b-Python-hf

OpenMath models were designed to solve mathematical problems by integrating text-based reasoning with code blocks
executed by Python interpreter. The models were trained on [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1),
a math instruction tuning dataset with 1.8M problem-solution pairs generated using permissively licensed
[Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) model.

<table border="1">
  <tr>
    <td></td>
    <td colspan="2" style="text-align: center;">greedy</td>
    <td colspan="2" style="text-align: center;">majority@50</td>
  </tr>
  <tr>
    <td style="text-align: center;">model</td>
    <td style="text-align: center;">GSM8K</td>
    <td style="text-align: center;">MATH</td>
    <td style="text-align: center;">GMS8K</td>
    <td style="text-align: center;">MATH</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath-CodeLlama-7B (<a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-7b-Python">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-7b-Python-hf">HF</a>)</td>
    <td style="text-align: center;">75.9</td>
    <td style="text-align: center;">43.6</td>
    <td style="text-align: center;">84.8</td>
    <td style="text-align: center;">55.6</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath-Mistral-7B (<a href="https://huggingface.co/nvidia/OpenMath-Mistral-7B-v0.1">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath-Mistral-7B-v0.1-hf">HF</a>)</td>
    <td style="text-align: center;">80.2</td>
    <td style="text-align: center;">44.5</td>
    <td style="text-align: center;">86.9</td>
    <td style="text-align: center;">57.2</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath-CodeLlama-13B (<a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-13b-Python">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-13b-Python-hf">HF</a>)</td>
    <td style="text-align: center;">78.8</td>
    <td style="text-align: center;">45.5</td>
    <td style="text-align: center;">86.8</td>
    <td style="text-align: center;">57.6</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath-CodeLlama-34B (<a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-34b-Python">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-34b-Python-hf">HF</a>)</td>
    <td style="text-align: center;">80.7</td>
    <td style="text-align: center;">48.3</td>
    <td style="text-align: center;">88.0</td>
    <td style="text-align: center;">60.2</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath-Llama2-70B (<a href="https://huggingface.co/nvidia/OpenMath-Llama-2-70b">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath-Llama-2-70b-hf">HF</a>)</td>
    <td style="text-align: center;"><b>84.7</b></td>
    <td style="text-align: center;">46.3</td>
    <td style="text-align: center;">90.1</td>
    <td style="text-align: center;">58.3</td>
  </tr>
  <tr>
    <td style="text-align: right;">OpenMath-CodeLlama-70B (<a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-70b-Python">nemo</a> | <a href="https://huggingface.co/nvidia/OpenMath-CodeLlama-70b-Python-hf">HF</a>)</td>
    <td style="text-align: center;">84.6</td>
    <td style="text-align: center;"><b>50.7</b></td>
    <td style="text-align: center;"><b>90.8</b></td>
    <td style="text-align: center;"><b>60.4</b></td>
  </tr>
</table>

The pipeline we used to produce these models is fully open-sourced!

- [Code](https://github.com/Kipok/NeMo-Skills)
- [Models](https://huggingface.co/collections/nvidia/openmath-65c5619de2ba059be0775014)
- [Dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)

See our [paper](https://arxiv.org/abs/2402.10176) for more details!

# How to use the models?

Try to [run inference with our models](https://github.com/Kipok/NeMo-Skills/blob/main/docs/inference.md) with just a few commands!

# Reproducing our results

We provide [all instructions](https://github.com/Kipok/NeMo-Skills/blob/main/docs/reproducing-results.md) to fully reproduce our results.

# Improving other models

To improve other models or to learn more about our code, read through the docs below.

- [NeMo-Skills Pipeline](https://github.com/Kipok/NeMo-Skills)
    - [Generating synthetic data](https://github.com/Kipok/NeMo-Skills/blob/main/docs/synthetic-data-generation.md)
    - [Finetuning models](https://github.com/Kipok/NeMo-Skills/blob/main/docs/finetuning.md)
    - [Evaluating models](https://github.com/Kipok/NeMo-Skills/blob/main/docs/evaluation.md)

In our pipeline we use [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/),
an end-to-end, cloud-native framework to build, customize, and deploy generative AI models anywhere.
It includes training and inferencing frameworks, guardrailing toolkits, data curation tools, and pretrained models,
offering enterprises an easy, cost-effective, and fast way to adopt generative AI.

# Citation

If you find our work useful, please consider citing us!

```bibtex
@article{toshniwal2024openmath,
  title   = {OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset},
  author  = {Shubham Toshniwal and Ivan Moshkov and Sean Narenthiran and Daria Gitman and Fei Jia and Igor Gitman},
  year    = {2024},
  journal = {arXiv preprint arXiv: Arxiv-2402.10176}
}
```

# License

The use of this model is governed by the [Llama 2 Community License Agreement](https://ai.meta.com/llama/license/)