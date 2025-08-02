# Tiny Language Models Framework

This repository contains the implementation and resources for the Tiny Language Models Framework project. This project's aim is to develop small-scale language models to facilitate detailed research into various aspects of transformer-based language models in the perspective of enveiling properties that may arise within larger language models (LLMs).

<p align="center">
  <img src="https://github.com/Modern-Compilers-Lab/Tiny-Language-Models-Framework/assets/86785811/946011ac-90ca-454f-baeb-d74b09a1721c" width="500" >
</p>

## Project Structure
- The project is structured into research projects (RPs). Each RP is self-contained in its respective top-level folder `research_project_X`.
- Additionally, a top-level `datasets` folder is available for the different research projects to share.

## 2025 Release
- This reformative edition has delivered the research projects (RP) `research_project_1` to `research_project_4`. For all these RPs, we have studied small transformer models trained to perform a custom task named TinyPy-CodeTracing, where a language model takes a python snippet as input, and produces its execution trace by duplicating the code snippet at each execution step, and annotating relevant execution information, like represented by the following figure:

<p align="center">
<img width="750" alt="tinypy_code_tracing_super_detailed" src="https://github.com/user-attachments/assets/65751d06-3185-4f20-82a0-ed28deb85862" />
</p>

- The input python snippets and their corresponding execution traces were synthetically generated using an original tool named TinyPy-CodeTracing Generator, illustrated by the figure below:
<p align="center">
<img width="750" height="2623" alt="data_gen_pipeline_5" src="https://github.com/user-attachments/assets/84259211-c271-4caf-997a-f5df6d4966d8" />
</p>

- More details can be found in the dedicated manuscript of this release.

- The top-level `tinypy_code_tracing_demo` contains a demonstration of the base research pipeline used during this release:

  - `1_data_generation/`: Contains the scripts used for generating data and preparing it for training:
 
    - `1_tinypy_generator_2.0.py`: First stage of TinyPy-CodeTracing Generator. Used to synthesize python snippets with user defined properties.
    - `2_tinypy_code_tracer.py`: Second stage of TinyPy-CodeTracing Generator. Used to create the execution trace of python snippet.
    - `3_determinism_filtering.py`: Third stage of TinyPy-CodeTracing Generator. Used to fitler our code snippets that do not meet a certain condition that can impact determinism during model inference (check manuscript for details).
    - `4_data_preparation.py`: Script to split the data into train-test-validation and tokenize the data.
    - `tinypy_code_tracer_tokenizer.py`: A custom tokenizer for our data. Used by `4_data_preparation.py`.

  - `2_model_training/train.py`: Our Pytorch training script with support for Multi-GPU data parallelism.

  - `3_inference/`:
    - `eval.py`: Script for evaluating a trained model with support for multi-inference parallelism.
    - `tinypy_code_tracer_tokenizer.py`: The same custom tokenizer used for data preparation. Required by `eval.py`.

# Contact

- **Younes Boukacem**: [yb2618@nyu.edu](mailto:yb2618@nyu.edu)
- **Hodhaifa Benouaklil**: [hb3020@nyu.edu](mailto:hb3020@nyu.edu)


# License
This project is licensed under the MIT License.

#  Acknowledgements
This work was supported in part through the NYU IT High Performance Computing resources, services, and staff expertise.
