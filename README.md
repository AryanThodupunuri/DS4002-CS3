# Extreme Weather Image Classification Case Study

## Project Overview

This repository contains the full materials for the CS3 Case Study, designed to facilitate the replication and analysis of our original Extreme Weather Image Classification project by a rising 2nd Year Data Science student.

The project addresses the need for automated, high-speed detection of severe weather events (Tropical Cyclones, Wildfires, Dust Storms) from satellite imagery using Convolutional Neural Networks (CNNs). The primary goal is to achieve reliable classification performance (F1-Score $\geq 0.90$) using Transfer Learning techniques.

### Repository Structure

| Folder / File | Content | Purpose |
| :--- | :--- | :--- |
| **`scripts/`** | Contains the main Python scripts/Jupyter Notebooks for data processing, model training, and evaluation. | Student's primary workspace for running the pipeline. |
| **`Supplemental/`** | Contains all external resources, guides, and documentation necessary for the project. **(See details below)** | Provides technical foundation, setup instructions, and hard-copy printouts. |
| `Hook_Document.pdf` | The one-page mission brief targeted at the 2nd year student. | Motivation and high-level project framing. |
| `Student_Rubric.pdf` | The full specifications and assessment criteria for the student task. | Detailed task guidance and self-assessment checklist. |

### Supplemental Folder Contents

The `Supplemental/` folder contains the following specific resources:

* **Setup/Execution Guide:** The **Setup Document** (`getting_started_guide.md`) which explains the environment setup, data acquisition from the Harvard Dataverse, and step-by-step execution of the primary scripts.
* **Core Technical Documentation:** Official **MobileNetV2 PyTorch Documentation** and a relevant **Transfer Learning with PyTorch** tutorial/blog post. These resources are critical for the student to understand the implementation details of the model required by the rubric.
* **Project Context:** A copy of the original **Slide Deck** (`Extreme Weather Image Classification (4).pdf`) used for the initial project presentation.

## Final Reflection Summary 

### 1. Choice of Transfer Learning Model

For this project, we chose **MobileNetV2** (or **EfficientNetV2-S**, depending on which one was your best performer) as our primary Transfer Learning architecture. This decision was largely driven by the goal of balancing performance with deployability. We recognized that using a model pre-trained on ImageNet would give us a massive head start on feature extraction—meaning the model already knew what edges and textures looked like.

We essentially froze the base convolutional layers and trained only the final classification head on our specialized weather data. This process proved extremely effective, as the Transfer Learning models immediately outperformed our simple Baseline CNN. The final metrics confirmed the success of this approach, allowing us to surpass the F1-score target of 0.88 with ease. 


### 2. The Most Difficult Analytical Trade-off

The trickiest analytical trade-off was definitely the **Class Collapsing**—moving from five distinct original weather categories down to just two: "Extreme" and "Normal." While computationally easier, this labeling choice required us to make judgment calls on ambiguous categories. For example, where do convective cell clouds fall? Are they always "Normal," or could their structure suggest potential severity? This decision directly impacted the meaning of our final "Extreme" classification and risked grouping visually distinct phenomena (like dust storms and hurricanes) under one label, potentially confusing the model. We had to ensure the benefits of clear binary output for emergency response outweighed the loss of detailed meteorological information.

### 3. Comparison to Original Hypothesis

The original hypothesis stated that Transfer Learning models would outperform the Baseline CNN and exceed **90% accuracy / 0.88 F1-score**. Our results confirmed this hypothesis strongly. Our best Transfer Learning model achieved a macro F1-score of **0.99** on the test set, with near-perfect precision and recall across both the Extreme and Normal classes. This dramatic performance boost demonstrated the overwhelming advantage of using pre-trained weights for specialized image classification tasks, validating the entire research approach.