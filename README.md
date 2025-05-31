# Multiverse Through Deepfakes: The MultiFakeVerse Dataset of Person-Centric Visual and Conceptual Manipulations
A pipeline for automatic and semantic context-relevant image tampering.

Dataset available on [Huggingface](https://huggingface.co/datasets/parulgupta/MultiFakeVerse)
Preview Version: [Link](https://huggingface.co/datasets/parulgupta/MultiFakeVerse_preview)

## Abstract
The rapid advancement of GenAI technology over the past few years has significantly contributed towards highly realistic deepfake content generation. Despite ongoing efforts, the research community still lacks a large-scale and reasoning capability driven deepfake benchmark dataset specifically tailored for person-centric object, context and scene manipulations. In this paper, we address this gap by introducing MultiFakeVerse, a large scale person-centric deepfake dataset, comprising 845,286 images generated through manipulation suggestions and image manipulations both derived from vision-language models (VLM). The VLM instructions were specifically targeted towards modifications to individuals or contextual elements of a scene that influence human perception of importance, intent, or narrative. This VLM-driven approach enables semantic, context-aware alterations such as modifying actions, scenes, and human-object interactions rather than synthetic or low-level identity swaps and region-specific edits that are common in existing datasets. Our experiments reveal that current state-of-the-art deepfake detection models and human observers struggle to detect these subtle yet meaningful manipulations.
![Image](https://github.com/user-attachments/assets/5aab4d7a-7342-4fa1-ab6c-13fa044daccb)
MultiFakeVerse. A brief overview of the proposed dataset. Here, we introduce subtle and profound person-centric deepfakes covering \textit{person-level}, \textit{object-level}, \textit{scene-level}, \textit{(person+object)-level}, \textit{(person+scene)-level} manipulations. Image best viewed in color.
![Image](https://github.com/user-attachments/assets/4cc6514e-8fd6-4b19-8005-c1d7f84a5ff9)
![Image](https://github.com/user-attachments/assets/55000bbb-16d6-46c3-99a8-b02cd885423e)
![Image](https://github.com/user-attachments/assets/35f59717-eb3e-49e6-82c5-a656014a0a7b)
Some more example images and their fakes obtained using VLM based image editing, The highlighted yellow boxes indicate the edited regions for images with localized edits. The rest of the images have all three kinds of modifications: Person-level, Object-level and Scene-level.
