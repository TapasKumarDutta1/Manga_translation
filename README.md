# Manga Translation using DETR, Trocr, and Transformer

This repository contains a project focused on translating manga content from Japanese to English. The translation process involves several steps: detection, recognition, and translation. The system utilizes state-of-the-art models and datasets for manga analysis and translation.

## Overview

The project integrates the following components:

### 1. DETR (DEtection TRansformer)

- **Objective**: Object detection within manga pages.
- **Dataset**: Trained on the Manga109 dataset.
- **Description**: DETR is employed for object detection tasks, identifying and localizing different elements within manga pages.

### 2. Trocr (TRansformer for Optical Character Recognition)

- **Objective**: Text recognition on manga images.
- **Dataset**: Extracted samples from the Manga109 dataset.
- **Description**: Trocr is responsible for recognizing text present within manga images, enabling the extraction of textual content for translation.

### 3. Transformer Architecture for Translation

- **Objective**: Japanese to English translation.
- **Dataset**: Japanese English Subtittle Corpus.
- **Description**: A Transformer-based architecture is utilized for translating the Japanese text extracted from manga images to English. This system ensures accurate and contextually relevant translations.

## How to Use

1. Clone the repository.
2. Install requirements using   `pip install -r requirements.txt`
3. Follow the colab example. `working_example.ipynb`


## Citation

```markdown
@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European conference on computer vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}

@misc{li2022trocr,
      title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models}, 
      author={Minghao Li and Tengchao Lv and Jingye Chen and Lei Cui and Yijuan Lu and Dinei Florencio and Cha Zhang and Zhoujun Li and Furu Wei},
      year={2022},
      eprint={2109.10282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{vaswani2023attention,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
