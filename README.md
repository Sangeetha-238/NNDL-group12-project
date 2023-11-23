# Art Generation from Text Descriptions

## Project Overview

This project aims to generate visual art from textual descriptions using a combination of Generative Adversarial Networks (GANs) and Natural Language Processing (NLP). It leverages the WikiArt dataset for art styles and the Flickr30k dataset for text-based image descriptions.

## Datasets

- WikiArt Dataset: A comprehensive collection of artworks categorized by styles, artists, and genres.

- Flickr30k Dataset: Contains 30,000 images with five captions each, providing a diverse set of image descriptions.

## Usage

1. Clone the repository

```bash
git clone https://github.com/Sangeetha-238/NNDL-group12-project.git
```

2. Install the required packages

```bash
pip install -r requirements.txt
```

3. Download the datasets

```bash
bash download_data.sh
```

4. Resize the WikiArt Dataset

```bash
cd code
python3 WikiArt_Data_Processing.py --data_dir "../data/WikiArt" --target_size 128 128 --csv_output_dir "../data/wikiart_image_data.csv"
```