# Old Photo Restoration

### Based on [Microsoft repository](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)

## Installation


Download pretrained models

```
python get_models.py
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to use?

### Full Pipeline

You could easily restore the old photos with one simple command after installation and downloading the pretrained model.

For images without scratches:

```
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
```

For scratched images:

```
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
              --with_scratch
```
