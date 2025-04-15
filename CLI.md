## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
# With a RAW file
python film_print_simulator.py input.dng output.tif

# With a JPEG file
python film_print_simulator.py input.jpg output.tif
```

Adjusting color balance with CMY brightness values:

```bash
python film_print_simulator.py input.tif output.tif --cyan 1.2 --magenta 0.9 --yellow 1.1
```
