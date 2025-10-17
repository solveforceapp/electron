# Λ∞ · Logos Infinity Animation

The `script/icositetrahedral_logos_infinity.py` helper renders a rotating
24-faced tetrakis hexahedron (also called an icositetrahedron) and exports both
MP4 and GIF versions of the motion graphic.

## Requirements

Install the Python dependencies before running the script:

```bash
pip install matplotlib numpy
```

Matplotlib's animation writers require additional system packages if you want to
produce MP4 output. On macOS or Linux, ensure that FFmpeg is available in your
`PATH`.

## Usage

```bash
python script/icositetrahedral_logos_infinity.py
```

Two files are generated in the current directory:

- `Icositetrahedral_Logos_Infinity_Rotation.mp4`
- `Icositetrahedral_Logos_Infinity_Rotation.gif`

You can customise the render destination and frame settings by importing and
calling `create_animation` directly:

```python
from pathlib import Path
from script.icositetrahedral_logos_infinity import create_animation

create_animation(output_dir=Path("./renders"), frames=range(0, 360), fps_mp4=30)
```
