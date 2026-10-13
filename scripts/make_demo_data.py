"""Create the synthetic demo dataset in data/demo/.

The scene mimics a mountain landscape with forest, shrubland, dry grassland,
bare ground and lakes, a fire with a severity gradient, and summer drying of
grasslands. Values are synthetic and only meant to run the pipeline.
"""

import pandas as pd

from burnmap.demo import PARAMS, write_demo_files

if __name__ == "__main__":
    features_path, samples_path = write_demo_files(PARAMS)
    samples = pd.read_csv(samples_path)
    print(f"make_demo_data: {features_path} ({PARAMS['shape'][0]}x{PARAMS['shape'][1]} px), "
          f"{len(samples)} training pixels in {samples['poly_id'].nunique()} polygons")
