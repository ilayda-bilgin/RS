import numpy as np
import m_phate
import scprep
import pickle
from collections import Counter
from bokeh.plotting import figure, output_file, save
from bokeh.models import Label, LabelSet, ColumnDataSource, Circle, ColorBar, HoverTool
from bokeh.io import output_notebook, show
import bokeh.palettes as bp
from bokeh.transform import linear_cmap


def generate_m_phate(data):
    # embedding
    m_phate_op = m_phate.M_PHATE(n_jobs=-16)
    m_phate_data = m_phate_op.fit_transform(data)
    print(f"m-PHATE done")
    return m_phate_data


def create_phate_visualization(data, m_phate_data, filename="phate-param.png"):
    shape_data = data.shape
    time = np.repeat(np.arange(shape_data[0]), shape_data[1])
    print(f"Plotting ...")
    plot = scprep.plot.scatter2d(
        m_phate_data,
        c=time,
        ticks=False,
        label_prefix="M-PHATE",
        filename=filename,
        dpi=600,
        title="Learnable Weigths per Time Step",
    )


def main():
    data = np.load("params_per_batch.npy")
    print(f"Data shape: {data.shape}")  # n_time_steps, n_points, n_dim
    m_phate_data = generate_m_phate(data)
    create_phate_visualization(data, m_phate_data, filename="phate-param.png")


if __name__ == "__main__":
    main()
