# mocca_gui/dendrogram_plotter.py

def show_dendrogram(
    Z,
    labels,
    cut_distance,
    fcn_to_color,
    bundle_to_color,
    title="Dendrogram"
):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    from scipy.cluster import hierarchy
    import matplotlib.colors as mcolors
    import numpy as np
    import re

    plt.figure(figsize=(10, 6))

    thresh = cut_distance + 0.01
    print("fcn_to_color:", fcn_to_color)
    # Convert RGBA floats → hex color strings
    def rgba_to_hex(rgba):
        hex_color = rgba[:3]
        return mcolors.to_hex(hex_color)
    fcn_palette_hex = [rgba_to_hex(c) for c in fcn_to_color.values()]
    hierarchy.set_link_color_palette(fcn_palette_hex)
    dendro = dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=thresh,
        above_threshold_color="grey"
    )
    plt.axhline(y=thresh, c='grey', lw=1, linestyle='dashed')

    leaf_order = dendro["leaves"]
    ivl = dendro["ivl"]

    # Extract bundle IDs from labels
    bundle_ids = []
    for label in ivl:
        m = re.match(r"B(\d+)", label)
        if m:
            bundle_id = int(m.group(1))
        else:
            bundle_id = None
        bundle_ids.append(bundle_id)

    # Create leaf colors dynamically
    ordered_leaf_colors = []
    for bundle_id in bundle_ids:
        if bundle_id is None:
            color = (0.5, 0.5, 0.5, 1.0)
        else:
            color = bundle_to_color.get(
                bundle_id,
                (0.5, 0.5, 0.5, 1.0)  # fallback grey
            )
        ordered_leaf_colors.append(color)

    ax = plt.gca()
    tick_labels = ax.get_xticklabels()

    for tick, color in zip(tick_labels, ordered_leaf_colors):
        tick.set_color(color)

    plt.title(title)
    plt.xlabel("Bundles")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.0001)  # Ensure the plot updates immediately
