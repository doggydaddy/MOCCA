# mocca_gui/dendrogram_plotter.py

def show_dendrogram(
    Z,
    labels,
    cut_distance,
    fcn_to_color,
    bundle_to_color,
    title="Dendrogram",
    truncate_mode=None,
    p=None,
):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.colors as mcolors
    import numpy as np
    import re

    plt.figure(figsize=(10, 6))

    thresh = cut_distance + 0.01

    # Convert RGBA/RGB floats -> hex color string
    def rgba_to_hex(color):
        return mcolors.to_hex(color[:3])

    # Parse bundle IDs from labels; labels are generated as "B{bundle} (FCN{...})"
    leaf_bundle_ids = []
    for label in labels:
        m = re.match(r"B(\d+)", str(label))
        leaf_bundle_ids.append(int(m.group(1)) if m else None)

    default_hex = rgba_to_hex((0.5, 0.5, 0.5, 1.0))
    leaf_hex_colors = [
        rgba_to_hex(bundle_to_color.get(bid, (0.5, 0.5, 0.5, 1.0)))
        if bid is not None else default_hex
        for bid in leaf_bundle_ids
    ]

    # Build descendants' color sets for each internal node (id: n..2n-2)
    n_leaves = len(labels)
    node_color_sets = {}
    node_heights = {}

    for i, row in enumerate(Z):
        node_id = n_leaves + i
        left = int(row[0])
        right = int(row[1])
        dist = float(row[2])

        if left < n_leaves:
            left_colors = {leaf_hex_colors[left]}
        else:
            left_colors = node_color_sets.get(left, {default_hex})

        if right < n_leaves:
            right_colors = {leaf_hex_colors[right]}
        else:
            right_colors = node_color_sets.get(right, {default_hex})

        node_color_sets[node_id] = left_colors | right_colors
        node_heights[node_id] = dist

    def link_color_func(k):
        """
        Color links by bundle colors:
        - above threshold: grey
        - below threshold and pure-color subtree: that color
        - below threshold but mixed descendant colors: grey
        """
        if k < n_leaves:
            return leaf_hex_colors[k]
        if node_heights.get(k, np.inf) > thresh:
            return "grey"
        color_set = node_color_sets.get(k, {default_hex})
        if len(color_set) == 1:
            return next(iter(color_set))
        return "grey"

    dendro_kwargs = dict(
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        link_color_func=link_color_func,
        color_threshold=thresh,
        above_threshold_color="grey",
    )
    if truncate_mode is not None:
        dendro_kwargs["truncate_mode"] = truncate_mode
    if p is not None:
        dendro_kwargs["p"] = p
    dendro = dendrogram(Z, **dendro_kwargs)
    plt.axhline(y=thresh, c='grey', lw=1, linestyle='dashed')

    leaf_order = dendro["leaves"]

    # Reorder leaf colors to plotted order and color x tick labels.
    # Under truncation (truncate_mode set), scipy's "leaves" can be internal
    # node ids standing in for a collapsed subtree (id >= n_leaves), not just
    # original leaf indices -- reuse link_color_func for those, exactly the
    # same rule already used to color the links themselves, instead of
    # indexing leaf_hex_colors (which is only valid for id < n_leaves).
    def leaf_display_color(idx):
        return leaf_hex_colors[idx] if idx < n_leaves else link_color_func(idx)

    ordered_leaf_colors = [leaf_display_color(idx) for idx in leaf_order]

    ax = plt.gca()
    tick_labels = ax.get_xticklabels()

    for tick, color in zip(tick_labels, ordered_leaf_colors):
        tick.set_color(color)

    # A leaf whose bundle/network has only one member never gets a colored
    # link: its only merge is, by construction, the one where it joins a
    # DIFFERENT cluster above the cut line (that's what makes it a
    # singleton), so link_color_func legitimately greys that merge out --
    # there is no below-threshold, single-color segment belonging to it for
    # scipy to draw. Mark each leaf's own color at its base explicitly so a
    # singleton bundle/network is still visible even though its connecting
    # line is grey.
    tick_positions = [tick.get_position()[0] for tick in tick_labels]
    ax.scatter(
        tick_positions, [0] * len(tick_positions),
        color=ordered_leaf_colors, s=20, zorder=5, clip_on=False,
    )

    plt.title(title)
    plt.xlabel("Bundles")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.0001)  # Ensure the plot updates immediately
