import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# Sizes for hexgons
HEX_SIZE = 1.0
vert = HEX_SIZE * np.sqrt(3)

# Define the positions for a standard Catan board layout with 0 at the top
# The positions are in (x, y) coordinates with (0,0) at the center
standard_positions = {
    0: (0, -2 * vert),
    1: (-1.5 * HEX_SIZE, -1.5 * vert),
    2: (1.5 * HEX_SIZE, -1.5 * vert),
    3: (-3 * HEX_SIZE, -vert),
    4: (0, -vert),
    5: (3 * HEX_SIZE, -vert),
    6: (-1.5 * HEX_SIZE, -0.5 * vert),
    7: (1.5 * HEX_SIZE, -0.5 * vert),
    8: (-3 * HEX_SIZE, 0),
    9: (0, 0),
    10: (3 * HEX_SIZE, 0),
    11: (-1.5 * HEX_SIZE, 0.5 * vert),
    12: (1.5 * HEX_SIZE, 0.5 * vert),
    13: (-3 * HEX_SIZE, vert),
    14: (0, vert),
    15: (3 * HEX_SIZE, vert),
    16: (-1.5 * HEX_SIZE, 1.5 * vert),
    17: (1.5 * HEX_SIZE, 1.5 * vert),
    18: (0, 2 * vert),
}

# Ideal adjacency map
adjacency_map = {
    0: [1, 2],
    1: [0, 3, 4, 6],
    2: [0, 4, 5, 7],
    3: [1, 6, 8],
    4: [1, 2, 6, 7, 9],
    5: [2, 7, 10],
    6: [1, 3, 4, 8, 9, 11],
    7: [2, 4, 5, 9, 10, 12],
    8: [3, 6, 11, 13],
    9: [4, 6, 7, 11, 12, 14],
    10: [5, 7, 12, 15],
    11: [6, 8, 9, 13, 14, 16],
    12: [7, 9, 10, 14, 15, 17],
    13: [8, 11, 16],
    14: [9, 11, 12, 16, 17, 18],
    15: [10, 12, 17],
    16: [11, 13, 14, 18],
    17: [12, 14, 15, 18],
    18: [14, 16, 17],
}


def visualize_board(board):
    """
    Visualize the Catan board with resource types and numbers.
    Args:
        board: Dictionary of the board, containing resource distribution and hex_data
            (a mapping hex_id to hex_data - type, number, position)
    """
    # Guard
    if not board or "hexagons" not in board:
        print("Error: Invalid board structure for visualization")
        return
    # Create a figure
    _, ax = plt.subplots(figsize=(10, 8))
    # Colors for different resource types
    resource_colors = {
        "lumber": "forestgreen",
        "brick": "firebrick",
        "sheep": "lightgreen",
        "wheat": "gold",
        "ore": "dimgray",
        "desert": "khaki",
    }
    # Draw each hex
    for hex_id, hex_data in board["hexagons"].items():
        x, y = hex_data["position"]
        vertices = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = np.pi / 180 * angle_deg
            dx = HEX_SIZE * np.cos(angle_rad)
            dy = HEX_SIZE * np.sin(angle_rad)
            vertices.append((x + dx, y + dy))
        # Create patch
        resource_type = hex_data["type"]
        color = resource_colors.get(resource_type, "white")
        hex_patch = patches.PathPatch(
            Path(vertices), facecolor=color, edgecolor="black", linewidth=2, alpha=0.7
        )
        ax.add_patch(hex_patch)
        # Add number label
        num = hex_data["number"]
        if num:
            # (6, 8) red and bold
            ax.text(
                x,
                y,
                str(num),
                ha="center",
                va="center",
                fontsize=12,
                color="red" if num in [6, 8] else "black",
                weight="bold" if num in [6, 8] else "normal",
            )
        # Add the resource type label below the number
        ax.text(x, y - 0.3, resource_type, ha="center", va="center", fontsize=8)
        # Add hex id
        ax.text(
            x,
            y + 0.3,
            f"ID: {hex_id}",
            ha="center",
            va="center",
            fontsize=8,
            color="blue",
        )
        # Without adjacency lines the plot breaks?
        # Plot them very thinly just to preserve structure
        for adj_id in hex_data["adjacents"]:
            if adj_id in board["hexagons"]:
                adj_x, adj_y = board["hexagons"][adj_id]["position"]
                # Draw a thin line connecting adjacent hexagons
                ax.plot([x, adj_x], [y, adj_y], "k-", linewidth=0.1, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_title("Catan Board Visualizer")
    ax.set_xticks([])
    ax.set_yticks([])
    # Invert y-axis to have 0 at the top
    ax.invert_yaxis()
    # Add a legend
    legend_elements = [
        patches.Patch(facecolor=color, edgecolor="black", label=r_type)
        for r_type, color in resource_colors.items()
        if r_type in board["resource_distribution"]
    ]
    ax.legend(
        handles=legend_elements,
        title="Resource Types",
        loc="upper right",
        bbox_to_anchor=(1.1, 1),
    )
    # Add resource distribution information
    if "resource_distribution" in board:
        resource_text = "Resource Distribution:\n"
        for resource, count in board["resource_distribution"].items():
            resource_text += f"{resource}: {count}\n"
        plt.figtext(
            0.02,
            0.02,
            resource_text,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8),
        )
    plt.tight_layout()
    plt.show()
