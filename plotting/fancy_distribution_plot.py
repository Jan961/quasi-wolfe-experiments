import matplotlib.pyplot as plt
import numpy as np


def fancy_distribution_plot(distributions: list, tick_labels: list, max_plot_width: int = 1, alpha=0.7,
                            number_of_segments=12,
                            separation_between_plots=0.1,
                            separation_between_subplots=0.1,
                            vertical_limits=None,
                            grid=False,
                            remove_outlier_above_segment=None,
                            remove_outlier_below_segment=None,
                            y_label=None,
                            title=None,
                            ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    number_of_plots = len(distributions)
    # print(f" number of plots {number_of_plots}")
    # print(f" max x line {number_of_plots * (max_plot_wwidth + separation_between_plots) + separation_between_plots}")

    ax.set_xlim(left=0, right=number_of_plots * (max_plot_width + separation_between_plots) + separation_between_plots)

    ticks = [separation_between_plots + max_plot_width / 2 + (max_plot_width + separation_between_plots) * i
             for i in range(0, number_of_plots)]
    # print(ticks)

    for i in range(len(distributions)):
        distribution = distributions[i]
        # print(f"distribution {distribution}")
        segments = np.linspace(np.min(distribution), np.max(distribution), number_of_segments + 1)[1:-1]
        # print(f"segments {segments}")
        segment_indices = number_of_segments - 1 - np.where(segments[:, None] >= distribution[None, :], 1, 0).sum(0)
        # print(f"quantile indices {segment_indices}")
        if remove_outlier_above_segment:
            a = remove_outlier_above_segment[i]
            distribution = distribution[segment_indices <= a]
            segment_indices = segment_indices[segment_indices <= a]

        if remove_outlier_below_segment:
            b = remove_outlier_below_segment[i]
            distribution = distribution[segment_indices >= b - 1]
            segment_indices = segment_indices[segment_indices >= b - 1]

        values, counts = np.unique(segment_indices, return_counts=True)
        # print(f"values {values}")
        # print(f"counts {counts}")
        counts_filled = []
        j = 0
        for k in range(number_of_segments):
            if k in values:
                counts_filled.append(counts[j])
                j += 1
            else:
                counts_filled.append(0)
        variances = (max_plot_width / 2) * (counts_filled / np.max(counts))
        # print(f"variances {variances}")
        jitter_unadjusted = np.random.uniform(-1, 1, len(distribution))
        jitter = np.take(variances, segment_indices) * jitter_unadjusted
        # print(f"jitter {jitter}")
        ax.scatter(jitter + ticks[i], distribution, alpha=alpha)

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    if vertical_limits:
        ax.set_ylim(bottom=vertical_limits[0], top=vertical_limits[1])
    if not grid:
        ax.grid(False)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    plt.show()
