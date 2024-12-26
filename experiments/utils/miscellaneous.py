import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import gmean
from scipy.ndimage import uniform_filter1d
from scipy import stats
from scipy.interpolate import CubicSpline, PPoly, splrep, PchipInterpolator


# extracts distances from the first non-zero value to the last non-zero value
def calculate_I1(y_distances: np.ndarray, discard_shorter_than: int = 4):
    # extract distances
    y_distances, _ = extract_distances(y_distances, discard_shorter_than)
    if y_distances is None:
        return np.nan

    x = np.log(y_distances[:-1])
    y = np.log(y_distances[1:])
    return stats.linregress(x, y).slope


# if is_single_run the distances will be extracted and it is assumed that each distance corresponds to one step
def calculate_I2(input_errors: np.ndarray, discard_shorter_than: int = 5,
                 is_single_run=True, input_step_counts: np.ndarray = None,
                 get_max_area=False):
    if is_single_run:
        errors, step_counts = extract_distances(input_errors, discard_shorter_than)
        if errors is None:
            return np.nan

    else:
        assert input_step_counts is not None

        step_counts, errors = filter_counts(input_step_counts, input_errors)

    # remove error values above 1e-10 to avoid round-off errors
    first_small = np.argmax(errors <= 1e-10)
    # print(f" first_small auto {first_small}")

    if first_small == 0 and errors[0] > 1e-10:
        filtered_errors = errors
        filtered_counts = step_counts
    else:
        filtered_errors = errors[:first_small]
        filtered_counts = step_counts[:first_small]

    if len(filtered_errors) < discard_shorter_than:
        return np.nan
    # print(f" len(filtered_errors) auto {len(filtered_errors)}")

    # first fit a spline to the not-rotated data, sample many points and fit another spline after rotation to prevent
    # deformations
    # but not too many otherwise there will be too many roots

    forget_log_x = np.log(1 / filtered_errors)
    translated_x = forget_log_x - forget_log_x[0]
    # translate y values (counts as well)
    filtered_counts = filtered_counts - filtered_counts[0]
    og_points = np.array([translated_x, filtered_counts]).T

    spline1 = PchipInterpolator(og_points[:, 0], og_points[:, 1])
    sample_x = np.linspace(translated_x[0], translated_x[-1], 200)
    sample_y = spline1(sample_x)
    sample_points = np.array([sample_x, sample_y]).T

    length_x = translated_x[-1] - translated_x[0]
    length_y = filtered_counts[-1] - filtered_counts[0]
    angle = -np.arctan(length_y / length_x)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_points = np.array([rotation_matrix @ point for point in sample_points])
    # print(f" rotated_points auto {rotated_points}")

    # try to reduce the impact of spline distortions when few data points are available
    rotated_points = rotated_points[np.argsort(rotated_points[:, 0])]

    spline2 = PchipInterpolator(rotated_points[:, 0], rotated_points[:, 1])
    roots = spline2.roots(extrapolate=False)
    to_insert1 = None
    if len(roots) == 0:
        all_roots = np.array([0, rotated_points[-1, 0]])
    elif not np.isclose(roots[-1], rotated_points[-1, 0]):
        to_insert1 = rotated_points[-1, 0]
        all_roots = np.append(roots, rotated_points[-1, 0])
    else:
        all_roots = roots
    all_roots[0] = 0
    all_roots = np.clip(all_roots, 0, rotated_points[-1, 0])

    # print(f" all_roots auto {all_roots}")
    triangle_area = abs(0.5 * length_x * length_y)
    areas = np.array(
        [spline2.integrate(all_roots[i], all_roots[i + 1]) for i in range(len(all_roots) - 1)])

    out = np.sum(areas) / triangle_area
    if out >= 1 or out <= -1:
        print(f" out {out}")
        print(f" somethings wrong ")
        print(f" roots {all_roots}")
        if to_insert1 is not None:
            print(f" inserted {to_insert1}")
        print(f" areas {areas}")
        print(f" length_x {length_x}")
        print(f" length_y {length_y}")
        print(f" triangle_area {triangle_area}")

    if get_max_area:
        np.max(np.abs(areas)) / triangle_area
    else:
        return np.sum(areas) / triangle_area


def filter_counts(counts ,epsilons):
    first_non_zero = np.argmax(counts > 0)
    idx_maximum = np.argmax(counts)
    if idx_maximum + 1 - first_non_zero <= 0:
        return None, None
    else:
        return counts[first_non_zero:idx_maximum + 1], epsilons[first_non_zero: idx_maximum+1]


def extract_distances(y_distances: np.ndarray, discard_shorter_than):
    first_non_zero = np.argmax(y_distances[::-1] > 0)
    if first_non_zero > 0 or y_distances[0] == 0:
        y_distances = y_distances[:-first_non_zero]
    if len(y_distances) < discard_shorter_than:
        return None, None

    # remove the all values until the sequence is strict increasing
    a = y_distances
    max_id = np.argmax(np.diff(a[::-1]) <= 0)
    if max_id == 0 and a[-2] > a[-1]:
        out = a
    else:
        out = a[-max_id - 1:]
    if len(out) < discard_shorter_than:
        return None, None

    return out, np.arange(len(out))


# changed to remove error values above 1e-10 to avoid round-off errors
# def filter_step_counts(step_counts: np.ndarray):

# first_non_zero = np.argmax(step_counts == 1)
#
# if first_non_zero == 0 and step_counts[0] == 0:
#     return None , None, None
#
# idx_maximum = np.argmax(step_counts)
#
# if idx_maximum + 1 - first_non_zero <= 0:
#     return None, None, None
# else:
#     return step_counts[first_non_zero:idx_maximum + 1], first_non_zero, idx_maximum


def geometric_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())


def smooth_running_mean(input: np.ndarray, window_size: int, geometric=False, cut_right_edge=True):
    if geometric:
        padded = np.pad(input, window_size - 1, mode="constant", constant_values=np.nan)
        windowed = sliding_window_view(padded, window_size)
        result = gmean(windowed, axis=1, nan_policy="omit")
        if cut_right_edge:
            return result[:-window_size + 1]
        else:
            return result

    else:

        return uniform_filter1d(input, size=window_size)


# assumong results has shape (total_repeats, number_of_conditions, max_steps)
def trim_a_results_array(results):
    index_to_cut = np.argmax(np.any(results[:, :, ::-1], axis=(0, 1)))
    if index_to_cut == 0 and np.all(results[:, :, 0] == 0):
        return None
    elif index_to_cut == 0:
        return results
    else:
        return results[:, :, :-index_to_cut]


def load_data(base_file_path: str, dimension, counts=False):

    if not counts:
        data = np.load(base_file_path + fr"\dimension_{dimension}.npz", "rb")
        lipschitz_values = data["lipschitz_values"]
        y_distances = data["y_distances"]
        counts = data["counts"]
        condition_numbers = data['condition_numbers']

        return lipschitz_values, y_distances, counts, condition_numbers
    else:
        data = np.load(base_file_path + fr"\dimension_{dimension}.npz", "rb")

        counts, lipschitz_values, condition_numbers = data['step_counts'], data['lipschitz_values'], data[
        'condition_numbers']

        return counts, lipschitz_values, condition_numbers



# point: (x,y) line: (a,b,c) where ax + by + c = 0
def distance_from_point_to_line(point, line):
    return (line[0] * point[0] + line[1] * point[1] + line[2]) / np.linalg.norm(line[:2])


def closest_point_on_line_to_point(point, line):
    a, b, c = line
    x0, y0 = point
    x = (b * (b * x0 - a * y0) - a * c) / (a ** 2 + b ** 2)
    y = (a * (-b * x0 + a * y0) - b * c) / (a ** 2 + b ** 2)
    return x, y


def remove_outliers(data, m=5):
    return np.where(abs(data - np.nanmean(data)) < m * np.nanstd(data), data, np.nan)
