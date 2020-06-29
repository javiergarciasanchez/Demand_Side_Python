from Firms import *

import matplotlib.pyplot as plt
import numpy as np
import itertools
# import mpl_toolkits.mplot3d


def calc_res_for_draw(p, q, f: Firm, comp: Firm = None, var="profit"):
    f_copy = copy(f)
    f_copy.p = p
    f_copy.q = q

    return max(0, getattr(calculate_result(f_copy, comp), var))


def opt_price_for_draw(q, f: Firm, comp: Firm = None):
    f_copy = copy(f)
    f_copy.q = q

    return opt_price(f_copy, comp)


def plot_vector_func(p_range, q_range, f: Firm, comp: Firm = None, var="profit", title="",
                     p_lines=None, q_lines=None, color=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if comp is None:
        comp_label = ""
    else:
        comp_label = "\n\n Comp price: " + str(comp.p) + ". Comp quality: " + str(comp.q)

    plt.title(title + comp_label)

    ax.set_xlabel("Price")
    ax.set_ylabel("Quality")

    p_grid, q_grid = np.meshgrid(p_range, q_range)
    v_fc = np.vectorize(calc_res_for_draw, otypes=[np.float64])
    ax.plot_wireframe(p_grid, q_grid, v_fc(p_grid, q_grid, f, comp, var), color=color)

    if p_lines is not None:
        # Add p lines
        for p in p_lines:
            p_const = list(itertools.repeat(p, len(q_range)))
            ax.plot(p_const, q_range, v_fc(p_const, q_range, f, comp, var),
                    color="black", linewidth=3)

    elif q_lines is not None:
        # Add q lines
        for q in p_lines:
            q_const = list(itertools.repeat(q, len(p_range)))
            ax.plot(p_range, q_const, v_fc(p_range, q_const, f, comp, var),
                    color="black", linewidth=3)

    return fig, ax


def plotPriceVectorFunc(p_range, q, f, comp=None, var="profit", title=None):
    fig, ax = plt.subplots()

    if title is None:
        title = var

    if comp is None:
        comp_label = ""
    else:
        comp_label = "\n\n Comp price: " + str(comp.p) + ". Comp quality: " + str(comp.q)

    ax.set_title(title + "\n\n Quality: " + str(q) + comp_label)
    ax.set_xlabel("Price")

    v_fc = np.vectorize(calc_res_for_draw, otypes=[np.float64])
    ax.plot(p_range, v_fc(p_range, q, f, comp, var))

    return fig, ax


def plotQualityVectorFunc(p, q_range, f, comp=None, var="profit", title=""):

    fig, ax = plt.subplots()

    if comp is None:
        comp_label = ""
    else:
        comp_label = "\n\n Comp price: " + str(comp.p) + ". Comp quality: " + str(comp.q)

    ax.set_title(title + "\n\n Price: " + str(p) + comp_label)

    ax.set_xlabel("Quality")

    v_fc = np.vectorize(calc_res_for_draw, otypes=[np.float64])
    ax.plot(q_range, v_fc(p, q_range, f, comp, var))

    return fig, ax
