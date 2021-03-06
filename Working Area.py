from Drawings import *

Params.discountQ = 1

fh = Firm(1, 30, None)
fh.p = opt_price(fh)

fd = Firm(10.98561125948348, 20, None)
#fd.p = opt_price(fd, fh)

# recalculate opt
fh.p = opt_price(fh, fd)

ft = n_firm(fh, q=9.95)
fto = n_firm(ft, p= opt_price(ft, fd))

p_range = np.arange(0.0, 60.0, 0.1)
q_range = np.arange(0.001, 40, 0.1)

var = "profit"
fig_h, ax_h = plot_vector_func(p_range, q_range, fh, fd, var=var, title=var, color="blue")
ax_h.scatter(fh.p, fh.q, getattr(calculate_result(fh, fd), var), c="red", s=30)
ax_h.view_init(azim=-119, elev=46)

plotPriceVectorFunc(p_range, fh.q, fh, fd, var="profit")

plotPriceVectorFunc(p_range, ft.q, ft, fd, var="quantity")

v_fc = np.vectorize(calc_res_for_draw, otypes=[np.float64])