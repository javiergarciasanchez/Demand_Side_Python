from Drawings import *

Params.discountQ = 1

fh = Firm(1, 30, math.inf)
fh.p = opt_price(fh)

# fd = None

fd = Firm(15, 20, None)
fd.p = opt_price(fd, fh)

# recalculate opt
fh.p = opt_price(fh, fd)

p_range = np.arange(0.0, 60.0, 0.1)
q_range = np.arange(0.001, 40, 0.1)

var = "profit"
fig_h, ax_h = plot_vector_func(p_range, q_range, fh, fd, var=var, title=var, color="blue")
ax_h.scatter(fh.p, fh.q, getattr(calculate_result(fh, fd), var), c="black", s=30)
ax_h.view_init(azim=-119, elev=46)

# 2D profit
plotPriceVectorFunc(p_range, fh.q, fh, fd, var="profit")

# add optimum profit line
q_range = np.arange(0.001, 40, 0.1)
v_opt_price = np.vectorize(opt_price_for_draw, otypes=[np.float64])
v_opt = v_opt_price(q_range, fh, fd)

v_calc_res = np.vectorize(calc_res_for_draw, otypes=[np.float64])
z = v_calc_res(v_opt, q_range, fh, fd, var=var)

ax_h.plot(v_opt, q_range, z, c="red", linewidth=5)

var = "quantity"
fig_h_q, ax_h_q = plot_vector_func(np.arange(0.0, 100.0, 0.5), np.arange(0.0001, 100, 0.5), fh,
                                   var=var, title=var, color="blue")
ax_h_q.scatter(fh.p, fh.q, getattr(calculate_result(fh), var), c="red", s=30)
ax_h_q.view_init(azim=-119, elev=46)

#####
# fd
#####
var = "profit"
p_range = np.arange(0, 60, 0.1)
q_range = np.arange(0.001, 40, 0.1)
fig_d, ax_d = plot_vector_func(p_range, q_range, fd, fh, var=var, title=var, color="blue")
ax_d.scatter(fd.p, fd.q, getattr(calculate_result(fd, fh), var), c="red", s=30)
ax_d.view_init(azim=-119, elev=46)

# 2D profit
plotPriceVectorFunc(p_range, fd.q, fd, fh, var="profit")


# add optimum profit line
q_range = np.arange(0.001, 40, 0.1)
v_opt_price = np.vectorize(opt_price_for_draw, otypes=[np.float64])
v_opt = v_opt_price(q_range, fd, fh)

v_calc_res = np.vectorize(calc_res_for_draw, otypes=[np.float64])
z = v_calc_res(v_opt, q_range, fd, fh, var=var)

ax_d.plot(v_opt, q_range, z, c="red", linewidth=5)
