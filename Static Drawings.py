from Drawings import *

Params.discountQ = 1

fh = Firm(1, 30, None)
ph = opt_price(fh)
fh.p = ph

# fd = None

fd = Firm(1, 10, None)
pd = opt_price(fd, fh)
fd.p = pd

# recalculate opt
ph = opt_price(fh, fd)
fh.p = ph

p_range = np.arange(0.0, 40.0, 0.1)
q_range = np.arange(0.001, 50, 0.1)

var = "profit"
fig_h, ax_h = plot_vector_func(p_range, q_range, fh, fd, var=var, title=var, color="blue")
ax_h.scatter(fh.p, fh.q, getattr(calculate_result(fh, fd), var), c="red", s=30)
ax_h.view_init(azim=-119, elev=46)

# add optimum profit line
q_range = np.arange(0.001, 35, 0.1)
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
p_range = np.arange(0, 100, 0.5)
q_range = np.arange(0.001, 17, 0.5)
fig_d, ax_d = plot_vector_func(p_range, q_range, fd, fh, var=var, title=var, color="blue")
ax_d.scatter(fd.p, fd.q, getattr(calculate_result(fd, fh), var), c="red", s=30)
ax_d.view_init(azim=-119, elev=46)

# add optimum profit line
q_range = np.arange(0.001, 17, 0.5)
v_opt_price = np.vectorize(opt_price_for_draw, otypes=[np.float64])
v_opt = v_opt_price(q_range, fd, fh)

v_calc_res = np.vectorize(calc_res_for_draw, otypes=[np.float64])
z = v_calc_res(v_opt, q_range, fd, fh, var=var)

ax_d.plot(v_opt, q_range, z, c="red", linewidth=5)
