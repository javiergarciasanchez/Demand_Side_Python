from Drawings import *

Params.discountQ = 1

fh = Firm(1, 30, None)
ph = opt_price(fh)
fh.p = ph

fd = Firm(1, 16, None)
pd = opt_price(fd, fh)
fd.p = pd

fd_seq, fh_seq, fd_res_seq, fh_res_seq = iterate_price(fh, fd)

fig, ax = plt.subplots()
fig.suptitle("Price")
ax.plot(list(o.p for o in fd_seq))
ax.plot(list(o.p for o in fh_seq))

fig, axProf = plt.subplots()
fig.suptitle("Profit")
axProf.plot(list(o.profit for o in fd_res_seq))
axProf.plot(list(o.profit for o in fh_res_seq))

fig, axLim = plt.subplots()
fig.suptitle("Limits")
axLim.plot(list(o.down_limit for o in fd_res_seq))
axLim.plot(list(o.high_limit for o in fh_res_seq))

fig, axCant = plt.subplots()
fig.suptitle("Quantities")
axCant.plot(list(o.quantity for o in fd_res_seq))
axCant.plot(list(o.quantity for o in fh_res_seq))


iters = len(vpd)
pRange = np.arange(0, 50, 0.1)
qRange = np.arange(0.001, 50, 0.1)
for i in range(iters):
    fig1, ax1 = plot_vector_func(pRange, qRange, vph[i], qh, profit,
                                 "Profit down: " + str(i), color="blue")
    ax1.scatter(vpd[i], qd, profit(vpd[i], qd, vph[i], qh, dd, dh, dd, dh),
                c="red", s=30)
    ax1.set_title(ax1.get_title() +
                  "\n Point - p: " + str(vph[i]) + " q: " + str(qh))
    ax1.view_init(azim=-119, elev=46)

    fig2, ax2 = plot_vector_func(pRange, qRange, vpd[i], qd, profit,
                                 "Profit High: " + str(i), color="green")
    ax2.scatter(vph[i], qh, profit(vph[i], qh, vpd[i], qd, dh, dd),
                c="red", s=30)
    ax2.set_title(ax2.get_title() +
                  "\n Point  p: " + str(vpd[i]) + "  q: " + str(qd))
    ax2.view_init(azim=-119, elev=46)

vN = np.vectorize(nashCondition, otypes=[np.float64])
pR = pRange = np.arange(0.001, 30, 0.1)
PD, PH = np.meshgrid(pR, pR)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(PD, PH, vN(PD, PH, 10, 15))
