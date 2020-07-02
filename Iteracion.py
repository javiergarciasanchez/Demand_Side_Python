from Drawings import *

Params.discountQ = 1

fh = Firm(1, 30, None)
# fh.p = opt_price(fh)

fd = Firm(1, 25, None)
# fd.p = opt_price(fd, fh)

fh_seq, fd_seq, fh_res_seq, fd_res_seq = iterate_price(fh, fd)

q_title = "Down q:" + str(fd_seq[0].q) + " - High q: " + str(fh_seq[0].q)

fig, ax = plt.subplots()
fig.suptitle("Price\n" + q_title)
ax.plot(list(o.p for o in fd_seq), label="Down")
ax.plot(list(o.p for o in fh_seq), label="High")
ax.legend()

fig, axProf = plt.subplots()
fig.suptitle("Profit\n" + q_title)
axProf.plot(list(o.profit for o in fd_res_seq), label="Down")
axProf.plot(list(o.profit for o in fh_res_seq), label="High")
axProf.legend()

fig, axLim = plt.subplots()
fig.suptitle("Limits\n" + q_title)
axLim.plot(list(o.down_limit for o in fd_res_seq), label="Lowest")
axLim.plot(list(o.high_limit for o in fd_res_seq), label="Middle")
axLim.legend()

fig, axCant = plt.subplots()
fig.suptitle("Quantities\n" + q_title)
axCant.plot(list(o.quantity for o in fd_res_seq), label="Down")
axCant.plot(list(o.quantity for o in fh_res_seq), label="High")
axCant.legend()

pRange = np.arange(0, 50, 0.1)
qRange = np.arange(0.001, 50, 0.1)

# Individual 3D Draw
i = len(fh_seq) - 1
# fd
fig, ax1 = plot_vector_func(pRange, qRange, fd_seq[i], fh_seq[i], "profit",
                            "Profit Down: " + str(i), color="blue")
ax1.scatter(fd_seq[i].p, fd_seq[i].q, calculate_result(fd_seq[i], fh_seq[i]).profit,
            c="red", s=30)
ax1.view_init(azim=-119, elev=46)

# fh
fig, ax2 = plot_vector_func(pRange, qRange, fh_seq[i], fd_seq[i], "profit",
                            "Profit High: " + str(i), color="blue")
ax2.scatter(fh_seq[i].p, fh_seq[i].q, calculate_result(fh_seq[i], fd_seq[i]).profit,
            c="red", s=30)
ax2.view_init(azim=-119, elev=46)

# Multidraws
tot_iters = len(fh_seq)
fig = plt.figure()
iters = 3
# Last iters
for i in range(tot_iters - iters, tot_iters):
    fig, ax1 = add_plot_vector_func(fig, iters, 2, i * 2 + 1, pRange, qRange, fd_seq[i], fh_seq[i], "profit",
                                    "Profit Down: " + str(i), color="blue")
    ax1.scatter(fd_seq[i].p, fd_seq[i].q, calculate_result(fd_seq[i], fh_seq[i]).profit,
                c="red", s=30)
    # ax1.set_title(ax1.get_title() +
    #               "\n Comp Point - p: " + str(fh_seq[i].p) + " q: " + str(fh_seq[i].q))
    ax1.view_init(azim=-119, elev=46)

    fig, ax2 = add_plot_vector_func(fig, iters, 2, (i + 1) * 2, pRange, qRange, fh_seq[i], fd_seq[i], "profit",
                                    "Profit High: " + str(i), color="blue")
    ax2.scatter(fh_seq[i].p, fh_seq[i].q, calculate_result(fh_seq[i], fd_seq[i]).profit,
                c="red", s=30)
    # ax2.set_title(ax2.get_title() +
    #               "\n Comp Point - p: " + str(fd_seq[i].p) + " q: " + str(fd_seq[i].q))
    ax2.view_init(azim=-119, elev=46)

vN = np.vectorize(nashCondition, otypes=[np.float64])
pR = pRange = np.arange(0.001, 30, 0.1)
PD, PH = np.meshgrid(pR, pR)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(PD, PH, vN(PD, PH, 10, 15))
