## %%
from Firms import *
from Drawings import *

qh = 30
dh = 1
ph = optPrice(qh)

qd = 10
dd = 0.8
pd = optPrice(qd, ph, qh, dd, dh)

vpd, vph, rd, rh = iterate_price(pd, qd, ph, qh, dd, dh)

fig, ax = plt.subplots()
fig.suptitle("Price")
ax.plot(vpd)
ax.plot(vph)

fig, axProf = plt.subplots()
fig.suptitle("Profit")
axProf.plot(list(o.prof for o in rd))
axProf.plot(list(o.prof for o in rh))

fig, axLim = plt.subplots()
fig.suptitle("Limits")
axLim.plot(list(o.lo for o in rd))
axLim.plot(list(o.hi for o in rd))

fig, axCant = plt.subplots()
fig.suptitle("Quantities")
axCant.plot(list(o.Quant for o in rd))
axCant.plot(list(o.Quant for o in rh))

## %%
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

## %%
pRange = np.arange(0, 50, 0.1)
qRange = np.arange(0.001, 25, 0.1)

# Lower Firm
p1 = 2
q1 = 10

# Higher Firm
p2 = 4.417587942488388
q2 = 15

pComp = p2
qComp = q2

plot_vector_func(pRange, qRange, pComp, qComp, profit, "Profit")
plot_vector_func(pRange, qRange, pComp, qComp, cant, "Quantity")

# with optimal price
vOptPrice = np.vectorize(optPrice, otypes=[np.float64])
pOpt = vOptPrice(qRange, pComp, qComp)

plot_vector_func(pRange, qRange, pComp, qComp, profit, "Profit",
                 pOpt, qRange)

plot_vector_func(pRange, qRange, 10, 20, cant, "Quantity")

plotPriceVectorFunc(pRange, q1, p2, q2, profit, "Profit")

plot_vector_func(pRange, qRange, None, None, profit, "Profit",
                 vOptPrice(qRange, None, None), qRange)
