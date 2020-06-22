# -*- coding: utf-8 -*-

# Global Parameters and functions


from scipy.optimize import minimize_scalar

# from statistics import mean

gini = 0.7
population = 4000
thetaMin = 1
discountQ = 0.7

eta = 0.8  # Exponent of quality in utility function

cParam = 0.01
gamma = 2

plambda = (1 + gini) / (2 * gini)


def utility(theta, p, q):
    return theta * q ** eta - p


def limit(p, q, pComp=None, qComp=None):
    # Return lo and hi limits

    if pComp is None or qComp is None:
        lo = p / q ** eta
        hi = None

    elif q > qComp:
        lo = (p - pComp) / (q ** eta - qComp ** eta)
        hi = None

    elif q == qComp and p <= pComp:
        # if both firms have equal p and q current firm takes all market
        lo = p / q ** eta
        hi = None

    elif q == qComp and p > pComp:
        # out of the market
        lo = None
        hi = None

    else:
        lo = p / q ** eta
        hi = (pComp - p) / (qComp ** eta - q ** eta)

    if lo is not None:
        lo = max(thetaMin, lo)

    if hi is not None:
        hi = max(thetaMin, hi)

    return lo, hi


def k(lim):
    return population * (thetaMin / lim) ** plambda


def cant(p, q, pComp=None, qComp=None):
    lo, hi = limit(p, q, pComp, qComp)

    if hi is None:
        return k(lo)
    else:
        return max(0, k(lo) - k(hi))


def cantPriceDeriv(p, q, pComp=None, qComp=None):
    return 0


def cost(q):
    return cParam * q ** gamma


def profit(p, q, pComp=None, qComp=None, d=1, dComp=1):
    return (p - cost(q)) * cant(p, q * d, pComp, qComp * dComp)


def minProfit(p, q, pComp=None, qComp=None, d=1, dComp=1):
    return - profit(p, q, pComp, qComp, d, dComp)


def profPriceDeriv(p, q, pComp=None, qComp=None):
    return ((p - cost(q)) * cantPriceDeriv(p, q, pComp, qComp)
            + cant(p, q, pComp, qComp))


def optPrice(q, pComp=None, qComp=None, d=1, dComp=1):
    if pComp is None or qComp is None:
        pComp = 0
        qComp = 0

    qd = q * d
    qCompd = qComp * dComp

    if qd > qCompd:
        return max(thetaMin * (qd ** eta - qCompd ** eta) + pComp,
                   (plambda * cost(q) - pComp) / (plambda - 1))

    elif qd < qCompd:
        return minimize_scalar(minProfit, args=(q, pComp, qComp, d, dComp)).x

    elif qd == qCompd:
        return pComp - 0.1


class FirmResult():
    prof = 0
    Quant = 0
    lo = thetaMin
    hi = thetaMin


class FirmMktSt():
    def __init__(self, p, q, pComp, qComp, d, dComp):
        self.p = p
        self.q = q
        self.d = d
        self.pComp = pComp
        self.qComp = qComp
        self.dComp = dComp


def collect_result(fms):
    r = FirmResult()

    r.prof = profit(fms.p, fms.q, fms.pComp, fms.qComp, fms.d, fms.dComp)
    r.Quant = cant(fms.p, fms.q * fms.d, fms.pComp, fms.qComp * fms.dComp)
    r.lo, r.hi = limit(fms.p, fms.q * fms.d, fms.pComp, fms.qComp * fms.dComp)

    return r


def iterate_price(pd, qd, ph, qh, dd, dh, iters=50):
    fmsd = FirmMktSt(pd, qd, ph, qh, dd, dh)
    fmsh = FirmMktSt(ph, qh, dh, pd, qd, dd)
    rd = [collect_result(fmsd)]
    rh = [collect_result(fmsh)]

    vpd = [pd]
    vph = [ph]

    break_tolerance = 5
    for i in range(iters):
        p = optPrice(qd, vph[-1], qh, dd, dh)
        vpd.append(p)
        fmsd.p = p
        fmsh.pComp = p
        rd.append(collect_result(fmsd))
        rh.append(collect_result(fmsh))

        p = optPrice(qh, vpd[-1], qd, dh, dd)
        vph.append(p)
        fmsh.p = p
        fmsd.pComp = p
        rd.append(collect_result(fmsd))
        rh.append(collect_result(fmsh))

        if abs(vpd[-2] - vpd[-1]) + abs(vph[-2] == vph[-1]) < 0.000001:
            break_tolerance -= 1

        if break_tolerance == 0:
            break

    return vpd, vph, rd, rh


def nashCondition(pd, ph, qd, qh):
    # Check conditions
    if (pd == ph) or (qd == qh):
        return None

    if (pd <= cost(qd)) or (ph <= cost(qh)):
        return None

    lo = pd / qd ** eta
    deltaQ = qh ** eta - qd ** eta
    lim = (ph - cost(qh)) * plambda / deltaQ

    retval1 = - (pd - cost(qd)) * plambda * pd * lim
    retval2 = k(lo) * lim + k(lim) * pd / deltaQ

    return retval1 * retval2 + k(lo) - k(lim)