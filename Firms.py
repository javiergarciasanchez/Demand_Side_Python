# -*- coding: utf-8 -*-

# Global Parameters and functions
import math
from copy import copy
from typing import Tuple, List

from scipy.optimize import minimize_scalar


class Params:
    # General Params
    gini = 0.7
    population = 4000.0

    cParam = 0.01
    gamma = 2  # Exponent of quality in cost function

    thetaMin = 1.0
    eta = 0.8  # Exponent of quality in utility function
    discountQ = 0.7

    price_tol = 1e-4


def plambda(gini_value):
    return (1 + gini_value) / (2 * gini_value)


def utility(theta, p, q):
    return theta * q ** Params.eta - p


def richest_consumer():
    return Params.thetaMin * Params.population ** (1.0 / plambda(Params.gini))


class Firm:
    p = 0.0
    q = 0.0

    # highest consumer that has tried firm product
    highest_cons = None

    # For simplicity reasons there won't be holes in the consumer segments that know the firm
    # Once the firm entered the market, it is assumed all consumers below hi_cons have tried the firm product
    # Nome means firm product is untried for all consumers

    def __init__(self, p, q, hc):
        self.p = p
        self.q = q
        self.highest_cons = hc


def n_firm(f: Firm, p=None, q=None):
    nf = copy(f)

    if p is not None:
        nf.p = p

    if q is not None:
        nf.q = q

    return nf


class Market:
    # when there is only one firm it is the highest
    high_firm: Firm = None
    down_firm: Firm = None

    # lower limits of firms. When there is only one firm only high_limit is set
    high_limit: float = None
    down_limit: float = None

    # The order is established using quality without taking into account discount
    def __init__(self, f1: Firm, f2: Firm = None):

        if f2 is None:
            self.high_firm = f1
        elif f1.q > f2.q:
            self.high_firm = f1
            self.down_firm = f2
        elif f1.q < f2.q:
            self.high_firm = f2
            self.down_firm = f1
        elif f1.p < f2.p:
            # f1.q == f2.q
            self.high_firm = f1
        elif f1.p > f2.p:
            # f1.q == f2.q
            self.high_firm = f2
        else:
            # f1.q == f2.q
            # f1.q == f2.q
            # if both firms offer the same, the whole market is assigned to f1
            self.high_firm = f1

        set_market_limits(self)


class FirmResult:
    profit = 0.0
    quantity = 0.0
    down_limit = Params.thetaMin
    high_limit = Params.thetaMin


def absolute_min_price(f: Firm):
    if f.highest_cons is None:
        q = f.q * Params.discountQ
    else:
        q = f.q

    return price_operator(q, Params.thetaMin)


def price_operator(q, theta, cp=0.0, cq=0.0):
    # it returns the price that makes the limit operator equal to theta

    assert q != cq

    lim = min(theta, richest_consumer())

    if q > cq:
        return lim * (q ** Params.eta - cq ** Params.eta) + cp
    else:
        # cq < q
        return cp - lim * (q ** Params.eta - cq ** Params.eta)


def absolute_max_price(f: Firm, comp: Firm = None):
    rc = richest_consumer()

    if f.highest_cons is None:
        q = f.q * Params.discountQ
    elif f.highest_cons < rc:
        q = f.q * Params.discountQ
    else:
        q = f.q

    if comp is None:
        return price_operator(q, rc)
    else:
        p = price_operator(q, rc, comp.p, comp.q)
        pd = price_operator(q, rc, comp.p, comp.q * Params.discountQ)
        return max(p, pd)


def discontinuities_by_q(q, hc, comp_p, comp_q, comp_hc) -> List[float]:
    if q > comp_q:
        # f is High

        # L == Lo
        lo = comp_p / comp_q ** Params.eta
        retval = [price_operator(q, lo, comp_p, comp_q)]

        # L = D_hc
        if hc is not None:
            retval += [price_operator(q, hc, comp_p, comp_p)]

        # L = H_hc
        if comp_hc is not None:
            retval += [price_operator(q, comp_hc, comp_p, comp_p)]

        # Lo = D_hc is independent of ph
        # Lo = H_hc is independent of ph

    elif q < comp_q:
        # f is down

        # L == Lo
        retval = [comp_p * q ** Params.eta / comp_q ** Params.eta]

        # L = D_hc
        if hc is not None:
            retval += [price_operator(q, hc, comp_p, comp_p)]

        # L = H_hc
        if comp_hc is not None:
            retval += [price_operator(q, comp_hc, comp_p, comp_p)]

        # Lo = D_hc
        if hc is not None:
            retval += [hc * q ** Params.eta]

        # Lo = H_hc
        if comp_hc is not None:
            retval += [comp_hc * q ** Params.eta]

    else:
        # q == comp_q

        # one firm is expelled. The one with lower price remains. Border occurs when prices are equal
        retval = [comp_p]

        # L = Lo
        # L = D_hc
        # L = H_hc
        # L doesn't exist

        # Lo = D_hc
        # Lo = H_hc
        # Only one firm remains thus H_hc is irrelevant
        if hc is not None:
            retval += [hc * q ** Params.eta]

    return retval


def discontinuities(f: Firm, comp: Firm = None) -> List[float]:
    # it includes abs min price and abs max price

    r_min = absolute_min_price(f)
    r_max = absolute_max_price(f, comp)
    retval: List[float] = [r_min]

    qd = f.p * Params.discountQ

    if comp is None:
        if f.highest_cons is not None:
            retval += [price_operator(f.q, f.highest_cons)]
            retval += [price_operator(qd, f.highest_cons)]

    else:
        # Two firms
        # we need to calculate the five discontinuities for the different combinations of discounted quality

        cqd = comp.q * Params.discountQ

        if f.highest_cons is None:
            retval += discontinuities_by_q(f.q, f.highest_cons, comp.p, comp.q, comp.highest_cons)
            retval += discontinuities_by_q(f.q, f.highest_cons, comp.p, cqd, comp.highest_cons)
        else:
            retval += discontinuities_by_q(f.q, f.highest_cons, comp.p, comp.q, comp.highest_cons)
            retval += discontinuities_by_q(f.q, f.highest_cons, comp.p, cqd, comp.highest_cons)
            retval += discontinuities_by_q(qd, f.highest_cons, comp.p, comp.q, comp.highest_cons)
            retval += discontinuities_by_q(qd, f.highest_cons, comp.p, cqd, comp.highest_cons)

    retval = sorted(filter(lambda x: r_min <= x < r_max, set(retval)))

    return retval + [r_max]


def limit_operator(ph, qh, pd=0.0, qd=0.0) -> float:
    # Returns False if order should be reverted

    if qh > qd:
        return (ph - pd) / (qh ** Params.eta - qd ** Params.eta)
    elif qd > qh:
        return (pd - ph) / (qd ** Params.eta - qh ** Params.eta)
    elif ph <= pd:
        # qd == qh. Only high firm remains
        return Params.thetaMin
    else:
        # qd == qh and ph > pd. Only down firm remains
        return Params.thetaMin


def lowest_limit(f: Firm) -> float:
    # For simplicity reasons no hole would be left in market segment

    if f.highest_cons is None:
        # f is entering industry
        retval = limit_operator(f.p, f.q * Params.discountQ)
    else:
        # Note there is a hole for theta if lim <= f.highest_cons <= theta <= lim_disc
        retval = limit_operator(f.p, f.q)

        if retval > f.highest_cons:
            retval = limit_operator(f.p, f.q * Params.discountQ)

    return max(Params.thetaMin, retval)


def middle_limit(down_firm: Firm, high_firm: Firm) -> Tuple[bool, float]:
    # For simplicity reasons no hole would be left in market segment
    # hi Firm would have preference when holes would appear

    d = Params.discountQ

    # indicates if order should be reversed due to discount
    reverse = False

    down__high = limit_operator(high_firm.p, high_firm.q, down_firm.p, down_firm.q)
    disc_down__high = limit_operator(high_firm.p, high_firm.q, down_firm.p, down_firm.q * d)
    down__disc_high = limit_operator(high_firm.p, high_firm.q * d, down_firm.p, down_firm.q)
    disc_down__disc_high = limit_operator(high_firm.p, high_firm.q * d, down_firm.p, down_firm.q * d)

    if down_firm.highest_cons is None and high_firm.highest_cons is None:
        # both entrants
        retval = disc_down__disc_high

    elif down_firm.highest_cons is None:
        # down is entrant
        retval = disc_down__high

    elif high_firm.highest_cons is None:
        # high is entrant
        # this is the unique case where order could be reversed
        reverse = (high_firm.q * d < down_firm.q) or \
                  ((high_firm.q * d == down_firm.q) and (high_firm.p > down_firm.p))
        retval = down__disc_high

    else:
        # Both are incumbents
        if down_firm.highest_cons == math.inf:
            retval = down__high
        elif down__high <= down_firm.highest_cons:
            retval = down__high
        elif disc_down__high < down_firm.highest_cons:
            retval = down_firm.highest_cons
        else:
            # down_firm.highest_cons <= disc_down__high
            retval = disc_down__high

    return reverse, max(Params.thetaMin, retval)


def set_market_limits(mkt: Market):
    # Set limits and changes order of firms if it is needed due to discount
    # Expels firm if it is needed
    # when there is only one firm it is the highest

    assert mkt.down_firm is None or mkt.down_firm.q < mkt.high_firm.q

    if mkt.down_firm is None:
        # there is only one firm
        mkt.high_limit = lowest_limit(mkt.high_firm)

    else:
        reverse, ml = middle_limit(mkt.down_firm, mkt.high_firm)
        if reverse:
            tmp_f = mkt.high_firm
            mkt.high_firm = mkt.down_firm
            mkt.down_firm = tmp_f

        mkt.down_limit = lowest_limit(mkt.down_firm)
        mkt.high_limit = ml

        if mkt.down_limit >= mkt.high_limit:
            # down firm is expelled
            mkt.down_firm = None
            mkt.down_limit = None
            mkt.high_limit = lowest_limit(mkt.high_firm)

    return mkt.down_limit, mkt.high_limit


def k(lim):
    return Params.population * (Params.thetaMin / lim) ** plambda(Params.gini)


def quantity(f: Firm, mkt: Market):
    # returns if firm is the higher and quantity

    if f == mkt.high_firm:
        return k(mkt.high_limit)
    elif f == mkt.down_firm:
        return k(mkt.down_limit) - k(mkt.high_limit)
    else:
        # Firm was not able to enter the market
        return 0


def cost(q):
    return Params.cParam * q ** Params.gamma


def calculate_result(f, comp=None) -> FirmResult:
    retval = FirmResult()

    mkt = Market(f, comp)

    if f == mkt.high_firm:
        retval.down_limit = mkt.high_limit
        retval.high_limit = math.inf
    elif f == mkt.down_firm:
        retval.down_limit = mkt.down_limit
        retval.high_limit = mkt.high_limit
    else:
        # firm is not in the market
        return retval

    retval.quantity = quantity(f, mkt)
    retval.profit = (f.p - cost(f.q)) * retval.quantity

    return retval


def min_profit(p, f, comp):
    tmp_f = copy(f)
    tmp_f.p = p
    return - calculate_result(tmp_f, comp).profit


def opt_price(f, comp=None):
    results = []

    dis = discontinuities(f, comp)
    low = dis.pop(0)

    for high in dis:
        if high - low >= Params.price_tol * 2:
            results.append(minimize_scalar(min_profit, method="bounded", args=(f, comp), bounds=(low, high),
                                           options={'xatol': Params.price_tol}))
        low = high

    max_prof = 0
    opt_p = 0
    for r in results:
        if - r.fun > max_prof:
            max_prof = - r.fun
            opt_p = r.x

    return opt_p


def iterate_price(f1, f2, max_iters=50):
    f1_seq = [f1]
    f1_res_seq = [calculate_result(f1, f2)]

    f2_seq = [f2]
    f2_res_seq = [calculate_result(f2, f1)]

    break_tolerance = 5
    for i in range(max_iters):
        # Optimize firm 1
        f = copy(f1_seq[-1])
        comp = f2_seq[-1]
        f.p = opt_price(f, comp)
        f1_seq.append(f)
        f1_res_seq.append(calculate_result(f, comp))

        # Optimize firm 2
        f = copy(f2_seq[-1])
        comp = f1_seq[-1]
        f.p = opt_price(f, comp)
        f2_seq.append(f)
        f2_res_seq.append(calculate_result(f, comp))

        if abs(f1_seq[-2].p - f1_seq[-1].p) + abs(f2_seq[-2].p - f2_seq[-1].p) < 0.000001:
            break_tolerance -= 1

        if break_tolerance == 0:
            break

    return f1_seq, f2_seq, f1_res_seq, f2_res_seq


def nashCondition(pd, ph, qd, qh):
    # Check conditions
    if (pd == ph) or (qd == qh):
        return None

    if (pd <= cost(qd)) or (ph <= cost(qh)):
        return None

    lo = pd / qd ** Params.eta
    delta_q = qh ** Params.eta - qd ** Params.eta
    lim = (ph - cost(qh)) * plambda(Params.gini) / delta_q

    retval1 = - (pd - cost(qd)) * plambda(Params.gini) * pd * lim
    retval2 = k(lo) * lim + k(lim) * pd / delta_q

    return retval1 * retval2 + k(lo) - k(lim)
