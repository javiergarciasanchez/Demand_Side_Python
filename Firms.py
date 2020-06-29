# -*- coding: utf-8 -*-

# Global Parameters and functions
import math
from copy import copy
from typing import Tuple

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


def min_price_for_q(f: Firm, comp: Firm = None):
    # If comp is not None, and firm is high, min price should keep competitor, ie L > lo
    # Otherwise L = Lo >= thetaMin


    qd = f.q * Params.discountQ
    cqd = comp.q * Params.discountQ

    hc = f.highest_cons

    if comp is None:
        if hc is None:
            return qd ** Params.eta * Params.thetaMin
        else:
            # ThetaMin is already a customer
            return f.q ** Params.eta * Params.thetaMin



    if f.q > qd >= comp.q > cqd:
        # Full high firm

    elif comp.q > cqd > f.q > qd:
        # full down firm

    elif comp.q >= f.q >= cqd >= qd:

    elif comp.q >= f.q > qd >= cqd:

    elif f.q >= comp.q > cqd >= qd:

    else:
        # f.q >= comp.q > qd >= cqd

    else:
        # q >= comp.q
        if comp.p / comp.d ** Params.eta <= comp.highest_cons:
            return comp.p * (comp.q / q) ** Params.eta
        else:
            return comp.p * (comp.q * Params.discountQ / q) ** Params.eta


def max_price_for_down_firm(q, comp_p, comp_q):

    assert  q <= comp_q

    return comp_p * (q / comp_q) ** Params.eta


def max_price_for_richest_consumer(f: Firm):

    max_q = Params.thetaMin * Params.population ** (1.0 / plambda(Params.gini)) * f.q ** Params.eta

    qd = f.q * Params.discountQ
    max_dq = Params.thetaMin * Params.population ** (1.0 / plambda(Params.gini)) * qd ** Params.eta

    if f.highest_cons is None:
        return max_dq

    elif f.highest_cons > richest_consumer():
        return max_q

    else:
        return max_dq


    return Params.thetaMin * Params.population ** (1.0 / plambda(Params.gini)) * q ** Params.eta


def max_price_for_high_firm(q, comp:Firm = None):
    # It is assumed competitor is already in the market, ie comp.highest_cons is not None

    if comp is None:
        comp_p = 0
        comp_q = 0
    else:
        comp_p = comp.p
        comp_q = comp.q

    return Params.thetaMin * Params.population ** (1.0 / plambda(Params.gini)) * \
           (q ** Params.eta - comp_q ** Params.eta) + comp_p


def max_price_for_down_firm_for_consumer(q, comp: Firm, theta):
    # Max price is the one that makes limit_operator equal to theta

    assert q <= comp.q
    return comp.p - theta * (comp.q ** Params.eta - q ** Params.eta)


def bounds_for_maximization(f: Firm, comp: Firm = None) -> [Tuple[float, float]]:
    # It is assumed competitor is already in the market, ie comp.highest_cons is not None

    qd = f.q * Params.discountQ
    f_hc = f.highest_cons

    if comp is None:
        if f_hc is None:
            rmin = min_price_for_q(qd)
            rmax = max_price_for_high_firm(qd)

        else:
            # if it is in the market thetaMin has already tried the product
            rmin = min_price_for_q(f.q)

            if f_hc >= richest_consumer():
                rmax = max_price_for_high_firm(f.q)
            else:
                rmax = max_price_for_high_firm(qd)

        return [(rmin, rmax)]

    elif qd > comp.q:
        # there are two bounds, keeping comp and expelling it
        if f_hc is None:
            rmin = min_price_for_q(qd, comp)
            rmax = max_price_for_high_firm(qd, comp)
        else:
            rmin =
        return max_price_for_high_firm(f)

    elif f.q < comp.q:
        # Max price to have at least thetaMin
        return max_price_for_down_firm_for_consumer(f.q, comp, Params.thetaMin)

    else:
        # qd <= comp.q <= f.q

        if f.highest_cons is None:
            # f is entrant, all its quality is discounted, thus it is a down firm
            # Max price to have at least thetaMin
            return max_price_for_down_firm_for_consumer(qd, comp, Params.thetaMin)

        else:
            # two maximization brackets? one if higher and one if down
            # max price for richo or max down?
            return max_price_for_down_firm_for_consumer(q, comp, Params.thetaMin)


        if f.highest_cons >= richest_consumer():
            # f.highest_cons has a value and all market has tried f


        else:
            # f is incumbent and has an upper market not yet tried (f.highest_cons <= theta <= richest_consumer)
            # if higher q = q * d
            # if lower q = q (because higher limit should be at least thetaMin




    if q > comp.q:
        # f is higher. Max price is the one that have at least one expected consumer
        return Params.thetaMin * Params.population ** (1.0 / plambda(Params.gini)) * q ** Params.eta

    elif comp.q > q:
        # f is lower. the max price is the one that makes middle limit equal to thetaMin
        return comp.p - Params.thetaMin * (comp_q ** Params.eta - q ** Params.eta)

    else:
        # qd == qh. Price should be minor than comp.p
        return comp.p


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
    # when there is online one firm it is the highest

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
            mkt.down_firm = None
            return None, mkt.high_limit
        else:
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


def max_price_old(f: Firm, comp: Firm = None):
    # It is used for for optimization reasons
    # It is calculated as the maximum price that allows to have an expected quantity of at least 1

    if f.highest_cons is None
        effect_q = f.q * Params.discountQ
    else:
        effect_q = f.q
Falta ver que pasa con el discount
    if comp is None:
        return Params.thetaMin * Params.population ** (1.0 / plambda(Params.gini)) * effect_q ** Params.eta

    elif f.q <= comp.q:
        return comp.p
    elif f.q ** Params.discountQ
        return 0


def opt_price(f, comp=None):
    return minimize_scalar(min_profit, args=(f, comp), bracket=(min_price(f), max_price(f))).x


def iterate_price(f1, f2, max_iters=50):
    f1_seq = [f1]
    f1_res_seq = [calculate_result(f1, f2)]

    f2_seq = [f2]
    f2_res_seq = [calculate_result(f2, f1)]

    break_tolerance = 5
    for i in range(max_iters):
        # Optimize down firm
        tmp_f = copy(f1_seq[-1])
        tmp_f.p = opt_price(f1_seq[-1], f2_seq[-1])
        f1_seq.append(tmp_f)
        f1_res_seq.append(calculate_result(f1_seq[-1], f2_seq[-1]))

        # Optimize higher firm
        tmp_f = copy(f2_seq[-1])
        tmp_f.p = opt_price(f2_seq[-1], f1_seq[-1])
        f2_seq.append(tmp_f)
        f2_res_seq.append(calculate_result(f2_seq[-1], f1_seq[-1]))

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
