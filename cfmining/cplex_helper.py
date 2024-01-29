# This file contains convenience functions for CPLEX MIP Objects
import numpy as np
from functools import reduce
from cplex import Cplex, SparsePair


# Parameter Setting
DEFAULT_CPLEX_PARAMETERS = {
    #
    "display_cplex_progress": False,
    # set to True to show CPLEX progress in console
    #
    "n_cores": 1,
    # Number of CPU cores to use in B & B
    # May have to set n_cores = 1 in order to use certain control callbacks in CPLEX 12.7.0 and earlier
    #
    "randomseed": 0,
    # This parameter sets the random seed differently for diversity of solutions.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/RandomSeed.html
    #
    "time_limit": 1e75,
    # runtime before stopping,
    #
    "node_limit": 9223372036800000000,
    # number of nodes to process before stopping,
    #
    "mipgap": np.finfo("float").eps,
    # Sets a relative tolerance on the gap between the best integer objective and the objective of the best node remaining.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpGap.html
    #
    "absmipgap": np.finfo("float").eps,
    # Sets an absolute tolerance on the gap between the best integer objective and the objective of the best node remaining.
    # When this difference falls below the value of this parameter, the mixed integer optimization is stopped.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpAGap.html
    #
    "objdifference": 0.0,
    # Used to update the cutoff each time a mixed integer solution is found. This value is subtracted from objective
    # value of the incumbent update, so that the solver ignore solutions that will not improve the incumbent by at
    # least this amount.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ObjDif.html#
    #
    "integrality_tolerance": 0.0,
    # specifies the amount by which an variable can differ from an integer and be considered integer feasible. 0 is OK
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpInt.html
    #
    "mipemphasis": 0,
    # Controls trade-offs between speed, feasibility, optimality, and moving bounds in MIP.
    # 0     =	Balance optimality and feasibility; default
    # 1	    =	Emphasize feasibility over optimality
    # 2	    =	Emphasize optimality over feasibility
    # 3 	=	Emphasize moving best bound
    # 4	    =	Emphasize finding hidden feasible solutions
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/MIPEmphasis.html
    #
    "bound_strengthening": -1,
    # Decides whether to apply bound strengthening in mixed integer programs (MIPs).
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/BndStrenInd.html
    # -1    = cplex chooses
    # 0     = no bound strengthening
    # 1     = bound strengthening
    #
    "cover_cuts": -1,
    # Decides whether or not cover cuts should be generated for the problem.
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/Covers.html
    # -1    = Do not generate cover cuts
    # 0	    = Automatic: let CPLEX choose
    # 1	    = Generate cover cuts moderately
    # 2	    = Generate cover cuts aggressively
    # 3     = Generate cover cuts very  aggressively
    #
    "zero_half_cuts": -1,
    # Decides whether or not to generate zero-half cuts for the problem. (set to off since these are not effective)
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ZeroHalfCuts.html
    # -1    = Do not generate MIR cuts
    # 0	    = Automatic: let CPLEX choose
    # 1	    = Generate MIR cuts moderately
    # 2	    = Generate MIR cuts aggressively
    #
    "mir_cuts": -1,
    # Decides whether or not to generate mixed-integer rounding cuts for the problem. (set to off since these are not effective)
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/MIRCuts.html
    # -1    = Do not generate zero-half cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate zero-half cuts moderately
    # 2	    = Generate zero-half cuts aggressively
    #
    "implied_bound_cuts": 0,
    # Decides whether or not to generate valid implied bound cuts for the problem.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ImplBdLocal.html
    # -1    = Do not generate locally valid implied bound cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate locally valid implied bound cuts moderately
    # 2	    = Generate locally valid implied bound cuts aggressively
    # 3	    = Generate locally valid implied bound cuts very aggressively
    #
    "locally_implied_bound_cuts": 3,
    # Decides whether or not to generate locally valid implied bound cuts for the problem.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ImplBdLocal.html
    # -1    = Do not generate locally valid implied bound cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate locally valid implied bound cuts moderately
    # 2	    = Generate locally valid implied bound cuts aggressively
    # 3	    = Generate locally valid implied bound cuts very aggressively
    #
    "scale_parameters": 0,
    # Decides how to scale the problem matrix.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ScaInd.html
    # 0     = equilibration scaling
    # 1     = aggressive scaling
    # -1    = no scaling
    #
    "numerical_emphasis": 0,
    # Emphasizes precision in numerically unstable or difficult problems.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/NumericalEmphasis.html
    # 0     = off
    # 1     = on
    #
    "poolsize": 100,
    # Limits the number of solutions kept in the solution pool
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolCapacity.html
    # number of feasible solutions to keep in solution pool
    #
    "poolrelgap": float("nan"),
    # Sets a relative tolerance on the objective value for the solutions in the solution pool.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolGap.html
    #
    "poolreplace": 2,
    # Designates the strategy for replacing a solution in the solution pool when the solution pool has reached its capacity.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolReplace.html
    # 0	= Replace the first solution (oldest) by the most recent solution; first in, first out; default
    # 1	= Replace the solution which has the worst objective
    # 2	= Replace solutions in order to build a set of diverse solutions
    #
    "repairtries": 20,
    # Limits the attempts to repair an infeasible MIP start.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/RepairTries.html
    # -1	None: do not try to repair
    #  0	Automatic: let CPLEX choose; default
    #  N	Number of attempts
    #
    "nodefilesize": (120 * 1024) / 1,
    # size of the node file (for large scale problems)
    # if the B & B can no longer fit in memory, then CPLEX stores the B & B in a node file
}


def set_cpx_display_options(
    cpx, display_mip=True, display_parameters=False, display_lp=False
):
    """
    Convenience function to turn on/off CPLEX functions
    :param cpx:
    :param display_mip:
    :param display_parameters:
    :param display_lp:
    :return:
    """
    cpx.parameters.mip.display.set(display_mip)
    cpx.parameters.simplex.display.set(display_lp)
    cpx.parameters.paramdisplay.set(display_parameters)

    if not (display_mip or display_lp):
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        cpx.set_error_stream(None)
        cpx.set_warning_stream(None)

    return cpx


def set_cpx_parameters(cpx, param=DEFAULT_CPLEX_PARAMETERS):
    """
    Set parameters of a Cplex object
    :param cpx: Cplex object
    :param param: dictionary of parameters
    :return: cpx
    """

    # get parameter handle
    p = cpx.parameters

    # Record calls to C API
    # cpx.parameters.record.set(True)

    if param["display_cplex_progress"] is (None or False):
        cpx = set_cpx_display_options(
            cpx, display_mip=False, display_lp=False, display_parameters=False
        )

    # major parameters
    p.randomseed.set(param["randomseed"])
    p.threads.set(param["n_cores"])
    p.output.clonelog.set(0)
    p.parallel.set(1)

    # solution strategy
    p.emphasis.mip.set(param["mipemphasis"])
    p.preprocessing.boundstrength.set(param["bound_strengthening"])

    # cuts
    p.mip.cuts.implied.set(param["implied_bound_cuts"])
    p.mip.cuts.localimplied.set(param["locally_implied_bound_cuts"])
    p.mip.cuts.zerohalfcut.set(param["zero_half_cuts"])
    p.mip.cuts.mircut.set(param["mir_cuts"])
    p.mip.cuts.covers.set(param["cover_cuts"])
    #
    # tolerances
    p.emphasis.numerical.set(param["numerical_emphasis"])
    p.mip.tolerances.integrality.set(param["integrality_tolerance"])

    # initialization
    p.mip.limits.repairtries.set(param["repairtries"])

    # solution pool
    p.mip.pool.capacity.set(param["poolsize"])
    p.mip.pool.replace.set(param["poolreplace"])

    # stopping
    p.mip.tolerances.mipgap.set(param["mipgap"])
    p.mip.tolerances.absmipgap.set(param["absmipgap"])

    if param["time_limit"] < DEFAULT_CPLEX_PARAMETERS["time_limit"]:
        cpx = set_cpx_time_limit(cpx, param["time_limit"])

    if param["node_limit"] < DEFAULT_CPLEX_PARAMETERS["node_limit"]:
        cpx = set_cpx_node_limit(cpx, param["node_limit"])

    return cpx


def set_cpx_time_limit(cpx, time_limit=None):
    """
    Convenience function to set a time limit on a Cplex object
    :param cpx: Cplex object
    :param time_limit: time limit in seconds
    :return: cpx: Cplex object
    """
    max_time_limit = float(cpx.parameters.timelimit.max())

    if time_limit is None:
        time_limit = max_time_limit
    else:
        time_limit = float(time_limit)
        time_limit = min(time_limit, max_time_limit)

    assert time_limit >= 0.0
    cpx.parameters.timelimit.set(time_limit)
    return cpx


def set_cpx_node_limit(cpx, node_limit=None):
    """
    Convenience function to set a node limit on a Cplex object.
    The node limit determines the maximum number of nodes that can be solved in
    branch and bound.
    :param cpx: Cplex object
    :param node_limit: time limit in seconds
    :return: cpx: Cplex object
    """
    max_node_limit = cpx.parameters.mip.limits.nodes.max()
    if node_limit == float("inf"):
        node_limit = max_node_limit
    elif node_limit is None:
        node_limit = max_node_limit
    else:
        node_limit = int(node_limit)
        node_limit = min(node_limit, max_node_limit)
    assert node_limit >= 0.0
    cpx.parameters.mip.limits.nodes.set(node_limit)
    return cpx
