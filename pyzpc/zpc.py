from cmath import isinf
import numpy as np
import cvxpy as cp
from typing import Tuple, Callable, List, Optional, Union, Dict
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pydatadrivenreachability import (
    concatenate_zonotope,
    compute_IO_LTI_matrix_zonotope,
    MatrixZonotope,
    Zonotope,
    CVXZonotope)

from pyzpc.utils import (
    Data,
    DataDrivenDataset,
    SystemZonotopes,
    OptimizationProblem,
    OptimizationProblemVariables)
import sys
#sys.setrecursionlimit(10000)


class ZPC(object):
    optimization_problem: Union[OptimizationProblem,None] = None
    dataset: DataDrivenDataset
    zonotopes: SystemZonotopes
    Msigma: MatrixZonotope

    def __init__(self, data: Data):
        """
        Solves the ZPC optimization problem
        See also https://arxiv.org/pdf/2103.14110.pdf

        :param data:                A tuple of input/output data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        """
        self.update_identification_data(data)
    
    @property
    def num_samples(self) -> int:
        """ Return the number of samples used to estimate the Matrix Zonotope Msigma """
        return self.dataset.Um.shape[0] + 1

    @property
    def dim_u(self) -> int:
        """ Return the dimensionality of u (the control signal) """
        return self.dataset.Um.shape[1]

    @property
    def dim_y(self) -> int:
        """ Return the dimensionality of y (the output signal) """
        return self.dataset.Yp.shape[1]

    def update_identification_data(self, data: Data):
        """
        Update identification data matrices of ZPC. You need to rebuild the optimization problem
        after calling this funciton.

        :param data:                A tuple of input/output data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        """
        assert len(data.u.shape) == 2, \
            "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert len(data.y.shape) == 2, \
            "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert data.y.shape[0] == data.u.shape[0], \
            "Input/output data must have the same length"

        Ym = data.y[:-1]
        Yp = data.y[1:]
        Um = data.u[:-1]
        self.dataset = DataDrivenDataset(Yp, Ym, Um)
        self.optimization_problem = None

    def _build_zonotopes(self, zonotopes: SystemZonotopes):
        """
        [Private method] Do not invoke directly.
        Builds all the zonotopes needed to solve ZPC. 
        """
        X0, W, V, Av, U, Y = zonotopes.X0, zonotopes.W, zonotopes.V, zonotopes.Av, zonotopes.U, zonotopes.Y
        assert X0.dimension == W.dimension and X0.dimension == self.dim_y \
            and V.dimension == W.dimension and Av.dimension == V.dimension and Y.dimension == X0.dimension, \
            'The zonotopes do not have the correct dimension'
        
        self.zonotopes = zonotopes
        Mw = concatenate_zonotope(W, self.num_samples - 1)
        Mv = concatenate_zonotope(V, self.num_samples - 1)
        Mav = concatenate_zonotope(Av, self.num_samples - 1)

        self.Msigma = compute_IO_LTI_matrix_zonotope(self.dataset.Ym, self.dataset.Yp, self.dataset.Um, Mw, Mv, Mav)

    def build_problem(self,
            zonotopes: SystemZonotopes,
            horizon: int,
            build_loss: Callable[[cp.Variable, cp.Variable], Expression],
            build_constraints: Optional[Callable[[cp.Variable, cp.Variable], Optional[List[Constraint]]]] = None) -> OptimizationProblem:
        """
        Builds the ZPC optimization problem
        For more info check section 3.2 in https://arxiv.org/pdf/2103.14110.pdf

        :param zonotopes:           System zonotopes
        :param horizon:             Horizon length
        :param build_loss:          Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a scalar value of type Expression
        :param build_constraints:   Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a list of constraints.
        :return:                    Parameters of the optimization problem
        """
        assert build_loss is not None, "Loss function callback cannot be none"

        self.optimization_problem = None
        self._build_zonotopes(zonotopes)

        # Build variables
        y0 = cp.Parameter(shape=(self.dim_y))
        u = cp.Variable(shape=(horizon, self.dim_u))
        y = cp.Variable(shape=(horizon, self.dim_y))
        s_l = cp.Variable(shape=(horizon, self.dim_y), nonneg=True)
        s_u = cp.Variable(shape=(horizon, self.dim_y), nonneg=True)
        s = cp.Variable(shape=(horizon, self.dim_y), nonneg=True)
        beta_u = cp.Variable(shape=(horizon, self.zonotopes.U.num_generators))

        RL = cp.Variable(shape=(horizon, self.dim_y))
        RR = cp.Variable(shape=(horizon, self.dim_y))

        
        R = [CVXZonotope(y0, np.zeros((self.dim_y, 1)))]
        U = [CVXZonotope(u[i, :], np.zeros((self.dim_u, 1))) for i in range(horizon)]
        Z = self.zonotopes.W + self.zonotopes.V + (-1 *self.zonotopes.Av)
    
        
        
        leftY = self.zonotopes.Y.interval.left_limit
        rightY = self.zonotopes.Y.interval.right_limit

        constraints = [
            # y >= np.array([leftY.flatten()] * horizon),
            # y <= np.array([rightY.flatten()] * horizon),
            y - s_l >= np.array([leftY.flatten()] * horizon),
            y + s_u <= np.array([rightY.flatten()] * horizon),
            beta_u >= -1.,
            beta_u <= 1.,
            u == np.array([self.zonotopes.U.center] * horizon) + (beta_u @ self.zonotopes.U.generators.T),
        ]
        betas = []
        sys = self.Msigma.sample()[0]
        for i in range(horizon):
            print(f'Building for step {i}')
            # card_cen = cp.hstack([R[i].center, U[i].center])
            # card_gen = cp.vstack([R[i].generators, np.zeros((U[i].dimension, R[i].num_generators))])
            # card_zono = CVXZonotope(card_cen, card_gen)

            Rnew: CVXZonotope = (R[i].cartesian_product(U[i]) *  sys) + Z.sample()[0]
            R.append(Rnew)
            
            # beta_y_u = cp.Variable(shape=(Rnew.num_generators))
            # beta_y_l = cp.Variable(shape=(Rnew.num_generators))

            # betas.append(beta_y_u)
            # betas.append(beta_y_l)
            
            leftR = Rnew.interval.left_limit # Rnew.center - cp.sum(cp.abs(Rnew.generators), axis=1)
            rightR = Rnew.interval.right_limit #Rnew.center + cp.sum(cp.abs(Rnew.generators), axis=1) #Rnew.interval.right_limit

            # import pdb
            # pdb.set_trace()
            constraints.extend([
                #y[i] + s[i] == Rnew.center,
                leftR >= y[i]- s_l[i],
                rightR <= y[i] + s_l[i],
                #y[i] - s_l[i] ==  Rnew.center + Rnew.generators @ beta_y_l,
                #y[i] + s_u[i] ==  Rnew.center + Rnew.generators @ beta_y_u,
                # beta_y_u >= -1.,
                # beta_y_u <= 1.,
                # beta_y_l >= -1.,
                # beta_y_l <= 1
            ])

        _constraints = build_constraints(u, y) if build_constraints is not None else (None, None)

        for idx, constraint in enumerate(_constraints):
            if constraint is None or not isinstance(constraint, Constraint) or not constraint.is_dcp():
                raise Exception(f'Constraint {idx} is not defined or is not convex.')

        constraints.extend([] if _constraints is None else _constraints)

        # Build loss
        _loss = build_loss(u, y)
        
        if _loss is None or not isinstance(_loss, Expression) or not _loss.is_dcp():
            raise Exception('Loss function is not defined or is not convex!')

        _regularizers =  0
        problem_loss = _loss + _regularizers

        # Solve problem
        objective = cp.Minimize(problem_loss)

        try:
            problem = cp.Problem(objective, constraints)
        except cp.SolverError as e:
            raise Exception(f'Error while constructing the DeePC problem. Details: {e}')

        self.optimization_problem = OptimizationProblem(
            variables = OptimizationProblemVariables(y0=y0, u=u, y=y, s_l=s_l, s_u=s_u, beta_u=beta_u),
            constraints = constraints,
            objective_function = problem_loss,
            problem = problem
        )

        return self.optimization_problem

    def build_problem2(self,
            zonotopes: SystemZonotopes,
            horizon: int,
            build_loss: Callable[[cp.Variable, cp.Variable], Expression],
            build_constraints: Optional[Callable[[cp.Variable, cp.Variable], Optional[List[Constraint]]]] = None) -> OptimizationProblem:
        """
        Builds the ZPC optimization problem
        For more info check section 3.2 in https://arxiv.org/pdf/2103.14110.pdf

        :param zonotopes:           System zonotopes
        :param horizon:             Horizon length
        :param build_loss:          Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a scalar value of type Expression
        :param build_constraints:   Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a list of constraints.
        :return:                    Parameters of the optimization problem
        """
        assert build_loss is not None, "Loss function callback cannot be none"

        self.optimization_problem = None
        self._build_zonotopes(zonotopes)

        # Build variables
        y0 = cp.Parameter(shape=(self.dim_y))
        u = cp.Variable(shape=(horizon, self.dim_u))
        y = cp.Variable(shape=(horizon, self.dim_y))
        s_l = cp.Variable(shape=(horizon, self.dim_y), nonneg=True)
        s_u = cp.Variable(shape=(horizon, self.dim_y), nonneg=True)
        beta_u = cp.Variable(shape=(horizon, self.zonotopes.U.num_generators))

        
        U = [CVXZonotope(u[i, :], np.zeros((self.dim_u, 1))) for i in range(horizon)]
        Z = self.zonotopes.W + self.zonotopes.V + (-1 *self.zonotopes.Av)
    
        leftY = self.zonotopes.Y.interval.left_limit
        rightY = self.zonotopes.Y.interval.right_limit

        constraints = [
            y - s_l >= np.array([leftY.flatten()] * horizon),
            y + s_u <= np.array([rightY.flatten()] * horizon),
            #beta_u >= -1.,
            #beta_u <= 1.,
            u >= np.array([self.zonotopes.U.interval.left_limit.flatten()] * horizon),
            u <= np.array([self.zonotopes.U.interval.right_limit.flatten()] * horizon)
            #u == np.array([self.zonotopes.U.center] * horizon) + (beta_u @ self.zonotopes.U.generators.T),
        ]

        num_trajectories = 1
        R: List[List[CVXZonotope]] = []
        for i in range(num_trajectories):
            R.append([CVXZonotope(y0, np.zeros((self.dim_y, 1)))])

        #Rhat = [Zonotope(y0, np.zeros((self.dim_y, 1)))]
        #Usampled = self.zonotopes.U.sample(num_trajectories * horizon).reshape((num_trajectories, horizon, self.dim_u))

        for k in range(num_trajectories):
            sys_sample: np.ndarray = self.Msigma.sample()[0]
            print(k)
            for i in range(horizon):
                print(f'{k}-{i}')
                card_cen = cp.hstack([R[k][i].center, U[i].center])
                card_gen = cp.vstack([R[k][i].generators, np.zeros((U[i].dimension, R[k][i].num_generators))])
                card_zono = CVXZonotope(card_cen, card_gen)
                R_ki: CVXZonotope = ((R[k][i].cartesian_product(U[i])) * sys_sample) + Z.sample()[0]
                #R_ki: CVXZonotope = (card_zono * sys_sample) + Z.sample()[0]
                #R_ki: CVXZonotope = (card_zono * sys_sample) + Z #.sample()[0]
                R[k].append(R_ki)

                leftR = R_ki.interval.left_limit # Rnew.center - cp.sum(cp.abs(Rnew.generators), axis=1)
                rightR = R_ki.interval.right_limit #Rnew.center + cp.sum(cp.abs(Rnew.generators), axis=1) #Rnew.interval.right_limit

                constraints.extend([
                    leftR >= y[i]- s_l[i],
                    rightR <= y[i] + s_l[i],
                ])


        # for i in range(horizon):
        #     print(i)
        #     import pdb
        #     pdb.set_trace()
        #     Ravg = None
        #     sys_sample = self.Msigma.sample()[0]
        #     for k in range(num_trajectories):
        #         Zu = Zonotope(Usampled[k,i], np.zeros((self.dim_u, 1)))
        #         R_ki = sys_sample * (R[k][i].cartesian_product(Zu)) + Z
        #         R[k].append(R_ki.reduce(50))
        #         Ravg =   Ravg + R_ki if Ravg is not None else R_ki
        #         Ravg.reduce(50)
        #         Ravg = Ravg.over_approximate()
            #Ravg = Zonotope(Ravg.center * (1/num_trajectories), Ravg.generators)
            
            #Rhat.append(Ravg)

        # for i in range(horizon):
        #     Rnew: CVXZonotope = self.Msigma * (U[i].cartesian_product(Rhat[i])) + Z

        #     leftR = Rnew.interval.left_limit # Rnew.center - cp.sum(cp.abs(Rnew.generators), axis=1)
        #     rightR = Rnew.interval.right_limit #Rnew.center + cp.sum(cp.abs(Rnew.generators), axis=1) #Rnew.interval.right_limit

        #     constraints.extend([
        #         leftR >= y[i]- s_l[i],
        #         rightR <= y[i] + s_l[i],
        #     ])


        _constraints = build_constraints(u, y) if build_constraints is not None else (None, None)

        for idx, constraint in enumerate(_constraints):
            if constraint is None or not isinstance(constraint, Constraint) or not constraint.is_dcp():
                raise Exception(f'Constraint {idx} is not defined or is not convex.')

        constraints.extend([] if _constraints is None else _constraints)

        # Build loss
        _loss = build_loss(u, y)
        
        if _loss is None or not isinstance(_loss, Expression) or not _loss.is_dcp():
            raise Exception('Loss function is not defined or is not convex!')

        _regularizers =  0
        problem_loss = _loss + _regularizers

        # Solve problem
        objective = cp.Minimize(problem_loss)

        try:
            problem = cp.Problem(objective, constraints)
        except cp.SolverError as e:
            raise Exception(f'Error while constructing the DeePC problem. Details: {e}')

        self.optimization_problem = OptimizationProblem(
            variables = OptimizationProblemVariables(y0=y0, u=u, y=y, s_l=s_l, s_u=s_u, beta_u=beta_u),
            constraints = constraints,
            objective_function = problem_loss,
            problem = problem
        )

        return self.optimization_problem

    def solve(
            self,
            y0: np.ndarray,
            **cvxpy_kwargs
        ) -> Tuple[np.ndarray, Dict[str, Union[float, np.ndarray, OptimizationProblemVariables]]]:
        """
        Solves the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        :param y0:                  The initial output
        :param cvxpy_kwargs:        All arguments that need to be passed to the cvxpy solve method.
        :return u_optimal:          Optimal input signal to be applied to the system, of length `horizon`
        :return info:               A dictionary with 5 keys:
                                    info['variables']: variables of the optimization problem
                                    info['value']: value of the optimization problem
                                    info['u_optimal']: the same as the first value returned by this function
        """
        assert len(y0) == self.dim_y, f"Invalid size"
        assert self.optimization_problem is not None, "Problem was not built"


        self.optimization_problem.variables.y0.value = y0
        try:
            #import pdb
            #pdb.set_trace()
            result = self.optimization_problem.problem.solve(**cvxpy_kwargs)
        except cp.SolverError as e:
            with open('zpc_logs.txt', 'w') as f:
                print(f'Error while solving the DeePC problem. Details: {e}', file=f)
            raise Exception(f'Error while solving the DeePC problem. Details: {e}')

        if np.isinf(result):
            raise Exception('Problem is unbounded')

        u_optimal = self.optimization_problem.variables.u.value
        info = {
            'value': result, 
            'variables': self.optimization_problem.variables,
            'u_optimal': u_optimal
            }

        return u_optimal, info