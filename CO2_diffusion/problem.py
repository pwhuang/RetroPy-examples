import os
os.environ['OMP_NUM_THREADS'] = '1'

from mesh_factory import MeshFactory
from reaktoro_transport.manager import DarcyFlowManagerUzawa as FlowManager
from reaktoro_transport.manager import ReactiveTransportManager
from reaktoro_transport.manager import XDMFManager as OutputManager
from reaktoro_transport.solver import TransientNLSolver

from reaktoro_transport.problem import MassBalanceBase
from reaktoro_transport.manager import ReactionManager

from dolfin import Expression, Constant

class ReactiveTransportManager(ReactiveTransportManager, MeshFactory):
    def __init__(self, nx, ny, const_diff):
        super().__init__(*self.get_mesh_and_markers(nx, ny))
        self.is_same_diffusivity = const_diff

    def set_activity_models(self):
        self.aqueous_phase.setChemicalModelHKF()
        self.aqueous_phase.setActivityModelDrummondCO2()

class Problem(ReactiveTransportManager, FlowManager, OutputManager,
              TransientNLSolver):
    """This class solves the CO2 convection problem."""

    def __init__(self, nx, ny, const_diff):
        super().__init__(nx, ny, const_diff)
        self.set_flow_residual(5e-10)

    def set_component_properties(self):
        self.set_molar_mass([39.0983, 35.453, 1.00794, 17.00734, 44.0095, 60.0089, 61.01684]) #g/mol
        self.set_solvent_molar_mass(18.0153)
        self.set_charge([1.0, -1.0, 1.0, -1.0, 0.0, -2.0, -1.0])

    def define_problem(self):
        self.set_components('K+', 'Cl-', 'H+', 'OH-', 'CO2(aq)', 'CO3--', 'HCO3-')
        self.set_solvent('H2O(l)')
        self.set_component_properties()

        self.set_component_fe_space()
        self.initialize_form()

        self.background_pressure = 1e5 #+ 1e-3*9806.65*20.0 # Pa

        KCl_amounts = [0.2, 0.5, 1.67316928e-04, 1.09822824e-10,
                       3.18741118e-02, 1.67316504e-04, 1.57238962e-10, 5.49498327e+01]

        KOH_amounts = [0.3, 1e-15, 1e-15, 0.3, 1e-15, 1e-15, 1e-15, 55.3]

        init_expr_list = []

        condition = """x[1]<=64.0 && x[1]>=36.0 && x[0]>=36.0 && x[0]<=64.0 ?"""
        condition = """
                    sqrt(pow(x[0]-50.0, 2) + pow(x[1]-50.0, 2)) < 20.0 &&
                    sqrt(pow(x[0]-50.0, 2) + pow(x[1]-50.0, 2)) > 6.0 ?
                    """

        for i in range(self.num_component):
            init_expr_list.append(condition + \
                                  str(KCl_amounts[i]) + ':' + str(KOH_amounts[i]))

        self.set_component_ics(Expression(init_expr_list, degree=1))
        self.set_solvent_ic(Expression(condition + \
                                       str(KCl_amounts[-1]) + ':' + str(KOH_amounts[-1]), degree=1))

    def set_fluid_properties(self):
        self.set_porosity(1.0)
        self.set_fluid_density(1e-3) # Initialization # g/mm^3
        self.set_fluid_viscosity(0.893e-3)  # Pa sec
        self.set_gravity([0.0, 0.0]) # -9806.65 mm/sec
        self.set_permeability(0.5**2/12.0) # mm^2

    def set_flow_ibc(self):
        self.mark_flow_boundary(pressure = [],
                                velocity = [self.marker_dict['top'], self.marker_dict['bottom'],
                                            self.marker_dict['left'], self.marker_dict['right']])

        self.set_pressure_bc([]) # Pa
        self.set_pressure_ic(Constant(0.0))
        self.set_velocity_bc([Constant([0.0, 0.0])]*4)

    def solve_flow(self, *args, **kwargs):
        pass

    @staticmethod
    def timestepper(dt_val, current_time, time_stamp):
        min_dt, max_dt = 5e-3, 30.0

        if (dt_val := dt_val*1.4) > max_dt:
            dt_val = max_dt
        elif dt_val < min_dt:
            dt_val = min_dt
        if dt_val > time_stamp - current_time:
            dt_val = time_stamp - current_time

        return dt_val
