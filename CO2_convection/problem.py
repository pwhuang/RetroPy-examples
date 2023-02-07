# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from mesh_factory import MeshFactory
from retropy.manager import DarcyFlowManagerUzawa as FlowManager
from retropy.manager import ReactiveTransportManager
from retropy.manager import HDF5Manager as OutputManager
from retropy.solver import TransientNLSolver

from retropy.problem import MassBalanceBase
from retropy.manager import ReactionManager

from dolfin import Expression, Constant

class FlowManager(FlowManager):
    def set_flow_fe_space(self):
        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('BDM', 1)

class BoundaryEquilibriumProblem(MassBalanceBase, ReactionManager):
    def __init__(self, component, solvent):
        self.set_components(*component)
        self.set_solvent(solvent)
        self.H_idx = self.component_dict['H+']

    def set_chem_editor(self, database):
        editor = super().set_chem_editor(database)
        self.gaseous_phase = editor.addGaseousPhase(['CO2(g)'])

        return editor

    def set_activity_models(self):
        self.aqueous_phase.setChemicalModelHKF()
        self.aqueous_phase.setActivityModelDrummondCO2()
        self.gaseous_phase.setChemicalModelPengRobinson()

    def _assign_chem_equi_results(self):
        self.fluid_density.vector()[:] = self.rho_temp  # kg/m3
        self.fluid_pH.vector()[:] = self.pH_temp
        self.ln_activity.vector()[:] = self.lna_temp[:, :-1].flatten()

        self.fluid_components.vector()[:] = self.molar_density_temp[:, :-1].flatten()
        self.solvent.vector()[:] = self.molar_density_temp[:, -1].flatten()

class ReactiveTransportManager(ReactiveTransportManager, MeshFactory):
    def __init__(self, nx, ny, const_diff):
        super().__init__(*self.get_mesh_and_markers(nx, ny))
        self.is_same_diffusivity = const_diff
        self.total_gaseous_CO2_amount = 1e8 # mols

    def setup_auxiliary_reaction_solver(self):
        self.aux_equi_problem = BoundaryEquilibriumProblem(list(self.component_dict.keys()),
                                                           self.solvent_name)

        self.aux_equi_problem.initialize_Reaktoro()

        reservoir_depth = 1.5e3
        self.background_pressure = 1e5 + 1e3*9.80665*reservoir_depth # Pa

        self.aux_equi_problem._set_temperature(298.15, 'K')
        self.aux_equi_problem._set_pressure(self.background_pressure, 'Pa')

    def set_dof_idx(self):
        self.boundary_cell_idx, self.dof_idx = self.mark_inflow_boundary_cells()

    def set_activity_models(self):
        self.aqueous_phase.setChemicalModelHKF()
        self.aqueous_phase.setActivityModelDrummondCO2()

    def _assign_chem_equi_results(self):
        self.fluid_density.vector()[:] = self.rho_temp #kg/m3
        self.fluid_pH.vector()[:] = self.pH_temp
        self.ln_activity.vector()[:] = self.lna_temp[:, :-1].flatten()

        self.fluid_components.vector()[:] = self.molar_density_temp[:, :-1].flatten()
        self.solvent.vector()[:] = self.molar_density_temp[:, -1].flatten()

    def _solve_chem_equi_over_dofs(self, pressure, fluid_comp):
        super()._solve_chem_equi_over_dofs(pressure, fluid_comp)

        for i in self.boundary_cell_idx:
            self.aux_equi_problem._set_species_amount(list(fluid_comp[i]) + \
                                                      [self.solvent.vector()[i]] + \
                                                      [self.total_gaseous_CO2_amount])

            self.aux_equi_problem.solve_chemical_equilibrium()

            self.rho_temp[i] = self.aux_equi_problem._get_fluid_density()
            self.pH_temp[i] = self.aux_equi_problem._get_fluid_pH()
            self.lna_temp[i] = self.aux_equi_problem._get_species_log_activity_coeffs()[:-1]
            self.molar_density_temp[i] = self.aux_equi_problem._get_species_amounts()[:-1]

class Problem(ReactiveTransportManager, FlowManager, OutputManager,
              TransientNLSolver):
    """This class solves the CO2 convection problem."""

    def __init__(self, nx, ny, const_diff):
        super().__init__(nx, ny, const_diff)
        self.set_flow_residual(1e-8)
        self.set_flow_max_steps(50)

    def set_component_properties(self):
        self.set_molar_mass([6.941, 1.00794, 17.00734, 44.0095, 60.0089, 61.01684]) #g/mol
        self.set_solvent_molar_mass(18.0153)
        self.set_charge([1.0, 1.0, -1.0, 0.0, -2.0, -1.0])

    def define_problem(self):
        self.set_components('Li+', 'H+', 'OH-', 'CO2(aq)', 'CO3--', 'HCO3-')
        self.set_solvent('H2O(l)')
        self.set_component_properties()

        self.set_component_fe_space()
        self.initialize_form()

        LiOH_amounts = [0.01, 1e-15, 0.01, 1e-15, 1e-15, 1e-15, 55.345] # kmol/m^3 # mol/L

        init_expr_list = []

        for i in range(self.num_component):
            init_expr_list.append(str(LiOH_amounts[i]))

        self.set_component_ics(Expression(init_expr_list, degree=1))
        self.set_solvent_ic(Expression(str(LiOH_amounts[-1]), degree=1))

    def set_fluid_properties(self):
        self.set_porosity(0.1)
        self.set_fluid_density(1e3) # Initialization # kg/m^3
        self.set_fluid_viscosity(0.893e-3)  # Pa sec
        self.set_gravity([0.0, -9.80665]) # m/sec^2
        self.set_permeability(1e-13) # m^2

    def set_flow_ibc(self):
        self.mark_flow_boundary(pressure = [],
                                velocity = [self.marker_dict['top'], self.marker_dict['bottom'],
                                            self.marker_dict['left'], self.marker_dict['right']])

        self.set_pressure_bc([]) # Pa
        self.set_pressure_ic(Constant(0.0))
        self.set_velocity_bc([Constant([0.0, 0.0])]*4)

    @staticmethod
    def timestepper(dt_val, current_time, time_stamp):
        min_dt, max_dt = 1e3, 86400.0*7.0

        if (dt_val := dt_val*1.1) > max_dt:
            dt_val = max_dt
        elif dt_val < min_dt:
            dt_val = min_dt
        if dt_val > time_stamp - current_time:
            dt_val = time_stamp - current_time

        return dt_val
