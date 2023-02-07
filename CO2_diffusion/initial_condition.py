from reaktoro_transport.problem import MassBalanceBase
from reaktoro_transport.manager import ReactionManager

class EquilibriumProblem(MassBalanceBase, ReactionManager):
    def set_activity_models(self):
        self.aqueous_phase.setChemicalModelHKF()

class BoundaryEquilibriumProblem(MassBalanceBase, ReactionManager):
    def set_chem_editor(self, database):
        editor = super().set_chem_editor(database)
        self.gaseous_phase = editor.addGaseousPhase(['CO2(g)'])

        return editor

    def set_activity_models(self):
        self.aqueous_phase.setChemicalModelHKF()
        self.aqueous_phase.setActivityModelDrummondCO2()
        self.gaseous_phase.setChemicalModelPengRobinson()

init_cond = EquilibriumProblem()
init_cond.set_components('K+', 'H+', 'OH-')
init_cond.set_solvent('H2O(l)')

init_cond.initialize_Reaktoro()
init_cond._set_temperature(298.15, 'K')
init_cond._set_pressure(1e5, 'Pa')
init_cond._set_species_amount([0.1, 1e-15, 0.1, 55.3])

init_cond.solve_chemical_equilibrium()

print(f"The KOH solution (0.1M) has the volume of {init_cond._get_fluid_volume()*1e3} Liters.")
print(f"The KOH solution has the density of {init_cond._get_fluid_density()} kg/m3.")

# Solves how much CO2 can dissolve into KCl solution
init_cond = BoundaryEquilibriumProblem()
init_cond.set_components('K+', 'Cl-', 'H+', 'OH-', 'CO2(aq)', 'HCO3-', 'CO3--')
init_cond.set_solvent('H2O(l)')

init_cond.initialize_Reaktoro()
init_cond._set_temperature(298.15, 'K')
init_cond._set_pressure(1e5, 'Pa')
init_cond._set_species_amount([0.2, 0.2, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 54.95, 100.0])

init_cond.solve_chemical_equilibrium()
dissolved_CO2_init_cond = init_cond._get_species_amounts()[:-1]

print(f"The KCl solution has the density of {init_cond._get_fluid_density()} kg/m3.")

init_cond = EquilibriumProblem()
init_cond.set_components('K+', 'Cl-', 'H+', 'OH-', 'CO2(aq)', 'HCO3-', 'CO3--')
init_cond.set_solvent('H2O(l)')

init_cond.initialize_Reaktoro()
init_cond._set_temperature(298.15, 'K')
init_cond._set_pressure(1e5, 'Pa')
init_cond._set_species_amount(dissolved_CO2_init_cond)

init_cond.solve_chemical_equilibrium()

print(f"The KCl solution (0.2M) has the volume of  {init_cond._get_fluid_volume()*1e3} Liters.")
print(init_cond._get_species_amounts())
