import sys
from problem import Problem

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([1.957e-3, 2.032e-3, 9.311e-3, 5.273e-3,
                                        2.045e-3, 0.804e-3, 1.101e-3]) #mm^2/sec

problem = Problem(nx=30, ny=30, const_diff=False)
problem.generate_output_instance(sys.argv[1])

problem.define_problem()
problem.setup_flow_solver(r_val=8e4, omega_by_r=1.0)
problem.setup_reaction_solver()
problem.setup_transport_solver()

time_stamps = []
problem.solve(dt_val=0.1, endtime=2e4, time_stamps=time_stamps)
