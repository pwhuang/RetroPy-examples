# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import sys
from problem import Problem

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([3.0e-9]*self.num_component)

problem = Problem(nx=150, ny=25, const_diff=True)
problem.generate_output_instance(sys.argv[1])

problem.define_problem()
problem.setup_flow_solver(r_val=1e11, omega_by_r=1.0)
problem.setup_reaction_solver()
problem.setup_auxiliary_reaction_solver()
problem.setup_transport_solver()

time_stamps = []
problem.solve(dt_val=86400.0*5.0, endtime=86400.0*365*5, time_stamps=time_stamps)
