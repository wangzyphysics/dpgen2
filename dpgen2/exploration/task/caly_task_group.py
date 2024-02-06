import itertools
import random
from typing import (
    List,
    Optional,
)

from dpgen2.constants import (
    calypso_input_file,
    lmp_conf_name,
    lmp_input_name,
    model_name_pattern,
)

from .calypso import (
    make_calypso_input,
)
from .task import (
    ExplorationTask,
)
from .task_group import (
    BaseExplorationTaskGroup,
    ExplorationTaskGroup,
)


class CalyTaskGroup(ExplorationTaskGroup):
    def __init__(
        self,
    ):
        super().__init__()

    def set_params(
        self,
        numb_of_species,
        name_of_atoms,
        atomic_number,
        numb_of_atoms,
        distance_of_ions,
        pop_size: int = 30,
        max_step: int = 5,
        system_name: str = "CALYPSO",
        numb_of_formula: List[int] = [1, 1],
        pressure: float = 0.001,
        fmax: float = 0.01,
        volume: float = 0,
        ialgo: int = 2,
        pso_ratio: float = 0.6,
        icode: int = 15,
        numb_of_lbest: int = 4,
        numb_of_local_optim: int = 4,
        command: str = "sh submit.sh",
        max_time: int = 9000,
        gen_type: int = 1,
        pick_up: bool = False,
        pick_step: int = 1,
        parallel: bool = False,
        split: bool = True,
        spec_space_group: List[int] = [2, 230],
    ):
        """
        Set calypso parameters
        """
        self.numb_of_species = numb_of_species
        self.name_of_atoms = name_of_atoms
        self.atomic_number = atomic_number
        self.numb_of_atoms = numb_of_atoms
        self.distance_of_ions = distance_of_ions
        self.pop_size = pop_size
        self.max_step = max_step
        self.system_name = system_name
        self.numb_of_formula = numb_of_formula
        self.pressure = pressure
        self.fmax = fmax
        self.volume = volume
        self.ialgo = ialgo
        self.pso_ratio = pso_ratio
        self.icode = icode
        self.numb_of_lbest = numb_of_lbest
        self.numb_of_local_optim = numb_of_local_optim
        self.command = command
        self.max_time = max_time
        self.gen_type = gen_type
        self.pick_up = pick_up
        self.pick_step = pick_step
        self.parallel = parallel
        self.split = split
        self.spec_space_group = spec_space_group

        self.caly_set = True

    def make_task(
        self,
    ) -> ExplorationTaskGroup:
        """
        Make the CALYPSO task group.

        Returns
        -------
        task_grp: ExplorationTaskGroup
            Return one calypso task group.
        """
        if not self.caly_set:
            raise RuntimeError("calypso settings are not set")
        # clear all existing tasks
        self.clear()
        self.add_task(self._make_caly_task())
        return self

    def _make_caly_task(self) -> ExplorationTask:
        task = ExplorationTask()
        task.add_file(
            calypso_input_file,
            make_calypso_input(
                lmp_conf_name,
                self.ens,
                self.graphs,
                self.nsteps,
                self.dt,
                self.neidelay,
                self.trj_freq,
                self.mass_map,
                self.tau_t,
                self.tau_p,
                self.use_clusters,
                self.relative_f_epsilon,
                self.relative_v_epsilon,
                self.pka_e,
                self.ele_temp_f,
                self.ele_temp_a,
                self.no_pbc,
                trj_seperate_files=False,
            ),
        )
        return task
