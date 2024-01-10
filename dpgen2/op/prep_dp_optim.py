import json
import shutil
import pickle
from pathlib import (
    Path,
)
from typing import (
    List,
    Tuple,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
)

from dpgen2.constants import (
    calypso_run_opt_file,
    calypso_check_opt_file,
)
from dpgen2.exploration.task import (
    ExplorationTaskGroup,
)
from dpgen2.utils import (
    BinaryFileInput,
    set_directory,
)


class PrepDPOptim(OP):
    r"""Prepare the working directories and input file for structure optimization with DP.

    `POSCAR_*`, `model.000.pb`, `calypso_run_opt.py` and `calypso_check_opt.py` will be copied
    or symlink to each optimization directory from `ip["work_path"]`, according to the
    popsize `ip["caly_input"]["PopSize"]`.
    The paths of these optimization directory will be returned as `op["optim_paths"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "caly_input": dict,  # calypso input params
                "work_dir": Artifact(Path),  # the directory where the structures are in
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                # "work_dir": Artifact(Path),  # the directory where the structures are in
                "optim_names": List[str],
                "optim_paths": Artifact(List[Path]),  # each optim_paths containing one structure and related file optim needed.
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
            - `caly_input` : (`dict`) Definitions for CALYPSO input file.
            - `work_dir` : (`Path`) The directory where the structures are in.

        Returns
        -------
        op : dict
            Output dict with components:

            - `optim_names`: (`List[str]`) The name of optim tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `optim_paths`: (`Artifact(List[Path])`) The parepared optim paths of the task containing input files (`calypso_run_opt.py` and `calypso_check_opt.py`, `frozen_model.pb`) needed to optimize structure by DP.
        """

        caly_input = ip["caly_input"]
        popsize = caly_input.get("PopSize", 30)

        optim_paths = []
        work_dir = ip["work_dir"]
        with set_directory(work_dir):
            for pop in range(popsize):
                optim_path = _cp_file_to_optim(pop)
                optim_paths.append(work_dir.joinpath(optim_path))

            optim_names = [str(ii) for ii in optim_paths]

        return OPIO(
            {
                "optim_names": optim_names,
                "optim_paths": optim_paths,
            }
        )

def _cp_file_to_optim(pop):

    optim_path = Path(f"pop{str(pop)}")
    optim_path.mkdir(parents=True, exist_ok=True)

    # calypso_run_opt.py calypso_check_opt.py model.000.pb POSCAR_pop
    shutil.copyfile(f"POSCAR_{str(pop)}", optim_path.joinpath("POSCAR"))
    shutil.copyfile(calypso_run_opt_file, optim_path.joinpath(calypso_run_opt_file))
    shutil.copyfile(calypso_check_opt_file, optim_path.joinpath(calypso_check_opt_file))
    Path("model.000.pb").symlink_to(optim_path.joinpath("frozen_model.pb"))

    return optim_path
