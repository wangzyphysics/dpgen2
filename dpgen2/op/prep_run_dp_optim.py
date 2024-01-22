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
    calypso_opt_dir_name,
    model_name_pattern,
)
from dpgen2.exploration.task import (
    ExplorationTaskGroup,
)
from dpgen2.utils import (
    BinaryFileInput,
    set_directory,
)
from dpgen2.utils.run_command import (
    run_command,
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
                "caly_structure_path_name": Artifact(
                    Path
                ),  # the directory where the structures are in
                "input_file_path_name": Artifact(
                    Path
                ),  # the models, scripts location. (prep_caly_input)
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                # "work_dir": Artifact(Path),  # the directory where the structures are in
                "optim_names": List[str],
                "optim_paths": Artifact(
                    List[Path]
                ),  # each optim_paths containing one structure and related file optim needed.
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
            - `caly_structure_path_name` : (`str`) The directory where the structures are in.
            - `input_file_path_name` : (`str`)

        Returns
        -------
        op : dict
            Output dict with components:

            - `optim_names`: (`List[str]`) The name of optim tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `optim_paths`: (`Artifact(List[Path])`) The parepared optim paths of the task containing input files (`calypso_run_opt.py` and `calypso_check_opt.py`, `frozen_model.pb`) needed to optimize structure by DP.
        """
        caly_input = ip["caly_input"]
        popsize = caly_input.get("PopSize", 30)

        work_dir = ip["caly_structure_path_name"]
        poscar_str = "POSCAR_%d"
        poscar_list = [
            Path(work_dir).joinpath(poscar_str % num).resolve()
            for num in range(1, popsize + 1)
        ]

        prep_calypso_work_dir = ip["input_file_path_name"]
        model_file = (
            Path(prep_calypso_work_dir).joinpath(model_name_pattern % 0).resolve()
        )
        calypso_run_opt_script = (
            Path(prep_calypso_work_dir).joinpath(calypso_run_opt_file).resolve()
        )
        calypso_check_opt_script = (
            Path(prep_calypso_work_dir).joinpath(calypso_check_opt_file).resolve()
        )

        optim_paths = []
        with set_directory(work_dir):
            for idx, poscar in enumerate(poscar_list):
                opt_dir = calypso_opt_dir_name % idx
                optim_paths.append(work_dir.joinpath(opt_dir))
                with set_directory(opt_dir):
                    Path("POSCAR").symlink_to(poscar)
                    Path("frozen_model.pb").symlink_to(model_file)
                    Path(calypso_run_opt_file).symlink_to(calypso_run_opt_script)
                    Path(calypso_check_opt_file).symlink_to(calypso_check_opt_script)
            optim_names = [str(ii) for ii in optim_paths]

        return OPIO(
            {
                "optim_names": optim_names,
                "optim_paths": optim_paths,
            }
        )
