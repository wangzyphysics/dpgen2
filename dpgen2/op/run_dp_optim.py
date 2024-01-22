import json
import logging
import os
import random
import re
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
    Tuple,
)

from dargs import (
    Argument,
    ArgumentEncoder,
    Variant,
    dargs,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    FatalError,
    OPIOSign,
    TransientError,
)

from dpgen2.constants import (
    calypso_check_opt_file,
    calypso_run_opt_file,
    calypso_opt_log_name,
    calypso_traj_log_name,
)
from dpgen2.utils import (
    BinaryFileInput,
    set_directory,
)
from dpgen2.utils.run_command import (
    run_command,
)


class RunOptim(OP):
    r"""Run structure optimization for CALYPSO-generated structures.

    Structure optimization will be executed in `optim_path`. The trajectory
    will be stored in files `op["traj"]` and `op["model_devi"]`, respectively.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "work_dir": Artifact(Path),
                "optim_name": str,
                "optim_path": Artifact(Path),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "log": Artifact(Path),
                "traj": Artifact(Path),
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
            - `config`: (`dict`) The config of calypso optim task. Check `RunOptim.optim_args` for definitions.
            - `optim_name`: (`str`) The name of the task.
            - `optim_path`: (`Artifact(Path)`) The path that contains one optim task prepareed by `PrepDPOptim`.

        Returns
        -------
        Any
            Output dict with components:
            - `log`: (`Artifact(Path)`) The log file of structure optimization.
            - `traj`: (`Artifact(Path)`) The trajectory file of structure optimization.
            - `POSCAR`: (`Artifact(Path)`) The poscar file of structure optimization.
            - `CONTCAR`: (`Artifact(Path)`) The contccar file of structure optimization.
            - `OUTCAR`: (`Artifact(Path)`) The outcar file of structure optimization.
            - `work_dir`: (`Artifact(Path)`)
            - `work_name`: (`str`)

        Raises
        ------
        TransientError
            On the failure of LAMMPS execution. Handle different failure cases? e.g. loss atoms.
        """
        config = ip["config"] if ip["config"] is not None else {}
        config = RunOptim.normalize_config(config)
        command = config.get("run_opt_command", "python")  # command should provide the python path

        optim_name = ip["optim_name"]
        optim_path = ip["optim_path"]
        work_dir = Path(optim_name)

        with set_directory(work_dir):
            # debug
            input_files = [ii.resolve() for ii in Path(work_dir).iterdir()]
            print(input_files)

            # run optimization
            command = " ".join([command, "-u", calypso_run_opt_file, "||", command, calypso_check_opt_file, ">", calypso_opt_log_name])
            ret, out, err = run_command(command, shell=True)
            if ret != 0:
                logging.error(
                    "".join(
                        (
                            "structure optimization failed\n",
                            "command was: ",
                            command,
                            "out msg: ",
                            out,
                            "\n",
                            "err msg: ",
                            err,
                            "\n",
                        )
                    )
                )
                raise TransientError("Structure optimization failed")

        ret_dict = {
            "log": work_dir / calypso_opt_log_name,
            "traj": work_dir / calypso_traj_log_name,
            "POSCAR": work_dir / "POSCAR",
            "CONTCAR": work_dir / "CONTCAR",
            "OUTCAR": work_dir / "OUTCAR",
            "work_dir": work_dir,
            "work_name": str(work_dir),
        }

        return OPIO(ret_dict)
