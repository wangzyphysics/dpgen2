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
    calypso_log_name,
)
from dpgen2.utils import (
    BinaryFileInput,
    set_directory,
)
from dpgen2.utils.run_command import (
    run_command,
)


class RunCalypso(OP):
    r"""Execute CALYPSO to generate structures in work_dir.

    Changing the work directory into `task_name`. All input files
    have been copied or symbol linked to this directory `task_name` by
    `PrepCalyInput`. The CALYPSO command is exectuted from directory `task_name`.
    The `caly.log` and the `work_dir` will be stored in `op["log"]` and
    `op["work_dir"]`, respectively.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_path": Artifact[Path],
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "log": Artifact(Path),
                "work_dir": Artifact(Path),
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

            - `config`: (`dict`) The config of lmp task. Check `RunCalypso.calypso_args` for definitions.
            - `task_name`: (`Path`) The name of the task.

        Returns
        -------
        Any
            Output dict with components:
            - `work_dir_name`: (`List[str]`) The name of directory of structures.
            - `log`: (`Artifact(Path)`) The log file of LAMMPS.
            - `work_dir`: (`Artifact(Path)`) The directory of structures.

        Raises
        ------
        TransientError
            On the failure of CALYPSO execution. Resubmit rule should be clear.
        """
        config = ip["config"] if ip["config"] is not None else {}
        config = RunCalypso.normalize_config(config)
        command = config.get("run_calypso_command", "calypso.x")

        work_dir = ip["task_name"]
        input_dat = work_dir.joinpath("input.dat").resolve()

        with set_directory(work_dir):

            # copy input.dat
            Path(input_dat.name).symlink_to(input_dat)
            # run calypso
            command = " ".join([command, ">", calypso_log_name])
            ret, out, err = run_command(command, shell=True)
            if ret != 0:
                logging.error(
                    "".join(
                        (
                            "calypso failed\n",
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
                raise TransientError("calypso failed")

        ret_dict = {
            "log": work_dir / calypso_log_name,
            "work_dir": work_dir,
            "work_dir_name": str(work_dir),
        }

        return OPIO(ret_dict)

    @staticmethod
    def calypso_args():
        doc_calypso_cmd = "The command of calypso (absolute path of calypso.x)."
        return [
            Argument("command", str, optional=True, default="calypso.x", doc=doc_calypso_cmd),
        ]

    @staticmethod
    def normalize_config(data={}):
        ta = RunCalypso.calypso_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)
        return data


config_args = RunCalypso.calypso_args
