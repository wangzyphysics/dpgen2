import os
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
    List,
    Optional,
    Dict,
    Type,
)

from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Step,
    Steps,
    OPTemplate,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    PythonOPTemplate,
    Slices,
)

from dpgen2.constants import (
    calypso_index_pattern,
)
from .caly_evo_step import CalyEvoStep
from dpgen2.utils.step_config import (
    init_executor,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict


class PrepRunCaly(Steps):
    def __init__(
        self,
        name: str,
        prep_caly_input_op: Type[OP],
        caly_evo_step_op: Type[CalyEvoStep],
        run_caly_model_devi_op: Type[OP],
        prep_config: dict = normalize_step_dict({}),
        run_config: dict = normalize_step_dict({}),
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
        }
        self._input_artifacts = {
            "models": InputArtifact(),
        }
        self._output_parameters = {
            "task_names": OutputParameter(),
        }
        self._output_artifacts = {
            "trajs": OutputArtifact(),
            "model_devis": OutputArtifact(),
        }

        super().__init__(
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )

        # TODO: RunModelDevi
        self._keys = ["prep-caly-input", "caly-evo-step", "run-caly-model-devi"]
        self.step_keys = {}
        ii = "prep-caly-input"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "caly-evo-step"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "run-caly-model-devi"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])

        self = _prep_run_caly(
            self,
            self.step_keys,
            prep_caly_input_op,
            caly_evo_step_op,
            run_caly_model_devi_op,
            prep_config=prep_config,
            run_config=run_config,
            upload_python_packages=upload_python_packages,
        )

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def keys(self):
        return self._keys


def _prep_run_caly(
    prep_run_caly_steps: Steps,
    step_keys: Dict[str, Any],
    prep_caly_input_op: Type[OP],
    caly_evo_step_op: OPTemplate,
    run_caly_model_devi_op: Type[OP],
    prep_config: dict = normalize_step_dict({}),
    run_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    print(f"---run_config--------{run_config}")
    print(f"---prep_config--------{prep_config}")
    prep_template_config = prep_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    prep_executor = init_executor(prep_config.pop("executor"))
    run_executor = init_executor(run_config.pop("executor"))
    template_slice_config = run_config.pop("template_slice_config", {})
    print(f"---prep_template_config--------{prep_template_config}")
    print(f"---run_template_config--------{run_template_config}")
    caly_config = run_template_config.pop("caly_config")

    # prep caly input files
    prep_caly_input = Step(
        "prep-caly-input",
        template=PythonOPTemplate(
            prep_caly_input_op,
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "caly_inputs": caly_config["caly_inputs"],
        },
        artifacts={},
        key=step_keys["prep-caly-input"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_caly_steps.add(prep_caly_input)

    temp_value = [Path("dir_none") for _ in range(len(caly_config["caly_inputs"]))]
    # collect traj dirs
    caly_evo_step = Step(
        name="caly-evo-step",
        template=caly_evo_step_op,
        parameters={
            "block_id": prep_run_caly_steps.inputs.parameters["block_id"],
            "task_name_list": prep_caly_input.outputs.parameters["task_names"],
        },
        artifacts={
            "models": prep_run_caly_steps.inputs.artifacts["models"],
            "input_file_list": prep_caly_input.outputs.artifacts["input_dat_files"],
            "caly_run_opt_files": prep_caly_input.outputs.artifacts["caly_run_opt_files"],
            "caly_check_opt_files": prep_caly_input.outputs.artifacts["caly_check_opt_files"],
            "results_list": temp_value,
            "step_list": temp_value,
            "opt_results_dir_list": temp_value,
        },
        key=step_keys["caly-evo-step"],
        # executor=prep_executor,
        # **prep_config,
    )
    prep_run_caly_steps.add(caly_evo_step)

    # run model devi
    run_caly_model_devi = Step(
        "run-caly-model-devi",
        template=PythonOPTemplate(
            run_caly_model_devi_op,
            output_artifact_archive={},
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "type_map": caly_config["type_map"],
            "task_name": caly_evo_step.outputs.parameters["task_names"],
        },
        artifacts={
            "traj_dirs": caly_evo_step.outputs.artifacts["traj_results"],
            "models": prep_run_caly_steps.inputs.artifacts["models"],
        },
        key=step_keys["run-caly-model-devi"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_caly_steps.add(run_caly_model_devi)

    prep_run_caly_steps.outputs.parameters[
        "task_names"
    ]._value_from_parameter = run_caly_model_devi.outputs.parameters["task_name"]
    prep_run_caly_steps.outputs.artifacts[
        "trajs"
    ]._from = run_caly_model_devi.outputs.artifacts["traj"]
    prep_run_caly_steps.outputs.artifacts[
        "model_devis"
    ]._from = run_caly_model_devi.outputs.artifacts["model_devi"]

    return prep_run_caly_steps