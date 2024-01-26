import os
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
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
    Workflow,
    argo_len,
    argo_range,
    argo_sequence,
    if_expression,
    download_artifact,
    upload_artifact,
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
from dpgen2.utils.step_config import (
    init_executor,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict


class PrepRunCaly(Steps):
    def __init__(
        self,
        name: str,
        caly_one_step: Type[OP],
        run_caly_model_devi: Type[OP],
        prep_config: dict = normalize_step_dict({}),
        run_config: dict = normalize_step_dict({}),
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
            "caly_inputs": InputParameter(),
        }
        self._input_artifacts = {
            "models": InputArtifact(),
        }
        self._output_parameters = {
            "task_names": OutputParameter(),
        }
        self._output_artifacts = {
            "traj_results": OutputArtifact(),
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
        self._keys = ["caly-one-step", "run-caly-model-devi"]
        self.step_keys = {}
        ii = "caly-one-step"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "run-caly-model-devi"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])

        self = _prep_run_caly(
            self,
            self.step_keys,
            caly_one_step,
            run_caly_model_devi,
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
    prep_run_caly_steps,
    step_keys,
    caly_one_step_op: Type[OP],
    run_caly_model_devi_op: Type[OP],
    prep_config: dict = normalize_step_dict({}),
    run_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    prep_executor = init_executor(prep_config.pop("executor"))
    run_executor = init_executor(run_config.pop("executor"))
    template_slice_config = run_config.pop("template_slice_config", {})

    print(prep_run_caly_steps.inputs.parameters["block_id"])
    # collect traj dirs
    caly_one_step = Step(
        "caly-one-step",
        template=caly_one_step_op,
        parameters={
            "block_id": prep_run_caly_steps.inputs.parameters["block_id"],
            "caly_inputs": prep_run_caly_steps.inputs.parameters["caly_inputs"],
        },
        artifacts={},
        key=step_keys["caly-one-step"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_caly_steps.add(caly_one_step)

    # run model devi
    run_caly_model_devi = Step(
        "run-caly-model-devi",
        template=run_caly_model_devi_op,
        parameters={
            "type_map": prep_run_caly_steps.inputs.parameters["caly_inputs"],
            "task_name": prep_run_caly_steps.inputs.parameters["block_id"],
            "traj_dirs": prep_run_caly_steps.inputs.parameters["block_id"],
            "models": prep_run_caly_steps.inputs.parameters["block_id"],
        },
        key=step_keys["prep-caly-input"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_caly_steps.add(caly_one_step)

    prep_run_caly_steps.outputs.artifacts[
        "trajs"
    ]._from = run_caly_model_devi.outputs.artifacts["traj"]
    prep_run_caly_steps.outputs.artifacts[
        "model_devis"
    ]._from = run_caly_model_devi.outputs.artifacts["model_devi"]

    return prep_run_caly_steps
