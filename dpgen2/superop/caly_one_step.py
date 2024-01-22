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


class CalyOneStep(Steps):
    def __init__(
        self,
        name: str,
        prep_dp_optim: Type[OP],
        run_dp_optim: Type[OP],
        prep_caly_nextstep: Type[OP],
        prep_config: dict = normalize_step_dict({}),
        run_config: dict = normalize_step_dict({}),
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
            "caly_inputs": InputParameter(),
            "caly_structure_path_names": InputParameter(),
            "input_file_path_names": InputParameter(),
        }
        self._input_artifacts = {
        }
        self._output_parameters = {
            "structure_paths": OutputParameter(),
        }
        self._output_artifacts = {
            "traj_paths": OutputArtifact(),
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

        self._keys = ["prep-dp-optim", "run-dp-optim", "prep-caly-nextstep"]
        self.step_keys = {}
        ii = "prep-dp-optim"
        self.step_keys[ii] = "--".join(
            ["%s" % self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )
        ii = "run-dp-optim"
        self.step_keys[ii] = "--".join(
            ["%s" % self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )
        ii = "prep-caly-nextstep"
        self.step_keys[ii] = "--".join(
            ["%s" % self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )

        self = _caly_one_step(
            self,
            self.step_keys,
            prep_dp_optim,
            run_dp_optim,
            prep_caly_nextstep,
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


def _caly_one_step(
    caly_one_step_steps,
    step_keys,
    prep_dp_optim_op: Type[OP],
    run_dp_optim_op: Type[OP],
    prep_caly_nextstep_op: Type[OP],
    prep_config: dict = normalize_step_dict({}),
    run_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    prep_executor = init_executor(prep_config.pop("executor"))  # e.g. cpu
    run_executor = init_executor(run_config.pop("executor"))  # e.g. gpu
    template_slice_config = run_config.pop("template_slice_config", {})

    prep_dp_optim = Step(
        "prep-dp-optim",
        template=PythonOPTemplate(
            prep_dp_optim_op,
            slices=Slices(
                "{{item}}",  # caly_task.{int}
                input_parameter=[
                    "caly_input",
                    "caly_structure_path_name",
                    "input_file_path_name",
                ],
                input_artifact=[],
                output_artifact=[
                    "optim_names",
                    "optim_paths",
                ],  # results will be aggregrated into list[optim_names], where optim_names is List[str]
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            # popsize can be obtained from caly_input dict
            "caly_input": caly_one_step_steps.inputs.parameters[
                "caly_inputs"
            ],  # list of caly_input(dict)
            # structures are stroing in caly_structure_path_name direcoty
            "caly_structure_path_name": caly_one_step_steps.inputs.parameters[
                "caly_structure_path_names"
            ],
            # model_file, calypso_run_opt_script and calypso_check_opt_script are stroing in input_file_path_name
            "input_file_path_name": caly_one_step_steps.inputs.parameters[
                "input_file_path_names"
            ],
        },
        artifacts={},
        key=step_keys["prep-dp-optim"],
        executor=prep_executor,
        **prep_config,
    )
    caly_one_step_steps.add(prep_dp_optim)

    run_dp_optim = Step(
        "run-dp-optim",
        template=PythonOPTemplate(
            run_dp_optim_op,
            slices=Slices(
                "{{item}}",  # caly_task.{int}/caly_pop.{int}
                input_parameter=["optim_name"],
                input_artifact=["optim_path"],
                # not sure whether return a path or every needed file.
                # output_artifact=["log", "traj", "POSCAR", "CONTCAR", "OUTCAR", "work_dir", "work_name"],
                output_parameter=[
                    "work_name"
                ],  # aggregrated into List[work_name], where work_name is like caly_task.{idx}/caly_pop.{idx}
                output_artifact=["work_dir"],
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "config": caly_one_step_steps.inputs.parameters["caly_config"],
            "optim_name": prep_dp_optim.outputs.parameters["optim_names"],
        },
        artifacts={
            "optim_path": prep_dp_optim.outputs.parameters["optim_paths"],
        },
        with_sequence=argo_sequence(
            argo_len(prep_dp_optim.outputs.parameters["optim_names"]),
            format=calypso_index_pattern,
        ),
        key=step_keys["run-dp-optim"],
        executor=run_executor,
        **run_config,
    )
    caly_one_step_steps.add(run_dp_optim)

    def aggregrate_optim_names(run_calypso_paths, dp_optim_paths):
        """after the run_dp_optim task done, all structures from different
        input.dat will mixed together, we have to group up the structures
        from same input.dat. As the dp_optim_paths are like ["caly_task.idx/caly_pop.idx"],
        which can be divided by caly_task.idx.
        """
        from collections import defaultdict

        d = defaultdict(list)
        for dir_name in dp_optim_paths:
            d[str(dir_name.parts[-2])].append(dir_name)
        return [d[str(each_dir)] for each_dir in run_calypso_paths]

    prep_caly_nextstep = Step(
        "prep-caly-nextstep",
        template=PythonOPTemplate(
            prep_caly_nextstep_op,
            slices=Slices(
                "{{item}}",
                input_parameter=[
                    "caly_input",
                    "run_calypso_name",
                    "run_dp_optim_names",
                ],
                input_artifact=[],
                # input_artifact=["optim_path"],
                # not sure whether return a path or every needed file.
                output_parameter=[
                    "structures_path_name", "finished"
                ],  # aggregrated into List[work_name], where work_name is like caly_task.{idx}/caly_pop.{idx}
                output_artifact=["structures_path", "log"],
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "config": caly_one_step_steps.inputs.parameters["config"],
            "caly_input": caly_one_step_steps.inputs.parameters["caly_configs"],
            "run_calypso_name": caly_one_step_steps.inputs.parameters[
                "caly_structure_path_names"
            ],
            "run_dp_optim_names": aggregrate_optim_names(
                caly_one_step_steps.inputs.parameters["caly_structure_path_names"],
                run_dp_optim.outputs.parameters["work_name"],
            ),
        },
        artifacts={
        },
        with_sequence=argo_sequence(
            argo_len(prep_dp_optim.outputs.parameters["optim_names"]),
            format=calypso_index_pattern,
        ),
        key=step_keys["prep-caly-nextstep"],
        executor=prep_executor,
        **run_config,
    )
    caly_one_step_steps.add(prep_caly_nextstep)

    # caly_one_step_steps.outputs.artifacts[
    #     "traj_paths"
    # ]._from = run_dp_optim.outputs.artifacts["work_name"]
    # caly_one_step_steps.outputs.artifacts[
    #     "structure_paths"
    # ]._from = prep_caly_nextstep.outputs.artifacts["structure_path_name"]
    # return caly_one_step_steps

    name = "calypso-nextstep-block"
    next_step = Step(
        name=name + "-next",
        template=caly_one_step_steps,
        parameters={
            "block_id": caly_one_step_steps.inputs.parameters["block_id"],
            "caly_inputs": caly_one_step_steps.inputs.parameters["caly_inputs"],
            "caly_structure_path_names": prep_caly_nextstep.outputs.parameters["caly_structure_path_names"],
            "input_file_path_names": prep_caly_nextstep.outputs.parameters["input_file_path_names"],
        },
        artifacts={
        },
        when="%s == false" % (prep_caly_nextstep.outputs.parameters["finished"]),
    )
    caly_one_step_steps.add(next_step)

    caly_one_step_steps.outputs.parameters[
        "traj_paths"
    ].value_from_expression = if_expression(
        _if=(prep_caly_nextstep.outputs.parameters["finished"] == True),
        _then=run_dp_optim.outputs.parameters["work_name"],
        _else=next_step.outputs.parameters["work_name"],
    )
    caly_one_step_steps.outputs.parameters[
        "structure_paths"
    ].value_from_expression = if_expression(
        _if=(prep_caly_nextstep.outputs.parameters["finished"] == True),
        _then=prep_caly_nextstep.outputs.parameters["structure_path_name"],
        _else=next_step.outputs.parameters["structure_path_name"],
    )

    return caly_one_step_steps
