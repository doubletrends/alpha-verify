# Source architecture

`src/` contains the BarrierLab implementation. It owns the package layout and routes
modifiers to the boundary that owns a change; it does not define an experiment or retain
generated artifacts.

## Entry points

- [Package boundary](barrierlab/README.md): public CLI, dependency direction, shared
  ownership, and safe modification map.
- [Pipeline protocol](barrierlab/pipeline/README.md): four-stage artifact workflow,
  statistical procedure, inputs, outputs, and stage-specific limits.

The executable public entrypoint is `barrierlab.cli:main`, installed as `barrierlab`.
Workspace declarations live outside this source boundary in [workspaces/](../workspaces/README.md).
