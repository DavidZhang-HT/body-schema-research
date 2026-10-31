"""Load the portable robot/checkpoints and execute one physical transition."""
import json
import numpy as np
import torch
from schema_v2_models import load_model, SensedWorld

torch.set_num_threads(1)
world = SensedWorld()
settle = world.passive('supine', 1200)
state = world.state()
model = load_model()
with torch.no_grad():
    imagined = model.imagine(torch.tensor(state['obs'][None]),
                             torch.tensor(state['qpos'][None], dtype=torch.float32),
                             torch.zeros(1, 3, 24))
actual, cost, _, _, residual, _ = world.sensed_step(np.zeros(24))
assert settle['settled'] and np.isfinite(actual['obs']).all()
assert bool(torch.isfinite(imagined[-1]['raw']).all())
print(json.dumps({'robot': 'Pi Plus', 'actuated_joints': world.m.nu,
                  'mass_kg': float(world.m.body_mass.sum()), 'loaded_checkpoints': True,
                  'physics_step_s': .32, 'imagined_horizon_s': .96,
                  'measured_effort': float(cost[0]),
                  'impulse_residual_norm': float(np.linalg.norm(residual))}, indent=2))
