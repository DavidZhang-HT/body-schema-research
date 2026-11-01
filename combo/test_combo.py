"""Physics/semantics regression checks; run python -m unittest combo.test_combo."""
import sys
import unittest
import tempfile
import pickle
from pathlib import Path
from unittest.mock import patch
from .plant import GroundWorld, BSchema, BOUNDS, np
from .modes import GRAPH, prior
from .score import score_v3, exploration_priority
from .explain import transition_label
from experiment import standing


class ComboTests(unittest.TestCase):
    def test_clone_and_augmented_action(self):
        w = GroundWorld()
        self.assertEqual((w.m.nu, w.m.nq, w.m.nv), (24, 31, 30))
        w.passive('supine', 0)
        initial = w.snapshot()
        a = .1*BOUNDS
        w.advance(a)
        expected = w.snapshot()
        w.restore(initial)
        w.advance(a)
        np.testing.assert_allclose(w.snapshot(), expected, atol=1e-10, rtol=0)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('ground_model', sys.modules)

    def test_port_standing_and_supine(self):
        w = GroundWorld()
        schema = BSchema(w.m)
        self.assertTrue(schema.certify(standing(w.m), feet_only=True)['ok'])
        w.passive('supine', 0)
        self.assertTrue(schema.certify(w.d.qpos)['feasible'])
        self.assertFalse(schema.certify(w.d.qpos, feet_only=True)['feasible'])

    def test_score_pathologies(self):
        r = dict(com_height=.25, base_height=.25, up_z=.7, kinetic_j=.001,
                 com_speed=.01, foot_load=50, nonfoot_load=58)
        moving = dict(r, kinetic_j=.5, com_speed=.3)
        self.assertGreater(score_v3([(r,.02)]), score_v3([(moving,.02)]))
        feet = dict(r, foot_load=108, nonfoot_load=0)
        self.assertEqual(score_v3([(r,.02)]), score_v3([(feet,.02)]))
        spike = dict(r, base_height=.4)
        self.assertEqual(score_v3([(r,.02),(spike,.02)]), score_v3([(r,.02)]))

    def test_exploration_has_no_hand_support_tax(self):
        r = dict(com_height=.2,base_height=.15,up_z=.8,kinetic_j=.1,
                 com_speed=.05,foot_load=80,nonfoot_load=0,total_load=80)
        supported = dict(r,nonfoot_load=40,total_load=120)
        self.assertEqual(exploration_priority(r), exploration_priority(supported))

    def test_resume_terminal_success(self):
        from . import search
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'dynamic'
            out.mkdir()
            n = search.Node(1.,np.zeros(140),[np.zeros(16)],'CROUCH',1,{})
            state = dict(config=('supine',99,16,1,1),selection='dynamic',
                         initial=np.zeros(140),beam=[n],history=[],attempts=[],edges={},
                         evals=16,level=1,elapsed=3.,settle={},
                         rng=np.random.default_rng(1).bit_generator.state,
                         success=True,winner=n)
            (out/'supine_99.checkpoint.pkl').write_bytes(pickle.dumps(state))
            with patch.object(search,'OUT',Path(temp)), patch.object(search,'GroundWorld') as world:
                result = search.run('supine',99,budget=16,width=1,branches=1,selection='dynamic')
                self.assertTrue(result['success'])
                self.assertEqual(result['evaluations'],16)
                world.return_value.advance.assert_not_called()
                with self.assertRaises(ValueError):
                    search.run('supine',99,budget=32,width=1,branches=1,selection='dynamic')

    def test_graph_and_label_semantics(self):
        self.assertIn('SUPINE', GRAPH['STAND'])
        self.assertIn('SIDE_L', GRAPH['SIDE_R'])
        self.assertLess(prior('SUPINE','STAND'), 1)
        c = dict(feasible=True,rho=.1,pen=0,selfd=0)
        s = dict(certificate=c,metrics=dict(kinetic_j=.1))
        self.assertEqual(transition_label([s])['label'], 'DYNAMIC_UNRESOLVED')
        s['certificate'] = dict(c, feasible=False, rho=None)
        self.assertEqual(transition_label([s])['label'], 'STATICS_GAP')


if __name__ == '__main__':
    unittest.main()
