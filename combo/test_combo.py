"""Physics/semantics regression checks; run python -m unittest combo.test_combo."""
import sys
import unittest
from .plant import GroundWorld, BSchema, BOUNDS, np
from .modes import GRAPH, prior
from .score import score_v3
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
