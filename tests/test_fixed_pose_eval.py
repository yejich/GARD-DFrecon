import unittest
import numpy as np
from mvr.fixed_pose_eval import score_poses

class FixedPoseMetricsTest(unittest.TestCase):
    def setUp(self):
        self.ref = np.tile(np.eye(4)[:3], (4, 1, 1))
        self.ref[:, :3, 3] = -np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
        self.info = {'kind': 'hypersim_gt'}

    def test_similarity_gauge_invariance(self):
        pred = self.ref.copy()
        pred[:, :3, 3] = 3 * pred[:, :3, 3] + [2, 4, 6]
        _, details = score_poses(pred, pred, self.ref, self.info)
        self.assertAlmostEqual(details['gard']['auc_5'], 1.)
        self.assertLess(details['gard']['ate_rmse_sim3'], 1e-10)

    def test_rotation_error_and_colmap_units(self):
        pred = self.ref.copy()
        angle = np.deg2rad(30)
        pred[1, :3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0,0,1]]
        scalars, details = score_poses(self.ref, pred, self.ref, {'kind': 'colmap_sfm'})
        self.assertAlmostEqual(details['gard']['pairs'][0]['rre'], 30.)
        self.assertLess(details['gard']['auc_5'], details['da3']['auc_5'])
        self.assertFalse(any('ate' in key for key in scalars))

if __name__ == '__main__':
    unittest.main()
