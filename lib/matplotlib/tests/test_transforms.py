from __future__ import print_function
import unittest

from nose.tools import assert_equal
import numpy.testing as np_test
from numpy.testing import assert_almost_equal
import numpy as np

import matplotlib.transforms as mtrans
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches


def test_non_affine_caching():
    class AssertingNonAffineTransform(mtrans.Transform):
        """
        This transform raises an assertion error when called when it 
        shouldn't be and self.raise_on_transform is True.
        
        """
        input_dims = output_dims = 2
        is_affine = False
        def __init__(self, *args, **kwargs):
            mtrans.Transform.__init__(self, *args, **kwargs)
            self.raise_on_transform = False
            self.underlying_transform = mtrans.Affine2D().scale(10, 10)
            
        def transform_path_non_affine(self, path):
            if self.raise_on_transform:
                assert False, ('Invalidated affine part of transform '
                                    'unnecessarily.')
            return self.underlying_transform.transform_path(path)
        transform_path = transform_path_non_affine
        
        def transform_non_affine(self, path):
            if self.raise_on_transform:
                assert False, ('Invalidated affine part of transform '
                                    'unnecessarily.')
            return self.underlying_transform.transform(path)
        transform = transform_non_affine
    
    my_trans = AssertingNonAffineTransform()
    ax = plt.axes()
    plt.plot(range(10), transform=my_trans + ax.transData)
    plt.draw()
    # enable the transform to raise an exception if it's non-affine transform
    # method is triggered again.
    my_trans.raise_on_transform = True
    ax.transAxes.invalidate()
    plt.draw()
    

def test_Affine2D_from_values():
    points = np.array([ [0,0],
               [10,20],
               [-1,0],
               ])

    t = mtrans.Affine2D.from_values(1,0,0,0,0,0)
    actual = t.transform(points)
    expected = np.array( [[0,0],[10,0],[-1,0]] )
    assert_almost_equal(actual,expected)

    t = mtrans.Affine2D.from_values(0,2,0,0,0,0)
    actual = t.transform(points)
    expected = np.array( [[0,0],[0,20],[0,-2]] )
    assert_almost_equal(actual,expected)

    t = mtrans.Affine2D.from_values(0,0,3,0,0,0)
    actual = t.transform(points)
    expected = np.array( [[0,0],[60,0],[0,0]] )
    assert_almost_equal(actual,expected)

    t = mtrans.Affine2D.from_values(0,0,0,4,0,0)
    actual = t.transform(points)
    expected = np.array( [[0,0],[0,80],[0,0]] )
    assert_almost_equal(actual,expected)

    t = mtrans.Affine2D.from_values(0,0,0,0,5,0)
    actual = t.transform(points)
    expected = np.array( [[5,0],[5,0],[5,0]] )
    assert_almost_equal(actual,expected)

    t = mtrans.Affine2D.from_values(0,0,0,0,0,6)
    actual = t.transform(points)
    expected = np.array( [[0,6],[0,6],[0,6]] )
    assert_almost_equal(actual,expected)
    

class NonAffineForTest(mtrans.Transform):
    """
    A class which looks like a non affine transform, but does whatever 
    the given transform does (even if it is affine). This is very useful
    for testing NonAffine behaviour with a simple Affine transform.
    
    """
    is_affine = False
    output_dims = 2
    input_dims = 2
    
    def __init__(self, real_trans, *args, **kwargs):
        self.real_trans = real_trans
        r = mtrans.Transform.__init__(self, *args, **kwargs)

    def transform_non_affine(self, values):
        return self.real_trans.transform(values)
    
    def transform_path_non_affine(self, path):
        return self.real_trans.transform_path(path)


class BasicTransformTests(unittest.TestCase):
    def setUp(self):
        self.ta1 = mtrans.Affine2D(shorthand_name='ta1').rotate(np.pi / 2)
        self.ta2 = mtrans.Affine2D(shorthand_name='ta2').translate(10, 0)
        self.ta3 = mtrans.Affine2D(shorthand_name='ta3').scale(1, 2)
        
        self.tn1 = NonAffineForTest(mtrans.Affine2D().translate(1, 2), shorthand_name='tn1')
        self.tn2 = NonAffineForTest(mtrans.Affine2D().translate(1, 2), shorthand_name='tn2')
        self.tn3 = NonAffineForTest(mtrans.Affine2D().translate(1, 2), shorthand_name='tn3')
            
        # creates a transform stack which looks like ((A, (N, A)), A)
        self.stack1 = (self.ta1 + (self.tn1 + self.ta2)) + self.ta3
        # creates a transform stack which looks like (((A, N), A), A)
        self.stack2 = self.ta1 + self.tn1 + self.ta2 + self.ta3
        # creates a transform stack which is a subset of stack2
        self.stack2_subset = self.tn1 + self.ta2 + self.ta3
        
        # when in debug, the transform stacks can produce dot images:
#        self.stack1.write_graphviz(file('stack1.dot', 'w'))
#        self.stack2.write_graphviz(file('stack2.dot', 'w'))
#        self.stack2_subset.write_graphviz(file('stack2_subset.dot', 'w'))
    
    def test_left_to_right_iteration(self):
        stack3 = (self.ta1 + (self.tn1 + (self.ta2 + self.tn2))) + self.ta3
#        stack3.write_graphviz(file('stack3.dot', 'w'))
        
        target_transforms = [stack3, 
                             (self.tn1 + (self.ta2 + self.tn2)) + self.ta3,
                             (self.ta2 + self.tn2) + self.ta3,
                             self.tn2 + self.ta3,
                             self.ta3,
                             ]
        r = [rh for lh, rh in stack3._iter_break_from_left_to_right()]
        self.assertEqual(len(r), len(target_transforms))
        
        for target_stack, stack in zip(target_transforms, r):
            self.assertEqual(target_stack, stack)
            
    def test_transform_shortcuts(self):
        self.assertEqual(self.stack1 - self.stack2_subset, self.ta1)
        self.assertEqual(self.stack2 - self.stack2_subset, self.ta1)
        
        # check that we cannot find a chain from the subset back to the superset
        # (since the inverse of the Transform is not defined.)
        self.assertRaises(ValueError, self.stack2_subset.__sub__, self.stack2)
        self.assertRaises(ValueError, self.stack1.__sub__, self.stack2)
        
        aff1 = self.ta1 + (self.ta2 + self.ta3)
        aff2 = self.ta2 + self.ta3
        
        self.assertEqual(aff1 - aff2, self.ta1)
        self.assertEqual(aff1 - self.ta2, aff1 + self.ta2.inverted())
        
        self.assertEqual(self.stack1 - self.ta3, self.ta1 + (self.tn1 + self.ta2))
        self.assertEqual(self.stack2 - self.ta3, self.ta1 + self.tn1 + self.ta2)
        
        self.assertEqual((self.ta2 + self.ta3) - self.ta3 + self.ta3, self.ta2 + self.ta3)
        
    def test_contains_branch(self):
        r1 = (self.ta2 + self.ta1)
        r2 = (self.ta2 + self.ta1)
        self.assertEqual(r1, r2)
        self.assertNotEqual(r1, self.ta1)
        self.assertTrue(r1.contains_branch(r2))
        self.assertTrue(r1.contains_branch(self.ta1))
        self.assertFalse(r1.contains_branch(self.ta2))
        self.assertFalse(r1.contains_branch((self.ta2 + self.ta2)))
        
        self.assertEqual(r1, r2)

        self.assertTrue(self.stack1.contains_branch(self.ta3))
        self.assertTrue(self.stack2.contains_branch(self.ta3))
        
        self.assertTrue(self.stack1.contains_branch(self.stack2_subset))
        self.assertTrue(self.stack2.contains_branch(self.stack2_subset))
        
        self.assertFalse(self.stack2_subset.contains_branch(self.stack1))
        self.assertFalse(self.stack2_subset.contains_branch(self.stack2))
         
        self.assertTrue(self.stack1.contains_branch((self.ta2 + self.ta3)))
        self.assertTrue(self.stack2.contains_branch((self.ta2 + self.ta3)))
         
        self.assertFalse(self.stack1.contains_branch((self.tn1 + self.ta2)))
                        
    def test_affine_simplification(self):
        # tests that a transform stack only calls as much is absolutely necessary
        # "non-affine" allowing the best possible optimization with complex
        # transformation stacks. 
        points = np.array([[0, 0], [10, 20], [np.nan, 1], [-1, 0]], dtype=np.float64)
        na_pts = self.stack1.transform_non_affine(points)
        all_pts = self.stack1.transform(points)
 
        na_expected = np.array([[1., 2.], [-19., 12.], 
                                [np.nan, np.nan], [1., 1.]], dtype=np.float64)
        all_expected = np.array([[11., 4.], [-9., 24.], 
                                 [np.nan, np.nan], [11., 2.]], dtype=np.float64)        
        
        # check we have the expected results from doing the affine part only
        np_test.assert_array_almost_equal(na_pts, na_expected)
        # check we have the expected results from a full transformation
        np_test.assert_array_almost_equal(all_pts, all_expected)
        # check we have the expected results from doing the transformation in two steps
        np_test.assert_array_almost_equal(self.stack1.transform_affine(na_pts), all_expected)
        # check that getting the affine transformation first, then fully transforming using that
        # yields the same result as before.
        np_test.assert_array_almost_equal(self.stack1.get_affine().transform(na_pts), all_expected)
        
        # check that the affine part of stack1 & stack2 are equivalent (i.e. the optimization
        # is working)
        expected_result = (self.ta2 + self.ta3).get_matrix()
        result = self.stack1.get_affine().get_matrix()
        np_test.assert_array_equal(expected_result, result)
        
        result = self.stack2.get_affine().get_matrix()
        np_test.assert_array_equal(expected_result, result)
        
        
class TestTransformPlotInterface(unittest.TestCase):
    def tearDown(self):
        plt.close()
        
    def test_line_extents_affine(self):
        ax = plt.axes()
        offset = mtrans.Affine2D().translate(10, 10)
        plt.plot(range(10), transform=offset + ax.transData)
        expeted_data_lim = np.array([[0., 0.], [9.,  9.]]) + 10
        np.testing.assert_array_almost_equal(ax.dataLim.get_points(), 
                                             expeted_data_lim)
                
    def test_line_extents_non_affine(self):
        ax = plt.axes()
        offset = mtrans.Affine2D().translate(10, 10)
        na_offset = NonAffineForTest(mtrans.Affine2D().translate(10, 10))
        plt.plot(range(10), transform=offset + na_offset + ax.transData)
        expeted_data_lim = np.array([[0., 0.], [9.,  9.]]) + 20
        np.testing.assert_array_almost_equal(ax.dataLim.get_points(), 
                                             expeted_data_lim)
    
    def test_patch_extents_non_affine(self):
        ax = plt.axes()
        offset = mtrans.Affine2D().translate(10, 10)
        na_offset = NonAffineForTest(mtrans.Affine2D().translate(10, 10))
        pth = mpath.Path(np.array([[0, 0], [0, 10], [10, 10], [10, 0]]))
        patch = mpatches.PathPatch(pth, transform=offset + na_offset + ax.transData)
        ax.add_patch(patch)
        expeted_data_lim = np.array([[0., 0.], [10.,  10.]]) + 20
        np.testing.assert_array_almost_equal(ax.dataLim.get_points(), 
                                             expeted_data_lim)
        
    def test_patch_extents_affine(self):
        ax = plt.axes()
        offset = mtrans.Affine2D().translate(10, 10)
        pth = mpath.Path(np.array([[0, 0], [0, 10], [10, 10], [10, 0]]))
        patch = mpatches.PathPatch(pth, transform=offset + ax.transData)
        ax.add_patch(patch)
        expeted_data_lim = np.array([[0., 0.], [10.,  10.]]) + 10
        np.testing.assert_array_almost_equal(ax.dataLim.get_points(), 
                                             expeted_data_lim)
        
    def test_line_extents_for_non_affine_transData(self):
        ax = plt.axes(projection='polar')
        # add 10 to the radius of the data
        offset = mtrans.Affine2D().translate(0, 10)
        
        plt.plot(range(10), transform=offset + ax.transData)
        # the data lim of a polar plot is stored in coordinates
        # before a transData transformation, hence the data limits
        # are not what is being shown on the actual plot.
        expeted_data_lim = np.array([[0., 0.], [9.,  9.]]) + [0, 10]
        np.testing.assert_array_almost_equal(ax.dataLim.get_points(), 
                                             expeted_data_lim)
