from kmod.ex.exutil import fid_score, polynomial_mmd_averages
from kmod.ex.exutil import fid_permutation_test
import numpy as np
import os
import sys
from kmod import util, data, kernel
from kmod.mctest import SC_MMD
from kmod.mctest import SC_GaussUME
import kmod.glo as glo
from kmod.ex import celeba as clba
from kmod.ex import cifar10 as cf10
from kmod.ex import lsun
from collections import defaultdict
from sympy import Rational
from sympy import sympify

# need independent_jobs package
# https://github.com/karlnapf/independent-jobs
# The independent_jobs and kgof have to be in the global search path (.bashrc)
import independent_jobs as inj
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.Log import logger

"""
All the method functions take the following mandatory inputs:
    - mix_ratios: a list of dictionaries of mixture ratios specifying
    the fraction of classes in each sample given the sample size.
    mix_ratios[0:2] is for X, Y, Z, which are samples for P, Q, R,
    and mix_ratios[3] is for test locations V. All the ratios are
    specified by sympy Rationals
    - n: total sample size. Each method function should draw exactly the number
          of points using the method sample_data_mixing and mix_ratios
    - r: repetition (trial) index. Drawing samples should make use of r to
          set the random seed.
    -------
    - A method function may have more arguments which have default values.
"""


def met_fid(mix_ratios, n, r):
    """
    Compute the FIDs FID(P, R) and FIR(Q, R).
    The bootstrap estimator from Binkowski et al. 2018 is used.
    The number of bootstrap sampling can be specified by the variable splits
    below. For the method for the non-bootstrap version, see the method
    met_fid_nbstrp.
    """
    sample_size = [n] * 3
    X, Y, Z, _ = sample_data_mixing(mix_ratios, prob_module, sample_size, r)

    # keeping it the same as the comparison in MMD gan paper, 10 boostrap resamplings
    splits = 10
    split_size = X.shape[0]
    assert X.shape == Y.shape
    assert X.shape == Z.shape
    split_method = 'bootstrap'
    split_args = {'splits': splits, 'n': split_size, 'split_method': split_method}

    with util.ContextTimer() as t:
        fid_scores_xz = fid_score(X, Z, **split_args)
        fid_scores_yz = fid_score(Y, Z, **split_args)

        fid_score_xz = np.mean(fid_scores_xz)
        fid_score_yz = np.mean(fid_scores_yz)

    result = {'splits': splits, 'sample_size': split_size, 'score_xz': fid_score_xz,
              'score_yz': fid_score_yz, 'time_secs': t.secs, 'method': 'fid'}

    return result


def met_fid_nbstrp(mix_ratios, n, r):
    """
    Compute the FIDs FID(P, R) and FIR(Q, R).
    Unlike met_fid, the estimator is constructed by plugging the sample means and
    the sample covariances into the definition of FID.
    """
    sample_size = [n] * 3
    X, Y, Z, _ = sample_data_mixing(mix_ratios, prob_module, sample_size, r)

    # keeping it the same as the comparison in MMD gan paper, 10 boostrap resamplings
    splits = 1
    split_size = X.shape[0]
    assert X.shape == Y.shape
    assert X.shape == Z.shape
    split_method = 'copy'
    split_args = {'splits': splits, 'n': split_size, 'split_method': split_method}

    with util.ContextTimer() as t:
        fid_scores_xz = fid_score(X, Z, **split_args)
        fid_scores_yz = fid_score(Y, Z, **split_args)

        fid_score_xz = np.mean(fid_scores_xz)
        fid_score_yz = np.mean(fid_scores_yz)

    result = {'splits': splits, 'sample_size': split_size, 'score_xz': fid_score_xz,
              'score_yz': fid_score_yz, 'time_secs': t.secs, 'method': 'fid'}

    return result


def met_fid_perm(mix_ratios, n, r):
    """
    FID permutation test.
    """
    sample_size = [n] * 3
    X, Y, Z, _ = sample_data_mixing(mix_ratios, prob_module, sample_size, r)

    test_result = fid_permutation_test(X, Y, Z, alpha=alpha, n_permute=50)
    return test_result


def met_kid_mmd(mix_ratios, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test with the KID kernel
    in Binkowski et al., 2018.
    """

    sample_size = [n] * 3
    X, Y, Z, _ = sample_data_mixing(mix_ratios, prob_module, sample_size, r)

    k = kernel.KKID()
    scmmd = SC_MMD(data.Data(X), data.Data(Y), k, alpha)
    return scmmd.perform_test(data.Data(Z))


def met_kid(mix_ratios, n, r):
    """
    Compute MMD with the KID kernel. Note that this is not a test.
    """
    sample_size = [n] * 3
    X, Y, Z, _ = sample_data_mixing(mix_ratios, prob_module, sample_size, r)

    n_set = 100
    sub_size = 1000

    with util.ContextTimer() as t:
        kid_scores_xz = polynomial_mmd_averages(
            X, Z, degree=3, gamma=None,
            coef0=1, ret_var=False,
            n_subsets=n_set, subset_size=sub_size
        )

        kid_scores_yz = polynomial_mmd_averages(
            Y, Z, degree=3, gamma=None,
            coef0=1, ret_var=False,
            n_subsets=n_set, subset_size=sub_size
        )

        kid_score_xz = np.mean(kid_scores_xz)
        kid_score_yz = np.mean(kid_scores_yz)

    result = {'n_set': n_set, 'sub_size': sub_size,
              'score_xz': kid_score_xz, 'score_yz': kid_score_yz,
              'time_secs': t.secs, 'sample_size': n, 'rep': r,
              'method': 'kid'
              }
    return result


def met_gume_J_1_v_smile_celeba(mix_ratios, n, r, J=1):
    """
    UME-based three-sample test for celebA problems
    with test locations being smiling images.
        * Use J=1 test location by default.
    """

    sample_size = [n] * 3 + [J]
    mix_ratios.append({'ref_smile': sympify(1.0)})
    X, Y, Z, V = sample_data_mixing(mix_ratios, prob_module, sample_size, r)
    test_result = SC_GaussUME.ume_test(X, Y, Z, V, alpha=alpha)
    return test_result


def met_gume_J_5_v_smile_celeba(mix_ratios, n, r):
    return met_gume_J_1_v_smile_celeba(mix_ratios, n, r, J=5)


def met_gume_J_10_v_smile_celeba(mix_ratios, n, r):
    return met_gume_J_1_v_smile_celeba(mix_ratios, n, r, J=10)


def met_gume_J_20_v_smile_celeba(mix_ratios, n, r):
    return met_gume_J_1_v_smile_celeba(mix_ratios, n, r, J=20)


def met_gume_J_40_v_smile_celeba(mix_ratios, n, r):
    return met_gume_J_1_v_smile_celeba(mix_ratios, n, r, J=40)


def met_gume_J_1_v_nonsmile_celeba(mix_ratios, n, r, J=1):
    """
    UME-based three-sample test for celebA problems
    with test locations nonbeing smiling images.
        * Use J=1 test location by default.
    """

    sample_size = [n] * 3 + [J]

    mix_ratios.append({'ref_nonsmile': sympify(1.0)})
    X, Y, Z, V = sample_data_mixing(mix_ratios, prob_module, sample_size, r)
    test_result = SC_GaussUME.ume_test(X, Y, Z, V, alpha=alpha)
    return test_result


def met_gume_J_5_v_nonsmile_celeba(mix_ratios, n, r):
    return met_gume_J_1_v_nonsmile_celeba(mix_ratios, n, r, J=5)


def met_gume_J_10_v_nonsmile_celeba(mix_ratios, n, r):
    return met_gume_J_1_v_nonsmile_celeba(mix_ratios, n, r, J=10)


def met_gume_J_20_v_nonsmile_celeba(mix_ratios, n, r):
    return met_gume_J_1_v_nonsmile_celeba(mix_ratios, n, r, J=20)


def met_gume_J_40_v_nonsmile_celeba(mix_ratios, n, r):
    return met_gume_J_1_v_nonsmile_celeba(mix_ratios, n, r, J=40)


def met_gume_J_2_v_mix_celeba(mix_ratios, n, r, J=2):
    """
    UME-based three-sample test for celebA problems
    with test locations being a mixture of smiling/nonsmiling
    images of the equal proportion.
        * Use J=2 test location by default.
    """

    sample_size = [n] * 3 + [J]
    mix_ratios.append({'ref_smile': sympify(0.5), 'ref_nonsmile': sympify(0.5)})
    X, Y, Z, V = sample_data_mixing(mix_ratios, prob_module, sample_size, r)
    test_result = SC_GaussUME.ume_test(X, Y, Z, V, alpha=alpha)
    return test_result


def met_gume_J_10_v_mix_celeba(mix_ratios, n, r):
    return met_gume_J_1_v_nonsmile_celeba(mix_ratios, n, r, J=10)


def met_gume_J_20_v_mix_celeba(mix_ratios, n, r):
    return met_gume_J_1_v_nonsmile_celeba(mix_ratios, n, r, J=20)


def met_gume_J_40_v_mix_celeba(mix_ratios, n, r):
    return met_gume_J_1_v_nonsmile_celeba(mix_ratios, n, r, J=40)


def met_gmmd_med(mix_ratios, n, r):
    """
    Bounliphone et al., 2016's MMD-based 3-sample test.
    * Gaussian kernel.
    * Gaussian width = mean of (median heuristic on (X, Z), median heuristic on
        (Y, Z))
    """

    sample_size = [n] * 3
    X, Y, Z, _ = sample_data_mixing(mix_ratios, prob_module, sample_size, r)
    test_result = SC_MMD.mmd_test(X, Y, Z, alpha=alpha)
    return test_result


def met_gume_J_1_v_dog_ci10(mix_ratios, n, r, J=1):
    sample_size = [n] * 3 + [J]

    mix_ratios.append({'dog': sympify(1.0)})
    X, Y, Z, V = sample_data_mixing(mix_ratios, prob_module, sample_size, r)
    test_result = SC_GaussUME.ume_test(X, Y, Z, V, alpha=alpha)
    return test_result


def met_gume_J_5_v_dog_ci10(mix_ratios, n, r, J=5):
    return met_gume_J_1_v_dog_ci10(mix_ratios, n, r, J=5)


def met_gume_J_10_v_dog_ci10(mix_ratios, n, r, J=5):
    return met_gume_J_1_v_dog_ci10(mix_ratios, n, r, J=10)


def met_gume_J_1_v_deer_ci10(mix_ratios, n, r, J=1):
    sample_size = [n] * 3 + [J]

    mix_ratios.append({'deer': sympify(1.0)})
    X, Y, Z, V = sample_data_mixing(mix_ratios, prob_module, sample_size, r)
    test_result = SC_GaussUME.ume_test(X, Y, Z, V, alpha=alpha)
    return test_result


def met_gume_J_5_v_deer_ci10(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_deer_ci10(mix_ratios, n, r, J=5)


def met_gume_J_10_v_deer_ci10(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_deer_ci10(mix_ratios, n, r, J=10)


def met_gume_J_1_v_horse_ci10(mix_ratios, n, r, J=1):
    sample_size = [n] * 3 + [J]

    mix_ratios.append({'horse': sympify(1.0)})
    X, Y, Z, V = sample_data_mixing(mix_ratios, prob_module, sample_size, r)
    test_result = SC_GaussUME.ume_test(X, Y, Z, V, alpha=alpha)
    return test_result


def met_gume_J_5_v_horse_ci10(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_horse_ci10(mix_ratios, n, r, J=5)


def met_gume_J_10_v_horse_ci10(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_horse_ci10(mix_ratios, n, r, J=10)


def met_gume_J_1_v_rest_lsun(mix_ratios, n, r, J=1):
    sample_size = [n] * 3 + [J]

    mix_ratios.append({'restaurant': sympify(1.0)})
    X, Y, Z, V = sample_data_mixing(mix_ratios, prob_module, sample_size, r)
    test_result = SC_GaussUME.ume_test(X, Y, Z, V, alpha=alpha)
    return test_result


def met_gume_J_10_v_rest_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_rest_lsun(mix_ratios, n, r, J=10)


def met_gume_J_20_v_rest_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_rest_lsun(mix_ratios, n, r, J=20)


def met_gume_J_40_v_rest_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_rest_lsun(mix_ratios, n, r, J=40)


def met_gume_J_1_v_conf_lsun(mix_ratios, n, r, J=1):
    sample_size = [n] * 3 + [J]

    mix_ratios.append({'confroom': sympify(1.0)})
    X, Y, Z, V = sample_data_mixing(mix_ratios, prob_module, sample_size, r)
    test_result = SC_GaussUME.ume_test(X, Y, Z, V, alpha=alpha)
    return test_result


def met_gume_J_10_v_conf_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_conf_lsun(mix_ratios, n, r, J=10)


def met_gume_J_20_v_conf_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_conf_lsun(mix_ratios, n, r, J=20)


def met_gume_J_40_v_conf_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_conf_lsun(mix_ratios, n, r, J=40)


def met_gume_J_120_v_conf_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_conf_lsun(mix_ratios, n, r, J=120)


def met_gume_J_1_v_kitchen_lsun(mix_ratios, n, r, J=1):
    sample_size = [n] * 3 + [J]

    mix_ratios.append({'kitchen': sympify(1.0)})
    X, Y, Z, V = sample_data_mixing(mix_ratios, prob_module, sample_size, r)
    test_result = SC_GaussUME.ume_test(X, Y, Z, V, alpha=alpha)
    return test_result


def met_gume_J_10_v_kitchen_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_kitchen_lsun(mix_ratios, n, r, J=10)


def met_gume_J_20_v_kitchen_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_kitchen_lsun(mix_ratios, n, r, J=20)


def met_gume_J_40_v_kitchen_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_kitchen_lsun(mix_ratios, n, r, J=40)


def met_gume_J_120_v_kitchen_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_kitchen_lsun(mix_ratios, n, r, J=120)


def met_gume_J_4_v_mix_lsun(mix_ratios, n, r, J=4):
    """
    UME-based three-sample test for LSUN problems
    with test locations being a mixture of kitchen/restaurant/confroom/bedroom
    images of the equal proportion.
        * Use J=4 test location by default.
    """

    sample_size = [n] * 3 + [J]
    mix_ratios.append({'kitchen': Rational(1, 4), 'restaurant': Rational(1, 4),
                       'confroom': Rational(1, 4), 'bedroom': Rational(1, 4),
                       }
                      )
    X, Y, Z, V = sample_data_mixing(mix_ratios, prob_module, sample_size, r)
    test_result = SC_GaussUME.ume_test(X, Y, Z, V, alpha=alpha)
    return test_result


def met_gume_J_20_v_mix_lsun(mix_ratios, n, r, J=4):
    return met_gume_J_4_v_mix_lsun(mix_ratios, n, r, J=20)


def met_gume_J_40_v_mix_lsun(mix_ratios, n, r, J=4):
    return met_gume_J_4_v_mix_lsun(mix_ratios, n, r, J=40)


def met_gume_J_120_v_mix_lsun(mix_ratios, n, r, J=4):
    return met_gume_J_4_v_mix_lsun(mix_ratios, n, r, J=120)


def met_gume_J_160_v_mix_lsun(mix_ratios, n, r, J=4):
    return met_gume_J_4_v_mix_lsun(mix_ratios, n, r, J=160)


def met_gume_J_1_v_3212e20_lsun(mix_ratios, n, r, J=1):
    sample_size = [n] * 3 + [J]

    mix_ratios.append({'3212_dcgan_20': sympify(1.0)})
    X, Y, Z, V = sample_data_mixing(mix_ratios, prob_module, sample_size, r)
    test_result = SC_GaussUME.ume_test(X, Y, Z, V, alpha=alpha)
    return test_result


def met_gume_J_10_v_3212e20_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_3212e20_lsun(mix_ratios, n, r, J=10)


def met_gume_J_40_v_3212e20_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_3212e20_lsun(mix_ratios, n, r, J=40)


def met_gume_J_120_v_3212e20_lsun(mix_ratios, n, r, J=1):
    return met_gume_J_1_v_3212e20_lsun(mix_ratios, n, r, J=120)



def sample_feat_array(class_spec, prob_module, seed=37):
    """
    Return a split of data in prob_module specified by class_spec.
    Args:
        - class_spec: a tuple of
        ('class_name', #sample for X, #sample for Y, #sample for Z,
        #sample for V) specifying a split of the data of 'class_name'.

    Returns:
        -(X, Y, Z, V): numpy arrays representing 3 sampels and test locations.
    """
    # sample feature matrix
    list_X = []
    list_Y = []
    list_Z = []
    list_pool = []

    with util.NumpySeedContext(seed=seed):
        for i, cs in enumerate(class_spec):
            if len(cs) != 5:
                err_msg = 'class spec must of length 5. Was {}'.format(len(cs))
                raise ValueError(err_msg)
            # load class data
            class_i = cs[0]
            feas_i = prob_module.load_feature_array(class_i, feature_folder=feature_folder)

            # split each class according to the spec
            class_sizes_i = cs[1:]
            # feas_i may contain more than what we need in total for a class.
            # if random locations are shared by trials, sample with the same seed
            if is_loc_common:
                pool_seed = 1
                pool_size = class_sizes_i[3]
                pool_ind = util.subsample_ind(feas_i.shape[0],
                                              pool_size, seed=pool_seed)
                pool = feas_i[pool_ind]
                feas_i = np.delete(feas_i, pool_ind, axis=0)
                class_sizes_i = class_sizes_i[:3]
                list_pool.append(pool)

            if sum(class_sizes_i) > 0:
                sub_ind = util.subsample_ind(feas_i.shape[0],
                                             sum(class_sizes_i), seed=seed+1)
                sub_ind = list(sub_ind)
                assert len(sub_ind) == sum(class_sizes_i)

                xyzp_feas_i = util.multi_way_split(feas_i[sub_ind, :],
                                                   class_sizes_i)

                # assignment
                list_X.append(xyzp_feas_i[0])
                list_Y.append(xyzp_feas_i[1])
                list_Z.append(xyzp_feas_i[2])
                if not is_loc_common:
                    list_pool.append(xyzp_feas_i[3])

    X = np.vstack(list_X)
    Y = np.vstack(list_Y)
    Z = np.vstack(list_Z)
    V = np.vstack(list_pool)
    return X, Y, Z, V


def sample_data_mixing(mix_ratios, prob_module, sample_size, r):
    """
    Generate three samples from the mixture ratios given a trial
    index r.

    Args:
        - mix_ratios: a list mixture ratios of classes of a given problem.
        It must be of length 3 or 4. 
        - prob_module: the module name for the problem
        - sample_size: a list of sample sizes for three sampels and test
        locations
        - r: trial index
    Return:
        -(X, Y, Z, V): numpy arrays representing 3 sampels and test locations.
        If mix_ratios does not have the ratio for test locations, V would be
        an empty array.
    """
    classes = prob_module.get_classes()
    class_spec_dict = defaultdict(list)
    for i, mix_ratio in enumerate(mix_ratios):
        for key in mix_ratio.keys():
            if key not in classes:
                err_msg = 'Invalid class specification. Key: {}'.format(key)
                raise ValueError(err_msg)
        for class_name in classes:
            if class_name in mix_ratio:
                ratio = mix_ratio[class_name]
            else:
                ratio = sympify(0)
            n = int((sympify(sample_size[i]) * ratio).evalf())
            class_spec_dict[class_name].append(n)

    class_spec = []
    for class_name, spec in class_spec_dict.items():
        if sum(spec) > 0:
            if len(spec) < 4:
                spec.append(0)
            name_spec_tuple = (class_name, ) + tuple(spec)
            class_spec.append(name_spec_tuple)
    seed = r + sample_size[0]
    return sample_feat_array(class_spec, prob_module, seed=seed)


def get_ns_pm_mixing_ratios(prob_label):
    """
    Return a tuple of (ns, pm, mix_ratios), where
        - ns: (a list of ) sample sizes n's
        - pm: the module name for the problem, e.g. clba=celeba,
        cf10=cifar10
        - mix_ratios: a list of dictionaries of mixture ratios specifying
        the fraction of classes in each sample given the sample size.
        mix_ratios[0:2] is for X, Y, Z, which are samples for P, Q, R,
        and mix_ratios[3] is for test locations V. All the ratios are
        specified by sympy Rationals
    """
    sp = sympify
    prob2tuples = {
        'clba_p_gs_q_gn_r_rs': (
            [2000], clba,
            [{'gen_smile': sp(1.0)}, {'gen_nonsmile': sp(1.0)},
             {'ref_smile': sp(1.0)}]
        ),
        'clba_p_gs_q_gn_r_rn': (
            [2000], clba,
            [{'gen_smile': sp(1.0)}, {'gen_nonsmile': sp(1.0)},
             {'ref_nonsmile': sp(1.0)}]
        ),
        'clba_p_gs_q_gn_r_rm': (
            [2000], clba,
            [{'gen_smile': sp(1.0)}, {'gen_nonsmile': sp(1.0)},
             {'ref_smile': sp(0.5), 'ref_nonsmile': sp(0.5)}]
        ),
        'clba_p_gs_q_gs_r_rn': (
            [2000], clba,
            [{'gen_smile': sp(1.0)}, {'gen_smile': sp(1.0)},
             {'ref_nonsmile': sp(1.0)}]
        ),
        'clba_p_rs_q_rn_r_rm': (
            [2000], clba,
            [{'ref_smile': sp(1.0)}, {'ref_nonsmile': sp(1.0)},
             {'ref_smile': sp(0.5), 'ref_nonsmile': sp(0.5)}]
        ),
        'clba_p_rs_q_rn_r_rum': (
            [2000], clba,
            [{'ref_smile': sp(1.0)}, {'ref_nonsmile': sp(1.0)},
             {'ref_smile': sp(0.3), 'ref_nonsmile': sp(0.7)}]
        ),
        'clba_p_rs_q_rs_r_rs': (
            [2000], clba,
            [{'ref_smile': sp(1.0)}, {'ref_smile': sp(1.0)},
             {'ref_smile': sp(1.0)}]
        ),
        'clba_p_rs_q_rn_r_rn': (
            [2000], clba,
            [{'ref_smile': sp(1.0)}, {'ref_nonsmile': sp(1.0)},
             {'ref_nonsmile': sp(1.0)}]
        ),
        'clba_p_gs_q_gs_r_rs': (
            [2000], clba,
            [{'gen_smile': sp(1.0)}, {'gen_smile': sp(1.0)},
             {'ref_smile': sp(1.0)}]
        ),
        'cf10_p_hd_q_dd_r_ad': (
            [3500], cf10,
            [
                {'horse': Rational(2000, 3500), 'dog': Rational(1500, 3500)},
                {'deer': Rational(2000, 3500), 'dog': Rational(1500, 3500)},
                {'deer': Rational(1500, 3500), 'dog': Rational(1500, 3500),
                 'airplane': Rational(500, 3500)}
            ]
        ),
        'lsun_p_3212b_q_1232b_r_1313': (
            [2000], lsun,
            [{'3212_began': sp(1.0)}, {'1232_began': sp(1.0)},
             {'kitchen': Rational(1, 8), 'restaurant': Rational(3, 8),
              'confroom': Rational(1, 8), 'bedroom': Rational(3, 8)},
             ]
        ),
        'lsun_p_3212b_q_1232b_r_1232': (
            [2000], lsun,
            [{'3212_began': sp(1.0)}, {'1232_began': sp(1.0)},
             {'kitchen': Rational(1, 8), 'restaurant': Rational(2, 8),
              'confroom': Rational(3, 8), 'bedroom': Rational(2, 8)},
             ]
        ),
        'lsun_p_3212d_q_1232d_r_1313': (
            [2000], lsun,
            [{'3212_dcgan': sp(1.0)}, {'1232_dcgan': sp(1.0)},
             {'kitchen': Rational(1, 8), 'restaurant': Rational(3, 8),
              'confroom': Rational(1, 8), 'bedroom': Rational(3, 8)},
             ]
        ),
        'lsun_p_3212d_q_1232d_r_1232': (
            [2000], lsun,
            [{'3212_dcgan': sp(1.0)}, {'1232_dcgan': sp(1.0)},
             {'kitchen': Rational(1, 8), 'restaurant': Rational(2, 8),
              'confroom': Rational(3, 8), 'bedroom': Rational(2, 8)},
             ]
        ),
        'lsun_p_3212_q_1232_r_1313': (
            [2000], lsun,
            [{'kitchen': Rational(3, 8), 'restaurant': Rational(2, 8),
              'confroom':Rational(1, 8), 'bedroom': Rational(2, 8)}, 
             {'kitchen': Rational(1, 8), 'restaurant': Rational(2, 8),
              'confroom': Rational(3, 8), 'bedroom': Rational(2, 8)},
             {'kitchen': Rational(1, 8), 'restaurant': Rational(3, 8),
              'confroom': Rational(1, 8), 'bedroom': Rational(3, 8)},
             ]
        ),
        'lsun_p_3212_q_1232_r_1232': (
            [2000], lsun,
            [{'kitchen': Rational(3, 8), 'restaurant': Rational(2, 8),
              'confroom':Rational(1, 8), 'bedroom': Rational(2, 8)}, 
             {'kitchen': Rational(1, 8), 'restaurant': Rational(2, 8),
              'confroom': Rational(3, 8), 'bedroom': Rational(2, 8)},
             {'kitchen': Rational(1, 8), 'restaurant': Rational(2, 8),
              'confroom': Rational(3, 8), 'bedroom': Rational(2, 8)},
             ]
        ),
        'lsun_p_3212_q_1232_r_3212': (
            [2000], lsun,
            [{'kitchen': Rational(3, 8), 'restaurant': Rational(2, 8),
              'confroom':Rational(1, 8), 'bedroom': Rational(2, 8)}, 
             {'kitchen': Rational(1, 8), 'restaurant': Rational(2, 8),
              'confroom': Rational(3, 8), 'bedroom': Rational(2, 8)},
             {'kitchen': Rational(3, 8), 'restaurant': Rational(2, 8),
              'confroom': Rational(1, 8), 'bedroom': Rational(2, 8)},
             ]
        ),
         'lsun_p_3212_q_3212_r_3212': (
            [2000], lsun,
            [{'kitchen': Rational(3, 8), 'restaurant': Rational(2, 8),
              'confroom':Rational(1, 8), 'bedroom': Rational(2, 8)}, 
             {'kitchen': Rational(3, 8), 'restaurant': Rational(2, 8),
              'confroom': Rational(1, 8), 'bedroom': Rational(2, 8)},
             {'kitchen': Rational(3, 8), 'restaurant': Rational(2, 8),
              'confroom': Rational(1, 8), 'bedroom': Rational(2, 8)},
             ]
        ),
        'lsun_p_e1_q_e10_r_e20': (
            [2000], lsun,
            [{'3212_dcgan_1': sp(1.0)},
             {'3212_dcgan_10': sp(1.0)},
             {'3212_dcgan_20': sp(1.0)},
             ]
        ),
    }

    if prob_label not in prob2tuples:
        err_msg = ('Unknown problem label. Need to be one of %s'
                   % str(list(prob2tuples.keys())))
        raise ValueError(err_msg)
    return prob2tuples[prob_label]


# Define our custom Job, which inherits from base class IndependentJob
class Ex3Job(IndependentJob):

    def __init__(self, aggregator, mix_ratios, prob_label, rep, met_func, n):
        walltime = 60*59*24
        #walltime = 60 * 59
        memory = int(n*1e-2) + 50

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                                memory=memory)
        # P, P are kmod.model.Model
        self.mix_ratios = mix_ratios
        self.prob_label = prob_label
        self.rep = rep
        self.met_func = met_func
        self.n = n

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):

        mix_ratios = self.mix_ratios
        r = self.rep
        n = self.n
        met_func = self.met_func
        prob_label = self.prob_label

        logger.info("computing. %s. prob=%s, r=%d,\
                n=%d" % (met_func.__name__, prob_label, r, n))
        with util.ContextTimer() as t:
            job_result = met_func(mix_ratios, n, r)

            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = met_func.__name__

        logger.info("done. ex2: %s, prob=%s, r=%d, n=%d. Took: %.3g s "%(func_name,
            prob_label, r, n, t.secs))

        # save result
        fname = '%s-%s-n%d_r%d_a%.3f.p' \
                % (prob_label, func_name, n, r, alpha)
        glo.ex_save_result(ex, job_result, prob_label, fname)

# This import is needed so that pickle knows about the class Ex3Job.
# pickle is used when collecting the results from the submitted jobs.


from kmod.ex.ex3_real_images import Ex3Job
from kmod.ex.ex3_real_images import met_fid
from kmod.ex.ex3_real_images import met_kid_mmd
from kmod.ex.ex3_real_images import met_gmmd_med
from kmod.ex.ex3_real_images import met_gume_J_10_v_nonsmile_celeba
from kmod.ex.ex3_real_images import met_gume_J_10_v_smile_celeba
from kmod.ex.ex3_real_images import met_gume_J_5_v_nonsmile_celeba
from kmod.ex.ex3_real_images import met_gume_J_5_v_smile_celeba
from kmod.ex.ex3_real_images import met_kid
from kmod.ex.ex3_real_images import met_gume_J_10_v_deer_ci10
from kmod.ex.ex3_real_images import met_gume_J_10_v_dog_ci10
from kmod.ex.ex3_real_images import met_gume_J_10_v_horse_ci10
from kmod.ex.ex3_real_images import met_gume_J_5_v_deer_ci10
from kmod.ex.ex3_real_images import met_gume_J_5_v_dog_ci10
from kmod.ex.ex3_real_images import met_gume_J_5_v_horse_ci10
from kmod.ex.ex3_real_images import met_gume_J_1_v_deer_ci10
from kmod.ex.ex3_real_images import met_gume_J_1_v_dog_ci10
from kmod.ex.ex3_real_images import met_gume_J_1_v_horse_ci10
from kmod.ex.ex3_real_images import met_fid_perm
from kmod.ex.ex3_real_images import met_fid_nbstrp
from kmod.ex.ex3_real_images import met_gume_J_20_v_smile_celeba
from kmod.ex.ex3_real_images import met_gume_J_40_v_smile_celeba
from kmod.ex.ex3_real_images import met_gume_J_20_v_nonsmile_celeba 
from kmod.ex.ex3_real_images import met_gume_J_40_v_nonsmile_celeba 
from kmod.ex.ex3_real_images import met_gume_J_10_v_mix_celeba
from kmod.ex.ex3_real_images import met_gume_J_20_v_mix_celeba
from kmod.ex.ex3_real_images import met_gume_J_40_v_mix_celeba
from kmod.ex.ex3_real_images import met_gume_J_1_v_rest_lsun
from kmod.ex.ex3_real_images import met_gume_J_10_v_rest_lsun
from kmod.ex.ex3_real_images import met_gume_J_20_v_rest_lsun
from kmod.ex.ex3_real_images import met_gume_J_40_v_rest_lsun
from kmod.ex.ex3_real_images import met_gume_J_1_v_conf_lsun
from kmod.ex.ex3_real_images import met_gume_J_10_v_conf_lsun
from kmod.ex.ex3_real_images import met_gume_J_20_v_conf_lsun
from kmod.ex.ex3_real_images import met_gume_J_40_v_conf_lsun
from kmod.ex.ex3_real_images import met_gume_J_120_v_conf_lsun
from kmod.ex.ex3_real_images import met_gume_J_1_v_kitchen_lsun
from kmod.ex.ex3_real_images import met_gume_J_10_v_kitchen_lsun
from kmod.ex.ex3_real_images import met_gume_J_20_v_kitchen_lsun
from kmod.ex.ex3_real_images import met_gume_J_40_v_kitchen_lsun
from kmod.ex.ex3_real_images import met_gume_J_120_v_kitchen_lsun
from kmod.ex.ex3_real_images import met_gume_J_4_v_mix_lsun
from kmod.ex.ex3_real_images import met_gume_J_20_v_mix_lsun
from kmod.ex.ex3_real_images import met_gume_J_40_v_mix_lsun
from kmod.ex.ex3_real_images import met_gume_J_120_v_mix_lsun
from kmod.ex.ex3_real_images import met_gume_J_160_v_mix_lsun
from kmod.ex.ex3_real_images import met_gume_J_1_v_3212e20_lsun
from kmod.ex.ex3_real_images import met_gume_J_10_v_3212e20_lsun
from kmod.ex.ex3_real_images import met_gume_J_40_v_3212e20_lsun
from kmod.ex.ex3_real_images import met_gume_J_120_v_3212e20_lsun


# --- experimental setting -----
ex = 3

# significance level of the test
alpha = 0.05

# repetitions for each sample size
reps = 1

# tests to try
method_funcs = [
    met_gmmd_med,
    # met_gume_J_5_v_smile_celeba,
    # met_gume_J_5_v_nonsmile_celeba,
    # met_gume_J_10_v_smile_celeba,
    # met_gume_J_10_v_nonsmile_celeba,
    # met_gume_J_20_v_smile_celeba,
    # met_gume_J_20_v_nonsmile_celeba,
    # met_gume_J_40_v_smile_celeba,
    # met_gume_J_40_v_nonsmile_celeba,
    # met_gume_J_10_v_deer_ci10,
    # met_gume_J_10_v_dog_ci10,
    # met_gume_J_10_v_horse_ci10,
    # met_fid,
    # met_kid,
    met_kid_mmd,
    # met_fid_perm,
    # met_fid_nbstrp,
    #met_gume_J_10_v_mix_celeba,
    #met_gume_J_20_v_mix_celeba,
    #met_gume_J_40_v_mix_celeba,
    #met_gume_J_20_v_mix_lsun,
    #met_gume_J_40_v_mix_lsun,
    #met_gume_J_120_v_mix_lsun,
    #met_gume_J_160_v_mix_lsun,
    #met_gume_J_10_v_conf_lsun,
    #met_gume_J_20_v_conf_lsun,
    #met_gume_J_40_v_conf_lsun,
    #met_gume_J_120_v_conf_lsun,
    #met_gume_J_10_v_kitchen_lsun,
    #met_gume_J_20_v_kitchen_lsun,
    #met_gume_J_40_v_kitchen_lsun,
    #met_gume_J_120_v_kitchen_lsun,
    met_gume_J_10_v_3212e20_lsun,
    met_gume_J_40_v_3212e20_lsun,
    met_gume_J_120_v_3212e20_lsun,
]

prob_module = lsun
feature_folder = 'alexnet_features'
# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting already exists.
is_rerun = False

is_loc_common = True
# ---------------------------


def run_problem(prob_label):
    """Run the experiment"""
    # ///////  submit jobs //////////
    # create folder name string
    #result_folder = glo.result_folder()
    from kmod.config import expr_configs
    tmp_dir = expr_configs['scratch_path']
    foldername = os.path.join(tmp_dir, 'kmod_slurm', 'e%d'%ex)
    logger.info("Setting engine folder to %s" % foldername)

    # create parameter instance that is needed for any batch computation engine
    logger.info("Creating batch parameter instance")
    batch_parameters = BatchClusterParameters(
        foldername=foldername, job_name_base="e%d_"%ex, parameter_prefix="")

    # Use the following line if Slurm queue is not used.
    # engine = SerialComputationEngine()
    partitions = expr_configs['slurm_partitions']
    if partitions is None:
        engine = SlurmComputationEngine(batch_parameters)
    else:
        engine = SlurmComputationEngine(batch_parameters, partition=partitions)
    n_methods = len(method_funcs)

    # problem setting
    ns, pm, mix_ratios = get_ns_pm_mixing_ratios(prob_label)


    # repetitions x len(ns) x #methods
    aggregators = np.empty((reps, len(ns), n_methods), dtype=object)

    for r in range(reps):
        for ni, n in enumerate(ns):
            for mi, f in enumerate(method_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-n%d_r%d_a%.3f.p' \
                        %(prob_label, func_name, n, r, alpha,)
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    job_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(job_result))
                    aggregators[r, ni, mi] = sra
                else:
                    # result not exists or rerun
                    job = Ex3Job(SingleResultAggregator(), mix_ratios, prob_label,
                                 r, f, n)

                    agg = engine.submit_job(job)
                    aggregators[r, ni, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    job_results = np.empty((reps, len(ns), n_methods), dtype=object)
    for r in range(reps):
        for ni, n in enumerate(ns):
            for mi, f in enumerate(method_funcs):
                logger.info("Collecting result (%s, r=%d, n=%d)" %
                            (f.__name__, r, n))
                # let the aggregator finalize things
                aggregators[r, ni, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                job_result = aggregators[r, ni, mi].get_final_result().result
                job_results[r, ni, mi] = job_result

    # func_names = [f.__name__ for f in method_funcs]
    # func2labels = exglobal.get_func2label_map()
    # method_labels = [func2labels[f] for f in func_names if f in func2labels]

    # save results
    results = {'job_results': job_results,
               'mix_ratios': mix_ratios,
               'alpha': alpha, 'repeats': reps, 'ns': ns,
               'method_funcs': method_funcs, 'prob_label': prob_label,
               }

    # class name
    fname = 'ex%d-%s-me%d_rs%d_nmi%d_nma%d_a%.3f.p' \
        %(ex, prob_label, n_methods, reps, min(ns), max(ns), alpha,)

    glo.ex_save_result(ex, results, fname)
    logger.info('Saved aggregated results to %s' % fname)


def main():
    if len(sys.argv) != 2:
        print('Usage: %s problem_label' % sys.argv[0])
        sys.exit(1)
    prob_label = sys.argv[1]
    run_problem(prob_label)


if __name__ == '__main__':
    main()
