import autograd
import autograd.numpy as np

import scipy
from numpy.core.umath_tests import inner1d
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from kmod import data, kernel, util
from kmod.mctest import SC_UME
from kmod import log

import types
import functools


gpu_mode = True
gpu_id = 2
image_size = 64
model_input_size = 224
batch_size = 128


def set_gpu_mode(is_gpu):
    global gpu_mode
    gpu_mode = is_gpu


def set_gpu_id(gpu):
    global gpu_id
    gpu_id = gpu


def set_model_input_size(size):
    global model_input_size
    model_input_size = size


def set_batch_size(size):
    global batch_size
    batch_size = size


def optimize_3sample_criterion(datap, dataq, datar, gen_p, gen_q, model, Zp0,
                               Zq0, gwidth0, reg=1e-3, max_iter=100,
                               tol_fun=1e-6, disp=False, locs_bounds_frac=100,
                               gwidth_lb=None, gwidth_ub=None):
    """
    Similar to optimize_2sets_locs_widths() but constrain V=W and
    constrain the two kernels to be the same Gaussian kernel.
    Optimize one set of test locations and one Gaussian kernel width by
    maximizing the test power criterion of the UME *three*-sample test

    This optimization function is deterministic.

    Args:
        - datap: a kgof.data.Data from P (model 1)
        - dataq: a kgof.data.Data from Q (model 2)
        - datar: a kgof.data.Data from R (data generating distribution)
        - gen_p: pytorch model representing the generator p (model 1)
        - gen_q: pytorch model representing the generator q (model 2)
        - Zp0: Jxd_n numpy array. Initial value for the noise vectors of J locations.
           This is for model 1. 
        - Zq0: Jxd_n numpy array. Initial V containing J locations. For both
           This is for model 22. 
        - model: a feature extractor applied to generated images 
        - gwidth0: initial value of the Gaussian width^2 for both UME(P, R),
              and UME(Q, R)
        - reg: reg to add to the mean/sqrt(variance) criterion to become
            mean/sqrt(variance + reg)
        - max_iter: #gradient descent iterations
        - tol_fun: termination tolerance of the objective value
        - disp: True to print convergence messages
        - locs_bounds_frac: When making box bounds for the test_locs, extend
              the box defined by coordinate-wise min-max by std of each
              coordinate (of the aggregated data) multiplied by this number.
        - gwidth_lb: absolute lower bound on both the Gaussian width^2
        - gwidth_ub: absolute upper bound on both the Gaussian width^2

        If the lb, ub bounds are None, use fraction of the median heuristics
            to automatically set the bounds.
    Returns:
        - Z_opt: optimized noise vectors Z
        - gw_opt: optimized Gaussian width^2
        - opt_result: info from the optimization
    """
    J, dn = Zp0.shape
    Z0 = np.vstack([Zp0, Zq0])

    X, Y, Z = datap.data(), dataq.data(), datar.data()
    n, dp = X.shape

    global image_size

    def flatten(gwidth, V):
        return np.hstack((gwidth, V.reshape(-1)))

    def unflatten(x):
        sqrt_gwidth = x[0]
        V = np.reshape(x[1:], (2*J, -1))
        return sqrt_gwidth, V

    # Parameterize the Gaussian width with its square root (then square later)
    # to automatically enforce the positivity.
    def obj_feat_space(sqrt_gwidth, V):
        k = kernel.KGauss(sqrt_gwidth**2)
        return -SC_UME.power_criterion(datap, dataq, datar, k, k, V, V,
                                       reg=reg)

    def flat_obj_feat(x):
        sqrt_gwidth, V = unflatten(x)
        return obj_feat_space(sqrt_gwidth, V)

    def obj_noise_space(sqrt_gwidth, z):
        zp = z[:J]
        zq = z[J:]
        torch_zp = to_torch_variable(zp, shape=(-1, zp.shape[1], 1, 1))
        torch_zq = to_torch_variable(zq, shape=(-1, zq.shape[1], 1, 1))
        # need preprocessing probably
        global model_input_size
        s = model_input_size
        upsample = nn.Upsample(size=(s, s), mode='bilinear')
        fp = model(upsample(gen_p(torch_zp))).cpu().data.numpy()
        fp = fp.reshape((J, -1))
        fq = model(upsample(gen_q(torch_zq))).cpu().data.numpy()
        fq = fq.reshape((J, -1))
        F = np.vstack([fp, fq])
        return obj_feat_space(sqrt_gwidth, F)

    def flat_obj_noise(x):
        sqrt_gwidth, z = unflatten(x)
        return obj_noise_space(sqrt_gwidth, z)

    def grad_power_noise(x):
        """
        Compute the gradient of the power criterion with respect to the width of Gaussian
        RBF kernel and the noise vector.

        Args:
            x: 1 + 2J*d_n vector
        Returns:
            the gradient of the power criterion with respect to kernel width/latent vector
        """

        with util.ContextTimer() as t:
            width, z = unflatten(x)
            zp = z[:J]
            zq = z[J:]

            # Compute the Jacobian of the generators with respect to noise vector
            torch_zp = to_torch_variable(zp, shape=(-1, zp.shape[1], 1, 1),
                                         requires_grad=True)
            torch_zq = to_torch_variable(zq, shape=(-1, zq.shape[1], 1, 1),
                                         requires_grad=True)
            gp_grad = compute_jacobian(torch_zp, gen_p(torch_zp).view(J, -1))  # J x d_pix x d_noise x 1 x 1
            gq_grad = compute_jacobian(torch_zq, gen_q(torch_zq).view(J, -1))  # J x d_pix x d_noise x 1 x 1
            v_grad_z = np.vstack([gp_grad, gq_grad])
            v_grad_z = np.squeeze(v_grad_z, [3, 4])  # 2J x d_pix x d_noise
            
            # Compute the Jacobian of the feature extractor with respect to noise vector
            vp_flatten = to_torch_variable(
                gen_p(torch_zp).view(J, -1).cpu().data.numpy(),
                shape=(J, 3, image_size, image_size),
                requires_grad=True
            )
            vq_flatten = to_torch_variable(
                gen_q(torch_zq).view(J, -1).cpu().data.numpy(),
                shape=(J, 3, image_size, image_size),
                requires_grad=True
            )
            size = (model_input_size, model_input_size)
            upsample = nn.Upsample(size=size, mode='bilinear')
            fp = model(upsample(vp_flatten))
            fq = model(upsample(vq_flatten))
            fp_grad = compute_jacobian(vp_flatten, fp.view(J, -1))  # J x d_nn x C x H x W
            fq_grad = compute_jacobian(vq_flatten, fq.view(J, -1))  # J x d_nn x C x H x W
            f_grad_v = np.vstack([fp_grad, fq_grad])
            f_grad_v = f_grad_v.reshape((2*J, f_grad_v.shape[1], -1))  # 2J x d_nn x d_pix

            # Compute the gradient of the objective function with respect to
            # the gaussian width and test locations
            F = np.vstack([fp.cpu().data.numpy(), fq.cpu().data.numpy()])
            F = np.reshape(F, (2*J, -1))
            grad_obj = autograd.elementwise_grad(flat_obj_feat)  # 1+(2J)*d_nn input
            obj_grad_f = grad_obj(flatten(width, F))
            obj_grad_width = obj_grad_f[0]
            obj_grad_f = np.reshape(obj_grad_f[1:], [(2*J), -1])  # 2J x d_nn array

            obj_grad_v = inner1d(obj_grad_f, np.transpose(f_grad_v, (2, 0, 1)))  # 2J x d_pix
            obj_grad_z = inner1d(obj_grad_v.T, np.transpose(v_grad_z, (2, 0, 1))).flatten()

        return np.concatenate([obj_grad_width.reshape([1]), obj_grad_z]) 

    # Initial point
    x0 = flatten(np.sqrt(gwidth0), Z0)

    # make sure that the optimized gwidth is not too small or too large.
    XYZ = np.vstack((X, Y, Z))
    med2 = util.meddistance(XYZ, subsample=1000)**2
    fac_min = 1e-2
    fac_max = 1e2
    if gwidth_lb is None:
        gwidth_lb = max(fac_min*med2, 1e-3)
    if gwidth_ub is None:
        gwidth_ub = min(fac_max*med2, 1e5)

    # # Make a box to bound test locations
    # XYZ_std = np.std(XYZ, axis=0)
    # # XYZ_min: length-d array
    # XYZ_min = np.min(XYZ, axis=0)
    # XYZ_max = np.max(XYZ, axis=0)
    # # V_lb: 2J x dn
    # V_lb = np.tile(XYZ_min - locs_bounds_frac*XYZ_std, (2*J, 1))
    # V_ub = np.tile(XYZ_max + locs_bounds_frac*XYZ_std, (2*J, 1))
    # # (J*d+1) x 2. Take square root because we parameterize with the square
    # # root
    # x0_lb = np.hstack((np.sqrt(gwidth_lb), np.reshape(V_lb, -1)))
    # x0_ub = np.hstack((np.sqrt(gwidth_ub), np.reshape(V_ub, -1)))
    # #x0_bounds = list(zip(x0_lb, x0_ub))

    # Assuming noise coming uniform dist over unit cube
    x0_bounds = [(gwidth_lb, gwidth_ub)] + [(-1, 1)] * (2*J*dn)

    # optimize. Time the optimization as well.
    # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
    with util.ContextTimer() as timer:
        opt_result = scipy.optimize.minimize(
            flat_obj_noise, x0,
            method='L-BFGS-B', bounds=x0_bounds,
            tol=tol_fun,
            options={
                'maxiter': max_iter, 'ftol': tol_fun, 'disp': disp,
                'gtol': 1.0e-08,
            },
            jac=grad_power_noise
        )

    opt_result = dict(opt_result)
    opt_result['time_secs'] = timer.secs
    x_opt = opt_result['x']
    sq_gw_opt, Z_opt = unflatten(x_opt)
    gw_opt = sq_gw_opt**2

    assert util.is_real_num(gw_opt), 'gw_opt is not real. Was %s' % str(gw_opt)
    return Z_opt, gw_opt, opt_result


def to_torch_variable(a, shape=None, requires_grad=False):
    """Convert array a to torch.autograd.Variable"""
    if shape is None:
        shape = a.shape
    if gpu_mode:
        global gpu_id
        v = Variable(torch.from_numpy(a).float().view(shape).cuda(gpu_id),
                     requires_grad=requires_grad)
    else:
        v = Variable(torch.from_numpy(a), requires_grad=requires_grad)
        v = v.float().view(shape)
    return v


def apply_to_models(inputs, models, return_variable=False, requires_grad=False):
    """ Apply inputs to corresponding torch models (functions)
        If return_variable is True, the list of the torch variables corresponding to
        inputs is returned.
        If requires_grad is True, those variables are set to requires_grad=True.
    """
    if len(inputs) != len(models):
        raise ValueError('models and samples must have equal length')
    samples = []
    variables = []
    for i in range(len(inputs)):
        x = inputs[i]
        model = models[i]
        v = to_torch_variable(x, shape=(-1, x.shape[1], 1, 1), requires_grad=requires_grad)
        sample = model(v).cpu().data.numpy()
        sample = np.reshape(sample, [sample.shape[0], -1])
        # sample = np.clip(sample, 0, 1)
        samples.append(sample)
        variables.append(v)
    if return_variable:
        return samples, variables
    else:
        return samples


# modify this later
def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (Torch)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size, numpy array
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = np.zeros((num_classes,) + tuple(inputs.size()))
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        global gpu_id
        grad_output = grad_output.cuda(gpu_id)

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.cpu().data.numpy()

    # return torch.transpose(jacobian, dim0=0, dim1=1)
    s = tuple(range(len(jacobian.shape)))
    return np.transpose(jacobian, (1, 0,) + s[2:])


def kernel_feat_decorator_with(model):
    """Add an extra feature extracion with the given torch model"""

    def kernel_feat_decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            n, d = args[0].shape
            width = int((d/3)**0.5)
            X = to_torch_variable(args[0].reshape((n, 3, width, width)))
            n, d = args[1].shape
            width = int((d/3)**0.5)
            # print(n, d, width)
            Y = to_torch_variable(args[1].reshape((n, 3, width, width)))
            X_ = model(X).cpu().data.numpy()
            X_ = X_.reshape((X_.shape[0], -1))
            Y_ = model(Y).cpu().data.numpy()
            Y_ = Y_.reshape((Y_.shape[0], -1))
            return func(X_, Y_, *args[2:])
        return new_func

    return kernel_feat_decorator


def decorate_all_methods_with(decorator):
    """decorate a class with a given decorator"""

    def decorate_all_methods(Cls):

        class NewCls(object):

            def __init__(self, *args, **kwargs):
                self.decorated_instance = Cls(*args, **kwargs)

            def __getattribute__(self, s):
                """
                Applying the decorator to all the methods in the class.
                Methods not in object are accessed after applying the
                decorator.
                """
                try:
                    x = super(NewCls, self).__getattribute__(s)
                except AttributeError:
                    pass
                else:
                    return x
                x = self.decorated_instance.__getattribute__(s)
                if isinstance(x, types.MethodType):
                    return decorator(x)
                else:
                    return x
        return NewCls
    return decorate_all_methods


def extract_feats(X, model, upsample=False):
    """
    Extract features using model. 
    Args:
        - X: an nxd numpy array representing a set of RGB images
        - model: a pytorch model
    Returns:
        - feat_X: an nxd' numpy array representing extracted features
        of the dimenstionality d'
    """
    global batch_size
    n = X.shape[0]
    width = int((X.size / (3 * n))**0.5)
    X = X.reshape((n, 3, width, width))
    # print('X.shape: {}'.format(X.shape))
    feat_X = []

    for i in range(0, n, batch_size):
        V = X[i: i+batch_size]
        V_ = to_torch_variable(V)
        if upsample:
            global model_input_size
            m = nn.Upsample((model_input_size, model_input_size),
                            mode='bilinear')
            V_ = m(V_)
        fX = model(V_).cpu().data.numpy()
        fX = fX.reshape((fX.shape[0], -1))
        # print('fX.shape: {}'.format(fX.shape))
        feat_X.append(fX)
    feat_X = np.vstack(feat_X)
    return feat_X


def opt_greedy_3sample_criterion(datap, dataq, datar, model, locs,
                                 gwidth, J, reg=1e-3, maximize=True,
                                 extract_feature=True):
    """
    Obtains a set of J test locations by maximizing (or minimizing)
    the power criterion of the UME three-sample test.
    The test locations are given by choosing a subset from given
    candidate locations by the greedy forward selection.

    Args:
        - datap: a kgof.data.Data from P (model 1)
        - dataq: a kgof.data.Data from Q (model 2)
        - datar: a kgof.data.Data from R (data generating distribution)
        - model: a torch model used for feature extraction
        - locs: an n_c x d numpy array representing a set of n_c candidate locations
        - gwidth: the Gaussian width^2 for both UME(P, R) and UME(Q, R)
        - reg: reg to add to the mean/sqrt(variance) criterion to become
            mean/sqrt(variance + reg)
        - maximize: if True, maximize the power criterion, otherwise minimize
        - extract_feature: if True, the data is tranformed by the given feature
          extractor model

    Returns:
        A set of indices representing obtinaed locations
    """

    # transform inputs to power criterion with feature extractor
    if extract_feature:
        dp = extract_feats(datap.data(), model)
        dp = data.Data(dp)
        dq = extract_feats(dataq.data(), model)
        dq = data.Data(dq)
        dr = extract_feats(datar.data(), model)
        dr = data.Data(dr)
        fV = extract_feats(locs, model)
    else:
        dp = datap
        dq = dataq
        dr = datar
        fV = locs

    def obj(V):
        k = kernel.KGauss(gwidth)
        if len(V.shape) < 2:
            V = V.reshape((-1,) + V.shape)
        if maximize:
            return SC_UME.power_criterion(dp, dq, dr, k, k, V, V,
                                          reg=reg)
        else:
            return -SC_UME.power_criterion(dp, dq, dr, k, k, V, V,
                                           reg=reg)

    def greedy_search(num_locs, loc_pool):
        best_loc_idx = []
        n = loc_pool.shape[0]
        for _ in range(num_locs):
            is_empty = (len(best_loc_idx) == 0)
            if not is_empty:
                best_locs = np.vstack(loc_pool[best_loc_idx])
            max_val = None
            for k in range(n):
                if k not in best_loc_idx:
                    if is_empty:
                        V = loc_pool[k].reshape((1, -1))
                    else:
                        V = np.vstack([loc_pool[k], best_locs])
                    score = obj(V)
                    if max_val is None:
                        max_val = score
                        best_idx = k
                    else:
                        if score > max_val:
                            max_val = score
                            best_idx = k
            best_loc_idx.append(best_idx)
        return best_loc_idx

    return greedy_search(J, fV)


def ume_ustat_h1_mean_variance(feature_matrix, return_variance=True, 
                               use_unbiased=True):
    """
    Compute the mean and variance of the asymptotic normal distribution
    under H1 of the test statistic. The mean converges to a constant as
    n->\infty.

    feature_matrix: n x J feature matrix
    return_variance: If false, avoid computing and returning the variance.
    use_unbiased: If True, use the unbiased version of the mean. Can be
        negative.

    Return the mean [and the variance]
    """
    Z = feature_matrix
    n = Z.size(0)
    J = Z.size(1)
    assert n > 1, 'Need n > 1 to compute the mean of the statistic.'
    if use_unbiased:
        # t1 = np.sum(np.mean(Z, axis=0)**2)*(n/float(n-1))
        t1 = torch.sum(torch.mean(Z, dim=0)**2).mul(n).div(n-1)
        # t2 = np.mean(np.sum(Z**2, axis=1))/float(n-1)
        t2 = torch.mean(torch.sum(Z**2, dim=1)).div(n-1)
        mean_h1 = t1 - t2
    else:
        # mean_h1 = np.sum(np.mean(Z, axis=0)**2)
        mean_h1 = torch.sum(torch.mean(Z, dim=0)**2)

    if return_variance:
        # compute the variance 
        # mu = np.mean(Z, axis=0)  # length-J vector
        mu = torch.mean(Z, dim=0)  # length-J vector
        # variance = 4.0*np.mean(np.dot(Z, mu)**2) - 4.0*np.sum(mu**2)**2
        variance = 4.0*torch.mean(torch.matmul(Z, mu)**2) - 4.0*torch.sum(mu**2)**2
        return mean_h1, variance
    else:
        return mean_h1


def ume_feature_matrix(self, X, Y, V, k):
    J = V.size(0)

    # n x J feature matrix
    g = k.eval(X, V) / np.sqrt(J)
    h = k.eval(Y, V) / np.sqrt(J)
    Z = g - h
    return Z


def run_optimize_3sample_criterion(datap, dataq, datar, gen_p, gen_q, model, Zp0,
                                   Zq0, gwidth0, reg=1e-3, max_iter=100,
                                   tol_fun=1e-6, disp=False, locs_bounds_frac=100,
                                   gwidth_lb=None, gwidth_ub=None, cuda=None):

    def power_criterion(X, Y, Z, Vp, Vq, k, reg):
        fea_pr = ume_feature_matrix(X, Z, Vp, k)  # n x Jp
        fea_qr = ume_feature_matrix(Y, Z, Vq, k)  # n x Jq
        umehp, var_pr = ume_ustat_h1_mean_variance(fea_pr, return_variance=True,
                                                   use_unbiased=True)
        umehq, var_qr = ume_ustat_h1_mean_variance(fea_qr, return_variance=True,
                                                   use_unbiased=True)

        if (var_pr <= 0).any():
            log.l().warning('Non-positive var_pr detected. Was {}'.format(var_pr))
        if (var_qr <= 0).any():
            log.l().warning('Non-positive var_qr detected. War {}'.format(var_qr))
        # assert var_pr > 0, 'var_pr was {}'.format(var_pr)
        # assert var_qr > 0, 'var_qr was {}'.format(var_qr)
        mean_h1 = umehp - umehq

        if not return_variance:
            return mean_h1

        # mean features
        mean_pr = torch.mean(fea_pr, dim=0)
        mean_qr = torch.mean(fea_qr, dim=0)
        t1 = 4.0*torch.mean(torch.matmul(fea_pr, mean_pr)*torch.matmul(fea_qr, mean_qr))
        t2 = 4.0*torch.sum(mean_pr**2)*torch.sum(mean_qr**2)

        # compute the cross-covariance
        var_pqr = t1 - t2
        var_h1 = var_pr - 2.0*var_pqr + var_qr

        power_criterion = mean_h1 / torch.sqrt(var_h1 + reg)
        return power_criterion

    gwidth = to_torch_variable(gwidth0, requires_grad=True)
    Zp = to_torch_variable(Zp0, requires_grad=True)
    Zq = to_torch_variable(Zq0, requires_grad=True)

    X = to_torch_variable(datap.data(), requires_grad=False)
    Y = to_torch_variable(dataq.data(), requires_grad=False)
    Z = to_torch_variable(datar.data(), requires_grad=False)

    optimizer = optim.LBFGS([gwidth, Zp, Zq], lr=1e-2, max_iter=max_iterk)
    transform = nn.Upsample((model_input_size, model_input_size),
                            mode='bilinear')

    k = kernel.PTKGauss(gwidth**2)
    for i in range(max_iter):
        def closure():
            optimizer.zero_grad()
            im_p = gen_p(Zp)
            im_q = gen_q(Zq)
            Vp = model(transform(im_p))
            Vq = model(transform(im_q))
            obj = -power_criterion(X, Y, Z, Vp, Vq, k, reg)
            obj.backward()
            return obj
        optimizer.step(closure)

    return Zp, Zq, gwidth
