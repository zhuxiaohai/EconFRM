import numpy as np
from econml.utilities import filter_none_kwargs, reshape, check_input_arrays


def get_pred_y(estimator, X=None, W=None, Z=None):
    """
    Score the fitted CATE model on a new data set. Generates nuisance parameters
    for the new data set based on the fitted nuisance models created at fit time.
    It uses the mean prediction of the models fitted by the different crossfit folds
    under different iterations. Then calls the score function of the model_final and
    returns the calculated score. The model_final model must have a score method.

    If model_final does not have a score method, then it raises an :exc:`.AttributeError`

    Parameters
    ----------
    Y: (n, d_y) matrix or vector of length n
        Outcomes for each sample
    T: (n, d_t) matrix or vector of length n
        Treatments for each sample
    X: optional (n, d_x) matrix or None (Default=None)
        Features for each sample
    W: optional (n, d_w) matrix or None (Default=None)
        Controls for each sample
    Z: optional (n, d_z) matrix or None (Default=None)
        Instruments for each sample
    sample_weight: optional(n,) vector or None (Default=None)
        Weights for each samples
    groups: (n,) vector, optional
        All rows corresponding to the same group will be kept together during splitting.

    Returns
    -------
    score : float or (array of float)
        The score of the final CATE model on the new data. Same type as the return
        type of the model_final.score method.
    """
    if not hasattr(estimator._ortho_learner_model_final, 'score'):
        raise AttributeError("Final model does not have a score method!")
    X, W, Z = check_input_arrays(X, W, Z)
    estimator._check_fitted_dims(X)
    estimator._check_fitted_dims_w_z(W, Z)
    if estimator.z_transformer is not None:
        Z = estimator.z_transformer.transform(reshape(Z, (-1, 1)))
    n_iters = len(estimator._models_nuisance)
    n_splits = len(estimator._models_nuisance[0])

    # for each mc iteration
    for i, models_nuisances in enumerate(estimator._models_nuisance):
        # for each model under cross fit setting
        for j, mdl in enumerate(models_nuisances):
            predy_temp = mdl._model_y.predict(X, W)
            if not isinstance(predy_temp, tuple):
                predy_temp = (predy_temp,)

            if i == 0 and j == 0:
                pred_y_output = [np.zeros((n_iters * n_splits,) + predy.shape) for predy in predy_temp]

            for it, predy in enumerate(predy_temp):
                pred_y_output[it][i * n_iters + j] = predy

    for it in range(len(pred_y_output)):
        pred_y_output[it] = np.mean(pred_y_output[it], axis=0)
        
    return pred_y_output


def get_pred_t(estimator, X=None, W=None, Z=None):
    """
    Score the fitted CATE model on a new data set. Generates nuisance parameters
    for the new data set based on the fitted nuisance models created at fit time.
    It uses the mean prediction of the models fitted by the different crossfit folds
    under different iterations. Then calls the score function of the model_final and
    returns the calculated score. The model_final model must have a score method.

    If model_final does not have a score method, then it raises an :exc:`.AttributeError`

    Parameters
    ----------
    Y: (n, d_y) matrix or vector of length n
        Outcomes for each sample
    T: (n, d_t) matrix or vector of length n
        Treatments for each sample
    X: optional (n, d_x) matrix or None (Default=None)
        Features for each sample
    W: optional (n, d_w) matrix or None (Default=None)
        Controls for each sample
    Z: optional (n, d_z) matrix or None (Default=None)
        Instruments for each sample
    sample_weight: optional(n,) vector or None (Default=None)
        Weights for each samples
    groups: (n,) vector, optional
        All rows corresponding to the same group will be kept together during splitting.

    Returns
    -------
    score : float or (array of float)
        The score of the final CATE model on the new data. Same type as the return
        type of the model_final.score method.
    """
    if not hasattr(estimator._ortho_learner_model_final, 'score'):
        raise AttributeError("Final model does not have a score method!")
    X, W, Z = check_input_arrays(X, W, Z)
    estimator._check_fitted_dims(X)
    estimator._check_fitted_dims_w_z(W, Z)
    if estimator.z_transformer is not None:
        Z = estimator.z_transformer.transform(reshape(Z, (-1, 1)))
    n_iters = len(estimator._models_nuisance)
    n_splits = len(estimator._models_nuisance[0])

    # for each mc iteration
    for i, models_nuisances in enumerate(estimator._models_nuisance):
        # for each model under cross fit setting
        for j, mdl in enumerate(models_nuisances):
            predt_temp = mdl._model_t.predict(X, W)
            if not isinstance(predt_temp, tuple):
                predt_temp = (predt_temp,)

            if i == 0 and j == 0:
                pred_t_output = [np.zeros((n_iters * n_splits,) + predt.shape) for predt in predt_temp]

            for it, predt in enumerate(predt_temp):
                pred_t_output[it][i * n_iters + j] = predt

    for it in range(len(pred_t_output)):
        pred_t_output[it] = np.mean(pred_t_output[it], axis=0)
        
    return pred_t_output


def get_nuisances(estimator, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
    """
    Score the fitted CATE model on a new data set. Generates nuisance parameters
    for the new data set based on the fitted nuisance models created at fit time.
    It uses the mean prediction of the models fitted by the different crossfit folds
    under different iterations. Then calls the score function of the model_final and
    returns the calculated score. The model_final model must have a score method.

    If model_final does not have a score method, then it raises an :exc:`.AttributeError`

    Parameters
    ----------
    Y: (n, d_y) matrix or vector of length n
        Outcomes for each sample
    T: (n, d_t) matrix or vector of length n
        Treatments for each sample
    X: optional (n, d_x) matrix or None (Default=None)
        Features for each sample
    W: optional (n, d_w) matrix or None (Default=None)
        Controls for each sample
    Z: optional (n, d_z) matrix or None (Default=None)
        Instruments for each sample
    sample_weight: optional(n,) vector or None (Default=None)
        Weights for each samples
    groups: (n,) vector, optional
        All rows corresponding to the same group will be kept together during splitting.

    Returns
    -------
    score : float or (array of float)
        The score of the final CATE model on the new data. Same type as the return
        type of the model_final.score method.
    """
    if not hasattr(estimator._ortho_learner_model_final, 'score'):
        raise AttributeError("Final model does not have a score method!")
    Y, T, X, W, Z = check_input_arrays(Y, T, X, W, Z)
    estimator._check_fitted_dims(X)
    estimator._check_fitted_dims_w_z(W, Z)
    X, T = estimator._expand_treatments(X, T)
    if estimator.z_transformer is not None:
        Z = estimator.z_transformer.transform(reshape(Z, (-1, 1)))
    n_iters = len(estimator._models_nuisance)
    n_splits = len(estimator._models_nuisance[0])

    # for each mc iteration
    for i, models_nuisances in enumerate(estimator._models_nuisance):
        # for each model under cross fit setting
        for j, mdl in enumerate(models_nuisances):
            nuisance_temp = mdl.predict(Y, T, **filter_none_kwargs(X=X, W=W, Z=Z, groups=groups))
            if not isinstance(nuisance_temp, tuple):
                nuisance_temp = (nuisance_temp,)

            if i == 0 and j == 0:
                nuisances = [np.zeros((n_iters * n_splits,) + nuis.shape) for nuis in nuisance_temp]

            for it, nuis in enumerate(nuisance_temp):
                nuisances[it][i * n_iters + j] = nuis

    for it in range(len(nuisances)):
        nuisances[it] = np.mean(nuisances[it], axis=0)

    return nuisances