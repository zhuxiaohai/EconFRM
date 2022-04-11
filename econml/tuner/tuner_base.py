import optuna
import matplotlib.pyplot as plt
from sklearn.base import clone


class Objective(object):
    def __init__(self, estimator, tuning_param_dict, coef_train_val_disparity,
                 train_kargs, val_kargs):
        self.estimator = estimator
        self.tuning_param_dict = tuning_param_dict
        self.coef_train_val_disparity = coef_train_val_disparity
        self.train_kargs = train_kargs
        self.val_kargs = val_kargs

    def train_val_score(self, train_score, val_score, w):
        output_scores = val_score - abs(train_score - val_score) * w
        return output_scores

    def __call__(self, trial):
        trial_param_dict = {}
        param = self.tuning_param_dict.get('booster')
        if isinstance(param, tuple):
            suggest_type = param[0]
            suggest_param = param[1]
            trial_param_dict['booster'] = eval('trial.suggest_' + suggest_type)('booster', **suggest_param)
        elif param is not None:
            trial_param_dict['booster'] = param
        booster = trial_param_dict.get('booster')
        for key, param in self.tuning_param_dict.items():
            if key == 'booster':
                continue
            if (booster is None) or booster == 'gbtree':
                if key in ['sample_type', 'normalize_type', 'one_drop', 'rate_drop', 'skip_drop']:
                    continue
            if isinstance(param, tuple):
                suggest_type = param[0]
                suggest_param = param[1]
                trial_param_dict[key] = eval('trial.suggest_' + suggest_type)(key, **suggest_param)
            else:
                trial_param_dict[key] = param
        learner = clone(self.estimator, safe=False)
        for key, value in trial_param_dict.items():
            setattr(learner, key, value)
        # learner = self.estimator(**trial_param_dict)
        learner.fit(**self.train_kargs)
        val_score = learner.score(**self.val_kargs)
        trial.set_user_attr("val_score", val_score)
        if self.coef_train_val_disparity > 0:
            train_score = learner.score(**self.train_kargs)
            trial.set_user_attr("train_score", train_score)
            best_score = self.train_val_score(train_score, val_score, self.coef_train_val_disparity)
        else:
            best_score = val_score
        return best_score


class OptunaSearch(object):
    def __init__(self, optimization_direction='maximize', coef_train_val_disparity=0.2,
                 n_startup_trials=20, n_warmup_steps=20, interval_steps=1,
                 pruning_percentile=75, maximum_time=60*10, n_trials=100,
                 random_state=2, optuna_verbosity=1, optuna_njobs=-1):
        self.optimization_direction = optimization_direction
        self.coef_train_val_disparity = coef_train_val_disparity
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.interval_steps = interval_steps
        self.pruning_percentile = pruning_percentile
        self.maximum_time = maximum_time
        self.n_trials = n_trials
        self.random_state = random_state
        self.optuna_verbosity = optuna_verbosity
        self.optuna_njobs = optuna_njobs
        self.study = None
        self.dynamic_params = {}
        self.static_params = {}
        self.fobj_dynamic_params = {}
        self.fobj_static_params = {}

    def get_params(self):
        """
        how to use the best params returned by this function:
        train_param = instance.get_params()
        model = xgb.train(train_param, train_dmatrix, num_boost_round=train_param['n_iterations'])
        test_probability_1d_array = model.predict(test_dmatrix)
        """
        if self.study:
            best_trial = self.study.best_trial
            best_param = best_trial.params

            output_param = {key: best_param[key] for key in best_param.keys() if key in self.dynamic_params}
            output_param.update(self.static_params)

            output_fobj_param = {key: best_param[key] for key in best_param.keys() if key in self.fobj_dynamic_params}
            output_fobj_param.update(self.fobj_static_params)

            return output_param, output_fobj_param, best_trial.user_attrs
        else:
            return None

    def plot_optimization(self):
        if self.study:
            return optuna.visualization.plot_optimization_history(self.study)

    def plot_score(self):
        if self.study:
            trial_df = self.study.trials_dataframe()
            _, ax1 = plt.subplots()
            ax1.plot(trial_df.index,
                     trial_df.user_attrs_train_score,
                     label='train')
            ax1.plot(trial_df.index,
                     trial_df.user_attrs_val_score,
                     label='val')
            plt.legend()
            plt.show()

    def plot_importance(self, names=None):
        if self.study:
            return optuna.visualization.plot_param_importances(self.study, params=names)

    def search(self, estimator, params, train_kargs, val_kargs):
        for key, param in params.items():
            if not isinstance(param, tuple):
                self.static_params[key] = param
            else:
                self.dynamic_params[key] = param

        objective = Objective(estimator, params, self.coef_train_val_disparity, train_kargs, val_kargs)

        if self.optuna_verbosity == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        # prune a step(a boosting round) if it's worse than the bottom (1 - percentile) in history
        pruner = optuna.pruners.PercentilePruner(percentile=self.pruning_percentile,
                                                 n_warmup_steps=self.n_warmup_steps,
                                                 interval_steps=self.interval_steps,
                                                 n_startup_trials=self.n_startup_trials)
        study = optuna.create_study(direction=self.optimization_direction, sampler=sampler, pruner=pruner)
        study.optimize(objective, timeout=self.maximum_time, n_trials=self.n_trials, n_jobs=self.optuna_njobs)
        self.study = study
        print("Number of finished trials: ", len(study.trials))