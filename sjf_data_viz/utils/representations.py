from sklearn.manifold._t_sne import TSNE, _kl_divergence, _kl_divergence_bh, _openmp_effective_n_threads, \
    _gradient_descent

class JumpingTSNE(TSNE):

    def __init__(self, jump_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jump_size = jump_size
        self.X_embedded_jumps = []
        self.momentum_jumps = []

    def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded,
              neighbors=None, skip_num_points=0):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5,
        }
        if self.method == 'barnes_hut':
            obj_func = _kl_divergence_bh
            opt_args['kwargs']['angle'] = self.angle
            # Repeat verbose argument for _kl_divergence_bh
            opt_args['kwargs']['verbose'] = self.verbose
            # Get the number of threads for gradient computation here to
            # avoid recomputing it at each iteration.
            opt_args['kwargs']['num_threads'] = _openmp_effective_n_threads()
        else:
            obj_func = _kl_divergence

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        P *= self.early_exaggeration
        self.jump_size = self.jump_size//2
        params, kl_divergence, it = self.jumped_gradient_descent(obj_func, params,
                                                                 **opt_args)
        self.jump_size = self.jump_size*2
        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early "
                  "exaggeration: %f" % (it + 1, kl_divergence))

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args['n_iter'] = self.n_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = self.n_iter_without_progress
            params, kl_divergence, it = self.jumped_gradient_descent(obj_func, params,
                                                                     **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations: %f"
                  % (it + 1, kl_divergence))

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded


    def jumped_gradient_descent(self, obj_func, params, **opt_args):
        it = opt_args['it'] - 1

        n_jump_without_progress = opt_args["n_iter_without_progress"] // self.jump_size + 1
        remaining = opt_args['n_iter'] - it
        n_jumps = remaining // self.jump_size + 1
        ct_no_improvement = 0
        kl_divergence_best = None
        
        new_opt_args = opt_args.copy()
        for jump in range(n_jumps):
            new_opt_args['it'] = it + 1
            new_opt_args['n_iter'] = it + self.jump_size
            params, kl_divergence, it = _gradient_descent(obj_func, params,
                                                          **new_opt_args)
            self.X_embedded_jumps.append(params.reshape(-1, self.n_components))
            self.momentum_jumps.append(opt_args['momentum'])
            if jump > 0:
                print("\rJump {}/{}: best_kl={:.6f}\t current_kl={:.6f}".format(jump + 1, n_jumps,
                                                                                kl_divergence_best,
                                                                                kl_divergence), end="", flush=True)
            if kl_divergence_best is None or kl_divergence < kl_divergence_best:
                kl_divergence_best = kl_divergence
                ct_no_improvement = 0
            else:
                ct_no_improvement += 1
                if ct_no_improvement >= n_jump_without_progress:
                    break
        print()
        return params, kl_divergence, it